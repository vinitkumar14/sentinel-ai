"""
SENTINEL-AI Redis Integration
================================
Async Redis client for:
  1. Pub/sub event broadcasting (multi-node deployments)
  2. Caching recent incidents and statistics
  3. Distributed IP block state synchronization
  4. Real-time dashboard websocket data feed

Falls back gracefully to in-memory state when Redis is unavailable.

Usage:
    redis_client = RedisClient(config)
    await redis_client.connect()

    # Publish a threat event
    await redis_client.publish_threat(prediction_result, threat_score)

    # Cache system stats
    await redis_client.cache_stats(stats_dict)

    # Subscribe to events (for multi-node dashboard)
    async for event in redis_client.subscribe("sentinel:threats"):
        process(event)
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from sentinel_ai.core.logger import get_logger

if TYPE_CHECKING:
    from sentinel_ai.ai.ensemble_predictor import PredictionResult
    from sentinel_ai.core.config import SentinelConfig

log = get_logger(__name__)

# Redis channel names
CHANNEL_THREATS = "sentinel:threats"
CHANNEL_BLOCKS = "sentinel:blocks"
CHANNEL_STATS = "sentinel:stats"
CHANNEL_DRIFT = "sentinel:drift"
CHANNEL_ALERTS = "sentinel:alerts"


class RedisClient:
    """
    Async Redis wrapper for SENTINEL-AI inter-process communication.

    Args:
        config: Loaded SentinelConfig.
    """

    def __init__(self, config: "SentinelConfig") -> None:
        self._config = config
        self._redis = None
        self._pubsub = None
        self._connected: bool = False
        self._enabled: bool = config.redis.enabled

    async def connect(self) -> bool:
        """
        Connect to Redis. Returns True on success, False if unavailable.
        Failure is non-fatal — system operates in single-node mode.
        """
        if not self._enabled:
            log.debug("Redis disabled in config")
            return False

        try:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
            self._redis = await aioredis.from_url(
                f"redis://{self._config.redis.host}:{self._config.redis.port}",
                db=self._config.redis.db,
                password=self._config.redis.password,
                max_connections=self._config.redis.max_connections,
                decode_responses=True,
            )
            # Ping to verify connection
            await self._redis.ping()
            self._connected = True
            log.info(
                "Redis connected",
                host=self._config.redis.host,
                port=self._config.redis.port,
            )
            return True

        except Exception as exc:
            log.warning(
                "Redis unavailable — operating in single-node mode",
                error=str(exc),
            )
            self._connected = False
            self._redis = None
            return False

    async def disconnect(self) -> None:
        """Close the Redis connection."""
        if self._redis:
            await self._redis.close()
            self._connected = False
            log.info("Redis disconnected")

    # ── Pub/Sub ───────────────────────────────────────────────────────────────

    async def publish_threat(
        self,
        result: "PredictionResult",
        threat_score: float,
        severity: str,
    ) -> None:
        """Publish a threat detection event to the threats channel."""
        if not self._connected:
            return
        try:
            payload = json.dumps({
                "event": "threat_detected",
                "attack_type": result.predicted_class,
                "src_ip": result.src_ip,
                "dst_ip": result.dst_ip,
                "confidence": round(result.confidence, 4),
                "threat_score": round(threat_score, 2),
                "severity": severity,
                "flow_id": result.flow_id,
                "timestamp": result.timestamp,
            })
            await self._redis.publish(CHANNEL_THREATS, payload)
        except Exception as exc:
            log.debug("Redis publish error", error=str(exc))

    async def publish_block(self, ip: str, reason: str, dry_run: bool) -> None:
        """Publish an IP block event."""
        if not self._connected:
            return
        try:
            payload = json.dumps({
                "event": "ip_blocked",
                "ip": ip,
                "reason": reason,
                "dry_run": dry_run,
                "timestamp": time.time(),
            })
            await self._redis.publish(CHANNEL_BLOCKS, payload)
        except Exception as exc:
            log.debug("Redis publish error", error=str(exc))

    async def publish_drift(self, drift_count: int, indicator: float) -> None:
        """Publish a drift detection event."""
        if not self._connected:
            return
        try:
            payload = json.dumps({
                "event": "drift_detected",
                "drift_count": drift_count,
                "indicator": indicator,
                "timestamp": time.time(),
            })
            await self._redis.publish(CHANNEL_DRIFT, payload)
        except Exception as exc:
            log.debug("Redis publish error", error=str(exc))

    async def subscribe_threats(self) -> AsyncIterator[dict]:
        """
        Async generator that yields threat events from Redis pub/sub.
        Use in multi-node deployments where other nodes publish events.
        """
        if not self._connected:
            return
        try:
            pubsub = self._redis.pubsub()
            await pubsub.subscribe(CHANNEL_THREATS, CHANNEL_BLOCKS, CHANNEL_DRIFT)
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        yield json.loads(message["data"])
                    except json.JSONDecodeError:
                        pass
        except Exception as exc:
            log.debug("Redis subscribe error", error=str(exc))

    # ── Caching ───────────────────────────────────────────────────────────────

    async def cache_stats(self, stats: dict[str, Any], ttl: int = 60) -> None:
        """Cache system stats in Redis with a TTL."""
        if not self._connected:
            return
        try:
            await self._redis.set(
                "sentinel:stats",
                json.dumps(stats),
                ex=ttl,
            )
        except Exception as exc:
            log.debug("Redis cache error", error=str(exc))

    async def get_cached_stats(self) -> Optional[dict]:
        """Retrieve cached stats from Redis."""
        if not self._connected:
            return None
        try:
            data = await self._redis.get("sentinel:stats")
            return json.loads(data) if data else None
        except Exception:
            return None

    async def cache_blocked_ips(self, ip_list: list[str]) -> None:
        """Sync blocked IPs to Redis for multi-node consistency."""
        if not self._connected:
            return
        try:
            pipe = self._redis.pipeline()
            pipe.delete("sentinel:blocked_ips")
            if ip_list:
                pipe.sadd("sentinel:blocked_ips", *ip_list)
            await pipe.execute()
        except Exception as exc:
            log.debug("Redis sync error", error=str(exc))

    async def get_blocked_ips(self) -> set[str]:
        """Get blocked IPs from Redis (for multi-node sync)."""
        if not self._connected:
            return set()
        try:
            members = await self._redis.smembers("sentinel:blocked_ips")
            return set(members)
        except Exception:
            return set()

    async def increment_counter(self, key: str, ttl: int = 3600) -> int:
        """Atomic counter increment (useful for rate limiting)."""
        if not self._connected:
            return 0
        try:
            pipe = self._redis.pipeline()
            pipe.incr(f"sentinel:{key}")
            pipe.expire(f"sentinel:{key}", ttl)
            results = await pipe.execute()
            return int(results[0])
        except Exception:
            return 0

    @property
    def is_connected(self) -> bool:
        return self._connected
