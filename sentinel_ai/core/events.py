"""
SENTINEL-AI Async Event Bus
=============================
A lightweight, in-process publish/subscribe system using asyncio queues.

Architecture:
  - Publishers emit SentinelEvent objects to named topics
  - Subscribers register async callbacks for specific EventTypes
  - The EventBus dispatches events to all matching subscribers concurrently
  - Bounded queues prevent memory exhaustion under high event rates

Usage:
    bus = EventBus()

    @bus.subscribe(EventType.THREAT_DETECTED)
    async def on_threat(event: SentinelEvent) -> None:
        print(f"Threat: {event.payload}")

    await bus.publish(SentinelEvent(
        type=EventType.THREAT_DETECTED,
        payload={"attack": "ddos", "src": "1.2.3.4"}
    ))

    await bus.start()   # starts dispatch loop
    await bus.stop()    # graceful shutdown
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Optional
from uuid import uuid4

from sentinel_ai.core.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Event types
# ─────────────────────────────────────────────────────────────────────────────


class EventType(Enum):
    """
    All event types that can flow through the SENTINEL-AI event bus.

    Each module publishes and/or subscribes to these event types to
    decouple components without direct method calls.
    """

    # ── Capture events ────────────────────────────────────────────────────────
    PACKET_CAPTURED = auto()       # A raw packet was captured from the wire
    FLOW_CREATED = auto()          # A new network flow was created
    FLOW_UPDATED = auto()          # An existing flow was updated with new packets
    FLOW_COMPLETED = auto()        # A flow timed out / reached max packets → ready for AI
    FLOW_FEATURES_READY = auto()   # Feature vector extracted → ready for inference

    # ── AI events ─────────────────────────────────────────────────────────────
    PREDICTION_MADE = auto()       # AI model made a prediction on a flow
    THREAT_DETECTED = auto()       # Flow classified as a threat (non-benign)
    ANOMALY_DETECTED = auto()      # Isolation Forest detected an anomaly
    DRIFT_DETECTED = auto()        # Concept drift detected in incoming data
    MODEL_UPDATED = auto()         # River online model updated
    MODEL_LOADED = auto()          # A model was loaded/reloaded

    # ── Defense events ────────────────────────────────────────────────────────
    IP_BLOCKED = auto()            # An IP was blocked (or simulated block in observe mode)
    IP_UNBLOCKED = auto()          # A block expired and IP was unblocked
    RATE_LIMITED = auto()          # An IP was rate-limited
    ALERT_SENT = auto()            # An alert notification was dispatched

    # ── System events ─────────────────────────────────────────────────────────
    SYSTEM_STARTED = auto()        # System fully initialized
    SYSTEM_SHUTDOWN = auto()       # Graceful shutdown initiated
    CONFIG_RELOADED = auto()       # Configuration hot-reloaded
    STATS_UPDATED = auto()         # Periodic statistics snapshot emitted


# ─────────────────────────────────────────────────────────────────────────────
# Event dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class SentinelEvent:
    """
    An event that flows through the SENTINEL-AI event bus.

    Attributes:
        type:       The EventType that identifies this event's topic.
        payload:    Arbitrary dict of event-specific data.
        event_id:   Unique identifier for deduplication/correlation.
        timestamp:  Unix epoch timestamp when the event was created.
        source:     Module/component that emitted this event.
    """

    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Event Bus
# ─────────────────────────────────────────────────────────────────────────────

# Type alias for async event handlers
EventHandler = Callable[[SentinelEvent], Coroutine[Any, Any, None]]


class EventBus:
    """
    Async publish/subscribe event bus.

    Thread-safety: designed for single-threaded asyncio use.
    For cross-thread publishing, use `publish_threadsafe()`.

    Args:
        queue_size: Maximum number of pending events in the internal queue.
                    Once full, new events are dropped with a warning.
    """

    def __init__(self, queue_size: int = 10_000) -> None:
        self._queue: asyncio.Queue[SentinelEvent] = asyncio.Queue(
            maxsize=queue_size
        )
        self._subscribers: dict[EventType, list[EventHandler]] = {}
        self._running: bool = False
        self._dispatch_task: Optional[asyncio.Task[None]] = None
        self._stats: dict[str, int] = {
            "published": 0,
            "dispatched": 0,
            "dropped": 0,
            "handler_errors": 0,
        }

    # ── Subscription ─────────────────────────────────────────────────────────

    def subscribe(
        self, event_type: EventType
    ) -> Callable[[EventHandler], EventHandler]:
        """
        Decorator to register an async function as a subscriber.

        Usage:
            @bus.subscribe(EventType.THREAT_DETECTED)
            async def handle_threat(event: SentinelEvent) -> None:
                ...
        """

        def decorator(handler: EventHandler) -> EventHandler:
            self._subscribers.setdefault(event_type, []).append(handler)
            log.debug(
                "Subscriber registered",
                event_type=event_type.name,
                handler=handler.__name__,
            )
            return handler

        return decorator

    def subscribe_handler(
        self, event_type: EventType, handler: EventHandler
    ) -> None:
        """Register a handler programmatically (non-decorator form)."""
        self._subscribers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a previously registered handler."""
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    # ── Publishing ────────────────────────────────────────────────────────────

    async def publish(self, event: SentinelEvent) -> None:
        """
        Enqueue an event for async dispatch.

        If the queue is full, the event is dropped and a warning is logged.
        This prevents back-pressure from starving the capture loop.

        Args:
            event: The SentinelEvent to publish.
        """
        try:
            self._queue.put_nowait(event)
            self._stats["published"] += 1
        except asyncio.QueueFull:
            self._stats["dropped"] += 1
            log.warning(
                "Event queue full — dropping event",
                event_type=event.type.name,
                dropped_total=self._stats["dropped"],
            )

    def publish_threadsafe(
        self, event: SentinelEvent, loop: asyncio.AbstractEventLoop
    ) -> None:
        """
        Thread-safe publish from a non-async context (e.g., Scapy callback).

        Args:
            event: The SentinelEvent to publish.
            loop:  The running event loop to schedule the coroutine on.
        """
        asyncio.run_coroutine_threadsafe(self.publish(event), loop)

    # ── Dispatch loop ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background event dispatch loop."""
        if self._running:
            return
        self._running = True
        self._dispatch_task = asyncio.create_task(
            self._dispatch_loop(), name="event-bus-dispatch"
        )
        log.info("EventBus started")

    async def stop(self) -> None:
        """Gracefully stop the dispatch loop after draining the queue."""
        self._running = False
        if self._dispatch_task and not self._dispatch_task.done():
            # Drain remaining events
            await self._drain()
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
        log.info(
            "EventBus stopped",
            published=self._stats["published"],
            dispatched=self._stats["dispatched"],
            dropped=self._stats["dropped"],
            handler_errors=self._stats["handler_errors"],
        )

    async def _dispatch_loop(self) -> None:
        """Internal loop: dequeue events and dispatch to subscribers."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                await self._dispatch(event)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as exc:  # noqa: BLE001
                log.error("Dispatch loop error", error=str(exc))

    async def _drain(self) -> None:
        """Drain all remaining events from the queue."""
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                await self._dispatch(event)
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def _dispatch(self, event: SentinelEvent) -> None:
        """Dispatch a single event to all registered handlers concurrently."""
        handlers = self._subscribers.get(event.type, [])
        if not handlers:
            return

        self._stats["dispatched"] += 1

        # Run all handlers concurrently; errors in one don't block others
        results = await asyncio.gather(
            *[handler(event) for handler in handlers],
            return_exceptions=True,
        )

        for result, handler in zip(results, handlers):
            if isinstance(result, Exception):
                self._stats["handler_errors"] += 1
                log.error(
                    "Event handler error",
                    event_type=event.type.name,
                    handler=handler.__name__,
                    error=str(result),
                )

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, int]:
        """Return a copy of event bus statistics."""
        return self._stats.copy()

    @property
    def queue_size(self) -> int:
        """Number of events currently pending in the queue."""
        return self._queue.qsize()
