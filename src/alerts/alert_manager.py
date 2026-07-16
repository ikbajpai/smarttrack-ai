"""
alert_manager.py
----------------
Alert dispatch and in-memory history for SmartTrack AI.

Provides:
    AlertHandler        — Abstract base class for pluggable notification backends.
    ConsoleAlertHandler — Built-in handler that logs events via loguru.
    AlertManager        — Receives IntrusionEvent objects, deduplicates them,
                          dispatches to registered handlers, and maintains an
                          in-memory event history.

This module depends only on IntrusionEvent from zone_manager.py.  It never
imports Detector, Tracker, ZoneManager internals, OpenCV, or Streamlit.  The
pipeline layer wires modules together.

Adding a new notification backend (CSV, Telegram, Webhook, …) requires only:
    1. Subclass AlertHandler and implement handle().
    2. Pass an instance to AlertManager(config, handlers=[MyHandler()]).
The AlertManager public interface remains unchanged.

Usage::

    import yaml
    from src.alerts.alert_manager import AlertManager

    with open("config/config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    am = AlertManager(cfg["alerts"])

    # Per-frame call after ZoneManager:
    processed = am.process(intrusion_events)

    # Retrieve recent alerts for a dashboard:
    recent = am.get_recent(20)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from src.zones.zone_manager import EventType, IntrusionEvent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MAX_HISTORY: int = 500
_DEFAULT_COOLDOWN_SECONDS: float = 5.0


# ---------------------------------------------------------------------------
# Handler abstraction
# ---------------------------------------------------------------------------

class AlertHandler(ABC):
    """Abstract base class for pluggable alert notification backends.

    Subclass this to add new delivery channels (CSV, Telegram, email, webhook,
    etc.) without modifying AlertManager.  Each handler receives one
    :class:`~src.zones.zone_manager.IntrusionEvent` at a time; batching or
    buffering is the handler's own responsibility.

    Example::

        class MyWebhookHandler(AlertHandler):
            def handle(self, event: IntrusionEvent) -> None:
                requests.post(WEBHOOK_URL, json={"track_id": event.track_id})
    """

    @abstractmethod
    def handle(self, event: IntrusionEvent) -> None:
        """Dispatch a single intrusion event to this backend.

        Implementations must not raise exceptions — failures should be caught
        internally and logged so that other handlers continue to run.

        Args:
            event: The intrusion event to dispatch.
        """


# ---------------------------------------------------------------------------
# Built-in handler: console / loguru
# ---------------------------------------------------------------------------

class ConsoleAlertHandler(AlertHandler):
    """Logs intrusion events to the console via loguru.

    ENTER events are logged at WARNING level to draw operator attention.
    EXIT events are logged at INFO level as routine bookkeeping.

    Args:
        verbose: When ``True`` (default), both ENTER and EXIT events are
            logged.  When ``False``, only ENTER events are logged.
    """

    def __init__(self, *, verbose: bool = True) -> None:
        self._verbose = verbose

    def handle(self, event: IntrusionEvent) -> None:
        """Log *event* to the console.

        Args:
            event: The intrusion event to log.
        """
        ts = event.timestamp.strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm

        if event.event_type == EventType.ENTER:
            logger.warning(
                "[ALERT] ENTER | track_id={} zone='{}' ts={}",
                event.track_id,
                event.zone_name,
                ts,
            )
        elif self._verbose:
            logger.info(
                "[ALERT] EXIT  | track_id={} zone='{}' ts={}",
                event.track_id,
                event.zone_name,
                ts,
            )


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class AlertManager:
    """Receives intrusion events, deduplicates them, and dispatches alerts.

    Handlers are registered at construction time via dependency injection.
    The public interface is stable regardless of which handlers are active,
    making it straightforward to add new notification channels.

    Deduplication is keyed on ``(track_id, zone_name, event_type)``.  An
    event is suppressed if an identical event was already processed within
    the last ``cooldown_seconds``.  This acts as a safety net on top of the
    geometric deduplication already performed by ZoneManager.

    Args:
        config: The ``alerts`` section of ``config/config.yaml``.  Recognised
            keys::

                enable_console:      true    # auto-register ConsoleAlertHandler
                cooldown_seconds:    5.0     # deduplication window
                max_history:         500     # in-memory history cap
        handlers: Optional list of :class:`AlertHandler` instances to register
            in addition to (or instead of) any auto-registered handlers.  Pass
            an empty list ``[]`` to suppress auto-registration.

    Example::

        import yaml
        from src.alerts.alert_manager import AlertManager, ConsoleAlertHandler

        with open("config/config.yaml") as fh:
            cfg = yaml.safe_load(fh)

        am = AlertManager(cfg["alerts"])
        am.process(zone_manager_events)
        recent = am.get_recent(10)
    """

    def __init__(
        self,
        config: dict[str, Any],
        handlers: list[AlertHandler] | None = None,
    ) -> None:
        self._cooldown = timedelta(
            seconds=float(config.get("cooldown_seconds", _DEFAULT_COOLDOWN_SECONDS))
        )
        self._max_history: int = int(config.get("max_history", _DEFAULT_MAX_HISTORY))

        self._history: deque[IntrusionEvent] = deque(maxlen=self._max_history)

        # (track_id, zone_name, event_type) → last dispatch time
        self._last_alert_time: dict[tuple[int, str, EventType], datetime] = {}

        # Register handlers.
        if handlers is not None:
            self._handlers: list[AlertHandler] = list(handlers)
        else:
            self._handlers = []
            if config.get("enable_console", True):
                self._handlers.append(ConsoleAlertHandler())

        logger.info(
            "AlertManager initialised | handlers={} cooldown={:.1f}s max_history={}",
            [type(h).__name__ for h in self._handlers],
            self._cooldown.total_seconds(),
            self._max_history,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, events: list[IntrusionEvent]) -> int:
        """Process a list of intrusion events from ZoneManager.

        Each event is checked against the deduplication window.  Events that
        pass are added to the history and dispatched to all registered handlers.

        Args:
            events: Output of
                :meth:`~src.zones.zone_manager.ZoneManager.check_intrusions`
                for the current frame.

        Returns:
            Number of events that were accepted (not suppressed by cooldown).
        """
        if not events:
            return 0

        accepted = 0
        for event in events:
            if not isinstance(event, IntrusionEvent):
                logger.warning(
                    "process(): received non-IntrusionEvent object ({}) — skipped.",
                    type(event).__name__,
                )
                continue

            if self._is_duplicate(event):
                logger.debug(
                    "Suppressed duplicate | track_id={} zone='{}' type={}",
                    event.track_id,
                    event.zone_name,
                    event.event_type,
                )
                continue

            self._record(event)
            self._dispatch(event)
            accepted += 1

        return accepted

    def get_recent(self, n: int | None = None) -> list[IntrusionEvent]:
        """Return recent events from the in-memory history, newest first.

        Args:
            n: Maximum number of events to return.  ``None`` returns the full
               history up to ``max_history``.

        Returns:
            List of :class:`~src.zones.zone_manager.IntrusionEvent` instances,
            ordered from most recent to oldest.
        """
        history = list(self._history)
        history.reverse()
        if n is None:
            return history
        return history[:max(0, int(n))]

    def clear_history(self) -> None:
        """Clear the in-memory alert history and deduplication state.

        Does not affect registered handlers or configuration.
        """
        self._history.clear()
        self._last_alert_time.clear()
        logger.info("AlertManager history cleared.")

    def register_handler(self, handler: AlertHandler) -> None:
        """Register an additional alert handler at runtime.

        Useful when a handler needs to be added after construction (e.g., once
        a Streamlit UI obtains a file path from the user).

        Args:
            handler: A concrete :class:`AlertHandler` instance to add.
        """
        if not isinstance(handler, AlertHandler):
            raise TypeError(
                f"handler must be an AlertHandler subclass, got {type(handler).__name__}"
            )
        self._handlers.append(handler)
        logger.info("Registered handler: {}", type(handler).__name__)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def history_count(self) -> int:
        """Number of events currently stored in the in-memory history."""
        return len(self._history)

    @property
    def handler_count(self) -> int:
        """Number of registered alert handlers."""
        return len(self._handlers)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_duplicate(self, event: IntrusionEvent) -> bool:
        """Return True if an identical event was dispatched within the cooldown window.

        Args:
            event: Candidate event to check.

        Returns:
            ``True`` when the event should be suppressed.
        """
        key = (event.track_id, event.zone_name, event.event_type)
        last = self._last_alert_time.get(key)
        if last is None:
            return False
        return (event.timestamp - last) < self._cooldown

    def _record(self, event: IntrusionEvent) -> None:
        """Add *event* to the history and update the deduplication timestamp.

        Args:
            event: Event that has been accepted for dispatch.
        """
        self._history.append(event)
        key = (event.track_id, event.zone_name, event.event_type)
        self._last_alert_time[key] = event.timestamp

    def _dispatch(self, event: IntrusionEvent) -> None:
        """Send *event* to every registered handler.

        Handler exceptions are caught individually so that one failing backend
        does not prevent others from running.

        Args:
            event: Event to dispatch.
        """
        for handler in self._handlers:
            try:
                handler.handle(event)
            except Exception as exc:
                logger.error(
                    "Handler {} raised an exception: {}",
                    type(handler).__name__,
                    exc,
                )
