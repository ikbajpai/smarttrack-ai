"""
test_alert_manager.py
---------------------
Unit tests for AlertManager, AlertHandler, and ConsoleAlertHandler.

All tests are self-contained; no video files, YOLO weights, or external
services are required.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.alerts.alert_manager import (
    AlertHandler,
    AlertManager,
    ConsoleAlertHandler,
)
from src.zones.zone_manager import EventType, IntrusionEvent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG: dict = {
    "enable_console": False,   # suppress auto-registration in most tests
    "cooldown_seconds": 5.0,
    "max_history": 10,
}


def _event(
    track_id: int = 1,
    zone_name: str = "Zone A",
    event_type: EventType = EventType.ENTER,
    *,
    seconds_ago: float = 0.0,
) -> IntrusionEvent:
    ts = datetime.utcnow() - timedelta(seconds=seconds_ago)
    return IntrusionEvent(
        timestamp=ts,
        track_id=track_id,
        zone_name=zone_name,
        event_type=event_type,
    )


class _CapturingHandler(AlertHandler):
    """Test double that records every event it receives."""

    def __init__(self) -> None:
        self.received: list[IntrusionEvent] = []

    def handle(self, event: IntrusionEvent) -> None:
        self.received.append(event)


class _RaisingHandler(AlertHandler):
    """Test double that always raises."""

    def handle(self, event: IntrusionEvent) -> None:
        raise RuntimeError("handler exploded")


# ---------------------------------------------------------------------------
# AlertHandler ABC
# ---------------------------------------------------------------------------

class TestAlertHandlerABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            AlertHandler()  # type: ignore[abstract]

    def test_subclass_without_handle_raises(self):
        class Incomplete(AlertHandler):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_works(self):
        handler = _CapturingHandler()
        ev = _event()
        handler.handle(ev)
        assert handler.received == [ev]


# ---------------------------------------------------------------------------
# ConsoleAlertHandler
# ---------------------------------------------------------------------------

class TestConsoleAlertHandler:
    def test_enter_logs_warning(self):
        handler = ConsoleAlertHandler()
        with patch("src.alerts.alert_manager.logger") as mock_log:
            handler.handle(_event(event_type=EventType.ENTER))
            mock_log.warning.assert_called_once()
            mock_log.info.assert_not_called()

    def test_exit_logs_info_when_verbose(self):
        handler = ConsoleAlertHandler(verbose=True)
        with patch("src.alerts.alert_manager.logger") as mock_log:
            handler.handle(_event(event_type=EventType.EXIT))
            mock_log.info.assert_called_once()
            mock_log.warning.assert_not_called()

    def test_exit_silent_when_not_verbose(self):
        handler = ConsoleAlertHandler(verbose=False)
        with patch("src.alerts.alert_manager.logger") as mock_log:
            handler.handle(_event(event_type=EventType.EXIT))
            mock_log.info.assert_not_called()
            mock_log.warning.assert_not_called()


# ---------------------------------------------------------------------------
# AlertManager.__init__
# ---------------------------------------------------------------------------

class TestAlertManagerInit:
    def test_no_handlers_when_console_disabled(self):
        am = AlertManager({"enable_console": False})
        assert am.handler_count == 0

    def test_console_handler_auto_registered_when_enabled(self):
        am = AlertManager({"enable_console": True})
        assert am.handler_count == 1
        assert isinstance(am._handlers[0], ConsoleAlertHandler)

    def test_explicit_handlers_override_auto_registration(self):
        h = _CapturingHandler()
        am = AlertManager({"enable_console": True}, handlers=[h])
        # Explicit list replaces auto-registration.
        assert am.handler_count == 1
        assert am._handlers[0] is h

    def test_empty_handlers_list_suppresses_auto_registration(self):
        am = AlertManager({"enable_console": True}, handlers=[])
        assert am.handler_count == 0

    def test_history_empty_at_start(self):
        am = AlertManager(_BASE_CONFIG)
        assert am.history_count == 0

    def test_cooldown_defaults(self):
        am = AlertManager({})
        assert am._cooldown.total_seconds() == 5.0

    def test_max_history_defaults(self):
        am = AlertManager({})
        assert am._max_history == 500

    def test_custom_cooldown_applied(self):
        am = AlertManager({"cooldown_seconds": 30.0})
        assert am._cooldown.total_seconds() == 30.0


# ---------------------------------------------------------------------------
# AlertManager.process
# ---------------------------------------------------------------------------

class TestAlertManagerProcess:
    def test_empty_list_returns_zero(self):
        am = AlertManager(_BASE_CONFIG)
        assert am.process([]) == 0

    def test_single_event_accepted(self):
        handler = _CapturingHandler()
        am = AlertManager(_BASE_CONFIG, handlers=[handler])
        count = am.process([_event()])
        assert count == 1
        assert len(handler.received) == 1

    def test_event_added_to_history(self):
        am = AlertManager(_BASE_CONFIG)
        am.process([_event()])
        assert am.history_count == 1

    def test_non_intrusion_event_skipped(self):
        am = AlertManager(_BASE_CONFIG)
        count = am.process(["not an event"])  # type: ignore[list-item]
        assert count == 0
        assert am.history_count == 0

    def test_returns_count_of_accepted_events(self):
        handler = _CapturingHandler()
        am = AlertManager(_BASE_CONFIG, handlers=[handler])
        events = [_event(track_id=1), _event(track_id=2), _event(track_id=3)]
        count = am.process(events)
        assert count == 3

    def test_dispatches_to_all_handlers(self):
        h1 = _CapturingHandler()
        h2 = _CapturingHandler()
        am = AlertManager(_BASE_CONFIG, handlers=[h1, h2])
        am.process([_event()])
        assert len(h1.received) == 1
        assert len(h2.received) == 1

    def test_handler_exception_does_not_stop_other_handlers(self):
        bad = _RaisingHandler()
        good = _CapturingHandler()
        am = AlertManager(_BASE_CONFIG, handlers=[bad, good])
        am.process([_event()])
        assert len(good.received) == 1

    def test_handler_exception_still_records_history(self):
        am = AlertManager(_BASE_CONFIG, handlers=[_RaisingHandler()])
        am.process([_event()])
        assert am.history_count == 1


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_duplicate_within_cooldown_suppressed(self):
        handler = _CapturingHandler()
        am = AlertManager({"cooldown_seconds": 10.0, "enable_console": False}, handlers=[handler])

        ev1 = _event(track_id=1, seconds_ago=0)
        ev2 = _event(track_id=1, seconds_ago=0)  # same key, within cooldown

        am.process([ev1])
        count = am.process([ev2])

        assert count == 0
        assert len(handler.received) == 1

    def test_event_after_cooldown_accepted(self):
        handler = _CapturingHandler()
        am = AlertManager({"cooldown_seconds": 5.0, "enable_console": False}, handlers=[handler])

        ev_old = _event(track_id=1, seconds_ago=10)  # 10s ago → outside 5s cooldown
        ev_new = _event(track_id=1, seconds_ago=0)

        am.process([ev_old])
        count = am.process([ev_new])

        assert count == 1
        assert len(handler.received) == 2

    def test_different_track_ids_not_deduplicated(self):
        handler = _CapturingHandler()
        am = AlertManager({"cooldown_seconds": 10.0, "enable_console": False}, handlers=[handler])

        am.process([_event(track_id=1)])
        count = am.process([_event(track_id=2)])

        assert count == 1
        assert len(handler.received) == 2

    def test_different_zones_not_deduplicated(self):
        handler = _CapturingHandler()
        am = AlertManager({"cooldown_seconds": 10.0, "enable_console": False}, handlers=[handler])

        am.process([_event(zone_name="Zone A")])
        count = am.process([_event(zone_name="Zone B")])

        assert count == 1

    def test_different_event_types_not_deduplicated(self):
        handler = _CapturingHandler()
        am = AlertManager({"cooldown_seconds": 10.0, "enable_console": False}, handlers=[handler])

        am.process([_event(event_type=EventType.ENTER)])
        count = am.process([_event(event_type=EventType.EXIT)])

        assert count == 1

    def test_zero_cooldown_never_deduplicates(self):
        handler = _CapturingHandler()
        am = AlertManager({"cooldown_seconds": 0.0, "enable_console": False}, handlers=[handler])

        am.process([_event(track_id=1)])
        count = am.process([_event(track_id=1)])

        assert count == 1
        assert len(handler.received) == 2


# ---------------------------------------------------------------------------
# AlertManager.get_recent
# ---------------------------------------------------------------------------

class TestGetRecent:
    def test_returns_empty_list_when_no_history(self):
        am = AlertManager(_BASE_CONFIG)
        assert am.get_recent() == []

    def test_most_recent_first(self):
        am = AlertManager({"enable_console": False, "cooldown_seconds": 0})
        ev1 = _event(track_id=1, seconds_ago=2)
        ev2 = _event(track_id=2, seconds_ago=1)
        ev3 = _event(track_id=3, seconds_ago=0)
        am.process([ev1, ev2, ev3])
        recent = am.get_recent()
        assert recent[0].track_id == 3
        assert recent[-1].track_id == 1

    def test_n_limits_results(self):
        am = AlertManager({"enable_console": False, "cooldown_seconds": 0})
        for i in range(5):
            am.process([_event(track_id=i)])
        assert len(am.get_recent(3)) == 3

    def test_n_none_returns_all(self):
        am = AlertManager({"enable_console": False, "cooldown_seconds": 0})
        for i in range(5):
            am.process([_event(track_id=i)])
        assert len(am.get_recent(None)) == 5

    def test_n_larger_than_history_returns_all(self):
        am = AlertManager({"enable_console": False, "cooldown_seconds": 0})
        am.process([_event()])
        assert len(am.get_recent(100)) == 1

    def test_n_zero_returns_empty(self):
        am = AlertManager({"enable_console": False, "cooldown_seconds": 0})
        am.process([_event()])
        assert am.get_recent(0) == []

    def test_history_bounded_by_max_history(self):
        am = AlertManager({"enable_console": False, "cooldown_seconds": 0, "max_history": 3})
        for i in range(6):
            am.process([_event(track_id=i)])
        assert am.history_count == 3


# ---------------------------------------------------------------------------
# AlertManager.clear_history
# ---------------------------------------------------------------------------

class TestClearHistory:
    def test_clears_history(self):
        am = AlertManager(_BASE_CONFIG)
        am.process([_event()])
        am.clear_history()
        assert am.history_count == 0

    def test_clears_deduplication_state(self):
        handler = _CapturingHandler()
        am = AlertManager({"enable_console": False, "cooldown_seconds": 60}, handlers=[handler])

        am.process([_event(track_id=1)])
        am.clear_history()
        count = am.process([_event(track_id=1)])  # cooldown reset → accepted

        assert count == 1
        assert len(handler.received) == 2

    def test_handlers_preserved_after_clear(self):
        h = _CapturingHandler()
        am = AlertManager(_BASE_CONFIG, handlers=[h])
        am.process([_event()])
        am.clear_history()
        am.process([_event()])
        assert len(h.received) == 2


# ---------------------------------------------------------------------------
# AlertManager.register_handler
# ---------------------------------------------------------------------------

class TestRegisterHandler:
    def test_register_valid_handler(self):
        am = AlertManager(_BASE_CONFIG)
        h = _CapturingHandler()
        am.register_handler(h)
        assert am.handler_count == 1

    def test_registered_handler_receives_events(self):
        am = AlertManager(_BASE_CONFIG)
        h = _CapturingHandler()
        am.register_handler(h)
        am.process([_event()])
        assert len(h.received) == 1

    def test_register_non_handler_raises_type_error(self):
        am = AlertManager(_BASE_CONFIG)
        with pytest.raises(TypeError):
            am.register_handler("not a handler")  # type: ignore[arg-type]
