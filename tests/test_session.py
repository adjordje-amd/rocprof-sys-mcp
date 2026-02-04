"""Tests for trace session management."""

import tempfile
from pathlib import Path

import pytest

from rocprof_sys_mcp.traces.session import TraceSession, TraceSessionManager


class TestTraceSession:
    def test_display_name_with_name(self):
        session = TraceSession(id="test", path=Path("/tmp/trace.proto"), name="my_trace")
        assert session.display_name == "my_trace"

    def test_display_name_without_name(self):
        session = TraceSession(id="test", path=Path("/tmp/trace.proto"))
        assert session.display_name == "trace"


class TestTraceSessionManager:
    def test_load_trace_creates_session(self, tmp_path):
        trace_file = tmp_path / "test.proto"
        trace_file.touch()

        manager = TraceSessionManager()
        session = manager.load_trace(trace_file)

        assert session is not None
        assert session.path == trace_file.resolve()
        assert manager.active_session == session

    def test_load_trace_sets_active(self, tmp_path):
        trace1 = tmp_path / "trace1.proto"
        trace2 = tmp_path / "trace2.proto"
        trace1.touch()
        trace2.touch()

        manager = TraceSessionManager()
        session1 = manager.load_trace(trace1)
        session2 = manager.load_trace(trace2)

        assert manager.active_session == session2
        assert manager.active_id == session2.id

    def test_load_trace_file_not_found(self, tmp_path):
        manager = TraceSessionManager()
        with pytest.raises(FileNotFoundError):
            manager.load_trace(tmp_path / "nonexistent.proto")

    def test_lru_eviction(self, tmp_path):
        manager = TraceSessionManager(max_traces=2)

        traces = []
        for i in range(3):
            trace = tmp_path / f"trace{i}.proto"
            trace.touch()
            traces.append(trace)

        session0 = manager.load_trace(traces[0])
        session1 = manager.load_trace(traces[1])
        session2 = manager.load_trace(traces[2])

        assert len(manager) == 2
        assert session0.id not in manager
        assert session1.id in manager
        assert session2.id in manager

    def test_get_session(self, tmp_path):
        trace = tmp_path / "trace.proto"
        trace.touch()

        manager = TraceSessionManager()
        session = manager.load_trace(trace)

        retrieved = manager.get_session(session.id)
        assert retrieved == session

        assert manager.get_session("nonexistent") is None

    def test_close_session(self, tmp_path):
        trace = tmp_path / "trace.proto"
        trace.touch()

        manager = TraceSessionManager()
        session = manager.load_trace(trace)
        session_id = session.id

        assert manager.close_session(session_id) is True
        assert session_id not in manager
        assert manager.active_session is None

    def test_list_sessions(self, tmp_path):
        manager = TraceSessionManager()

        assert manager.list_sessions() == []

        trace = tmp_path / "trace.proto"
        trace.touch()
        session = manager.load_trace(trace)

        sessions = manager.list_sessions()
        assert len(sessions) == 1
        assert sessions[0] == session
