"""Trace session management with LRU caching."""

import logging
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TraceSession:
    """Represents a loaded trace session."""

    id: str
    path: Path
    loaded_at: datetime = field(default_factory=datetime.now)
    name: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    _processor: Optional[object] = field(default=None, repr=False)

    @property
    def display_name(self) -> str:
        """Get display name for the trace."""
        return self.name or self.path.stem

    def close(self) -> None:
        """Close the trace processor if open."""
        if self._processor is not None:
            try:
                self._processor.close()
            except Exception as e:
                logger.warning(f"Error closing trace processor: {e}")
            self._processor = None


class TraceSessionManager:
    """Manages multiple trace sessions with LRU eviction."""

    DEFAULT_MAX_TRACES = 10

    def __init__(self, max_traces: int = DEFAULT_MAX_TRACES):
        self._max_traces = max_traces
        self._sessions: OrderedDict[str, TraceSession] = OrderedDict()
        self._active_id: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def active_session(self) -> Optional[TraceSession]:
        """Get the currently active trace session."""
        with self._lock:
            if self._active_id and self._active_id in self._sessions:
                return self._sessions[self._active_id]
            return None

    @property
    def active_id(self) -> Optional[str]:
        """Get the ID of the currently active trace."""
        return self._active_id

    def load_trace(
        self,
        path: Path,
        name: Optional[str] = None,
        set_active: bool = True,
    ) -> TraceSession:
        """Load a trace file and create a session."""
        path = Path(path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        with self._lock:
            for session in self._sessions.values():
                if session.path == path:
                    if set_active:
                        self._active_id = session.id
                        self._sessions.move_to_end(session.id)
                    return session

            while len(self._sessions) >= self._max_traces:
                oldest_id, oldest_session = self._sessions.popitem(last=False)
                logger.info(f"Evicting trace session: {oldest_id}")
                oldest_session.close()

            session_id = str(uuid.uuid4())[:8]
            session = TraceSession(
                id=session_id,
                path=path,
                name=name,
            )

            self._sessions[session_id] = session

            if set_active:
                self._active_id = session_id

            logger.info(f"Loaded trace session {session_id}: {path}")
            return session

    def get_session(self, session_id: str) -> Optional[TraceSession]:
        """Get a trace session by ID."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                self._sessions.move_to_end(session_id)
            return session

    def set_active(self, session_id: str) -> bool:
        """Set the active trace session."""
        with self._lock:
            if session_id in self._sessions:
                self._active_id = session_id
                self._sessions.move_to_end(session_id)
                return True
            return False

    def list_sessions(self) -> list[TraceSession]:
        """List all loaded trace sessions."""
        with self._lock:
            return list(self._sessions.values())

    def close_session(self, session_id: str) -> bool:
        """Close and remove a trace session."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                session.close()
                if self._active_id == session_id:
                    self._active_id = (
                        next(iter(self._sessions.keys())) if self._sessions else None
                    )
                return True
            return False

    def close_all(self) -> None:
        """Close all trace sessions."""
        with self._lock:
            for session in self._sessions.values():
                session.close()
            self._sessions.clear()
            self._active_id = None

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions
