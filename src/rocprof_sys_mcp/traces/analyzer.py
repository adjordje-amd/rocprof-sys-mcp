"""Perfetto trace analysis with SQL queries."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="trace_analyzer")


@dataclass
class QueryResult:
    """Result of a trace query."""

    success: bool
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    error: Optional[str] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert result to pandas DataFrame."""
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_dict(self) -> dict:
        """Convert result to dictionary format."""
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
        }


@dataclass
class TraceSummary:
    """High-level summary of a trace."""

    duration_ns: int
    slice_count: int
    track_count: int
    process_count: int
    thread_count: int
    gpu_slice_count: int
    tables: list[str]


@dataclass
class GpuKernelStats:
    """GPU kernel performance statistics."""

    kernel_name: str
    call_count: int
    total_duration_ns: int
    avg_duration_ns: float
    min_duration_ns: int
    max_duration_ns: int
    percentage_of_total: float


class TraceAnalyzer:
    """Analyzes Perfetto traces using SQL queries."""

    def __init__(self, trace_path: Path):
        self.trace_path = Path(trace_path)
        self._processor: Optional[Any] = None
        self._lock = asyncio.Lock()

    def _get_processor(self) -> Any:
        """Get or create the TraceProcessor (synchronous)."""
        if self._processor is None:
            from perfetto.trace_processor import TraceProcessor

            self._processor = TraceProcessor(trace=str(self.trace_path))
        return self._processor

    async def _run_sync(self, func, *args, **kwargs):
        """Run a synchronous function in the thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, lambda: func(*args, **kwargs))

    def _query_sync(self, sql: str) -> QueryResult:
        """Execute SQL query synchronously."""
        try:
            tp = self._get_processor()
            result = tp.query(sql)
            df = result.as_pandas_dataframe()
            return QueryResult(
                success=True,
                columns=list(df.columns),
                rows=df.values.tolist(),
                row_count=len(df),
            )
        except Exception as e:
            logger.exception(f"Query failed: {sql}")
            return QueryResult(
                success=False,
                columns=[],
                rows=[],
                row_count=0,
                error=str(e),
            )

    async def query(self, sql: str) -> QueryResult:
        """Execute SQL query on the trace."""
        async with self._lock:
            return await self._run_sync(self._query_sync, sql)

    def _get_summary_sync(self) -> TraceSummary:
        """Get trace summary synchronously."""
        tp = self._get_processor()

        duration_result = tp.query(
            "SELECT COALESCE(MAX(ts + dur) - MIN(ts), 0) as duration FROM slice"
        )
        duration_df = duration_result.as_pandas_dataframe()
        duration_ns = int(duration_df["duration"].iloc[0]) if len(duration_df) > 0 else 0

        slice_result = tp.query("SELECT COUNT(*) as cnt FROM slice")
        slice_df = slice_result.as_pandas_dataframe()
        slice_count = int(slice_df["cnt"].iloc[0]) if len(slice_df) > 0 else 0

        track_result = tp.query("SELECT COUNT(*) as cnt FROM track")
        track_df = track_result.as_pandas_dataframe()
        track_count = int(track_df["cnt"].iloc[0]) if len(track_df) > 0 else 0

        process_result = tp.query("SELECT COUNT(DISTINCT upid) as cnt FROM process")
        process_df = process_result.as_pandas_dataframe()
        process_count = int(process_df["cnt"].iloc[0]) if len(process_df) > 0 else 0

        thread_result = tp.query("SELECT COUNT(DISTINCT utid) as cnt FROM thread")
        thread_df = thread_result.as_pandas_dataframe()
        thread_count = int(thread_df["cnt"].iloc[0]) if len(thread_df) > 0 else 0

        gpu_slice_count = 0
        try:
            gpu_result = tp.query(
                """
                SELECT COUNT(*) as cnt FROM slice s
                JOIN track t ON s.track_id = t.id
                WHERE t.name LIKE '%GPU%' OR t.name LIKE '%HIP%' OR t.name LIKE '%HSA%'
                """
            )
            gpu_df = gpu_result.as_pandas_dataframe()
            gpu_slice_count = int(gpu_df["cnt"].iloc[0]) if len(gpu_df) > 0 else 0
        except Exception:
            pass

        tables_result = tp.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables_df = tables_result.as_pandas_dataframe()
        tables = tables_df["name"].tolist() if len(tables_df) > 0 else []

        return TraceSummary(
            duration_ns=duration_ns,
            slice_count=slice_count,
            track_count=track_count,
            process_count=process_count,
            thread_count=thread_count,
            gpu_slice_count=gpu_slice_count,
            tables=tables,
        )

    async def get_summary(self) -> TraceSummary:
        """Get high-level trace summary."""
        async with self._lock:
            return await self._run_sync(self._get_summary_sync)

    def _get_gpu_kernel_stats_sync(self, limit: int = 20) -> list[GpuKernelStats]:
        """Get GPU kernel statistics synchronously."""
        tp = self._get_processor()

        total_result = tp.query(
            """
            SELECT COALESCE(SUM(dur), 0) as total FROM slice s
            JOIN track t ON s.track_id = t.id
            WHERE t.name LIKE '%GPU%' OR t.name LIKE '%HIP%' OR t.name LIKE '%HSA%'
            """
        )
        total_df = total_result.as_pandas_dataframe()
        total_gpu_time = int(total_df["total"].iloc[0]) if len(total_df) > 0 else 0

        if total_gpu_time == 0:
            return []

        kernel_result = tp.query(
            f"""
            SELECT
                s.name as kernel_name,
                COUNT(*) as call_count,
                SUM(s.dur) as total_duration_ns,
                AVG(s.dur) as avg_duration_ns,
                MIN(s.dur) as min_duration_ns,
                MAX(s.dur) as max_duration_ns
            FROM slice s
            JOIN track t ON s.track_id = t.id
            WHERE t.name LIKE '%GPU%' OR t.name LIKE '%HIP%' OR t.name LIKE '%HSA%'
            GROUP BY s.name
            ORDER BY total_duration_ns DESC
            LIMIT {limit}
            """
        )
        kernel_df = kernel_result.as_pandas_dataframe()

        stats = []
        for _, row in kernel_df.iterrows():
            total_dur = int(row["total_duration_ns"])
            stats.append(
                GpuKernelStats(
                    kernel_name=str(row["kernel_name"]),
                    call_count=int(row["call_count"]),
                    total_duration_ns=total_dur,
                    avg_duration_ns=float(row["avg_duration_ns"]),
                    min_duration_ns=int(row["min_duration_ns"]),
                    max_duration_ns=int(row["max_duration_ns"]),
                    percentage_of_total=(total_dur / total_gpu_time * 100)
                    if total_gpu_time > 0
                    else 0.0,
                )
            )
        return stats

    async def get_gpu_kernel_stats(self, limit: int = 20) -> list[GpuKernelStats]:
        """Get GPU kernel performance statistics."""
        async with self._lock:
            return await self._run_sync(self._get_gpu_kernel_stats_sync, limit)

    def _get_schema_sync(self) -> dict[str, list[dict]]:
        """Get trace schema synchronously."""
        tp = self._get_processor()

        tables_result = tp.query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables_df = tables_result.as_pandas_dataframe()

        schema = {}
        for table_name in tables_df["name"].tolist():
            try:
                pragma_result = tp.query(f"PRAGMA table_info({table_name})")
                pragma_df = pragma_result.as_pandas_dataframe()
                columns = []
                for _, row in pragma_df.iterrows():
                    columns.append(
                        {
                            "name": str(row["name"]),
                            "type": str(row["type"]),
                        }
                    )
                schema[table_name] = columns
            except Exception as e:
                logger.warning(f"Failed to get schema for table {table_name}: {e}")
                schema[table_name] = []

        return schema

    async def get_schema(self) -> dict[str, list[dict]]:
        """Get trace database schema."""
        async with self._lock:
            return await self._run_sync(self._get_schema_sync)

    def _get_metadata_sync(self) -> dict:
        """Get trace metadata synchronously."""
        tp = self._get_processor()

        metadata = {
            "trace_path": str(self.trace_path),
            "trace_size_bytes": self.trace_path.stat().st_size,
        }

        try:
            meta_result = tp.query("SELECT name, str_value FROM metadata")
            meta_df = meta_result.as_pandas_dataframe()
            for _, row in meta_df.iterrows():
                metadata[str(row["name"])] = str(row["str_value"])
        except Exception:
            pass

        return metadata

    async def get_metadata(self) -> dict:
        """Get trace metadata."""
        async with self._lock:
            return await self._run_sync(self._get_metadata_sync)

    def close(self) -> None:
        """Close the trace processor."""
        if self._processor is not None:
            try:
                self._processor.close()
            except Exception as e:
                logger.warning(f"Error closing trace processor: {e}")
            self._processor = None

    def __del__(self):
        self.close()
