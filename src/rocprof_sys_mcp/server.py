"""MCP server for rocprofiler-systems profiling integration."""

import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from rocprof_sys_mcp.profiling.runner import ProfilingConfig, ProfilingRunner
from rocprof_sys_mcp.reports.generator import ReportFormat, ReportGenerator
from rocprof_sys_mcp.traces.analyzer import TraceAnalyzer
from rocprof_sys_mcp.traces.session import TraceSession, TraceSessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "rocprof-sys-mcp",
    instructions="MCP server for rocprofiler-systems profiling and trace analysis",
)

session_manager = TraceSessionManager()
profiling_runner = ProfilingRunner()
report_generator = ReportGenerator()

_analyzers: dict[str, TraceAnalyzer] = {}


def _get_analyzer(session: TraceSession) -> TraceAnalyzer:
    """Get or create an analyzer for a trace session."""
    if session.id not in _analyzers:
        _analyzers[session.id] = TraceAnalyzer(session.path)
    return _analyzers[session.id]


def _get_active_session() -> TraceSession:
    """Get the active trace session or raise an error."""
    session = session_manager.active_session
    if session is None:
        raise ValueError("No active trace. Use load_trace first.")
    return session


@mcp.tool()
async def profile_application(
    command: Annotated[str, Field(description="The command/executable to profile")],
    args: Annotated[
        Optional[list[str]], Field(description="Arguments to pass to the command")
    ] = None,
    working_dir: Annotated[
        Optional[str], Field(description="Working directory for the command")
    ] = None,
    output_dir: Annotated[
        Optional[str], Field(description="Directory to store profiling output")
    ] = None,
    profile: Annotated[
        bool, Field(description="Enable call-stack based profiling output")
    ] = True,
    trace: Annotated[
        bool, Field(description="Enable detailed Perfetto trace output")
    ] = True,
    flat_profile: Annotated[
        bool, Field(description="Generate flat profile instead of call-stack profile")
    ] = False,
    # Sampling options
    sample: Annotated[
        Optional[str], Field(description="Enable sampling: 'cputime' or 'realtime'")
    ] = None,
    sampling_freq: Annotated[
        Optional[float], Field(description="Sampling frequency (interrupts per second)")
    ] = None,
    sampling_wait: Annotated[
        Optional[float], Field(description="Delay before first sample (seconds)")
    ] = None,
    sampling_duration: Annotated[
        Optional[float], Field(description="Duration of sampling (seconds)")
    ] = None,
    # Host/Device monitoring
    host: Annotated[
        bool, Field(description="Enable host metrics (CPU freq, memory usage, etc.)")
    ] = False,
    device: Annotated[
        bool, Field(description="Enable device metrics (GPU temp, memory, power, etc.)")
    ] = False,
    # Trace timing
    trace_wait: Annotated[
        Optional[float], Field(description="Delay before collecting trace (seconds)")
    ] = None,
    trace_duration: Annotated[
        Optional[float], Field(description="Duration of trace collection (seconds)")
    ] = None,
    trace_buffer_size: Annotated[
        Optional[int], Field(description="Trace buffer size in KB")
    ] = None,
    trace_fill_policy: Annotated[
        Optional[str], Field(description="Buffer policy when full: 'discard' or 'ring_buffer'")
    ] = None,
    # Backend options
    use_rocm: Annotated[
        Optional[bool], Field(description="Enable ROCm API and kernel tracing")
    ] = None,
    use_sampling: Annotated[
        Optional[bool], Field(description="Enable statistical sampling of call-stack")
    ] = None,
    use_amd_smi: Annotated[
        Optional[bool], Field(description="Enable AMD SMI metrics (GPU power, temp, memory)")
    ] = None,
    use_kokkosp: Annotated[
        Optional[bool], Field(description="Enable Kokkos Tools support")
    ] = None,
    use_mpip: Annotated[
        Optional[bool], Field(description="Enable MPI profiling support")
    ] = None,
    use_rcclp: Annotated[
        Optional[bool], Field(description="Enable RCCL profiling support")
    ] = None,
    # Category filtering
    enable_categories: Annotated[
        Optional[list[str]], Field(
            description="Categories to enable (e.g., rocm, rocm_hip_api, rocm_kernel_dispatch, sampling)"
        )
    ] = None,
    disable_categories: Annotated[
        Optional[list[str]], Field(description="Categories to disable")
    ] = None,
    # GPU selection
    gpus: Annotated[
        Optional[list[str]], Field(description="GPU IDs for SMI queries (e.g., ['0', '1'] or ['0-3'])")
    ] = None,
    # Advanced options
    perfetto_annotations: Annotated[
        Optional[bool], Field(description="Include debug annotations in trace (may increase size)")
    ] = None,
    config_file: Annotated[
        Optional[str], Field(description="Path to rocprof-sys configuration file")
    ] = None,
    source_env: Annotated[
        Optional[str], Field(description="Path to environment file to source before profiling (e.g., setup.sh for development builds)")
    ] = None,
    env_vars: Annotated[
        Optional[dict[str, str]], Field(description="Additional environment variables to set for the profiling run")
    ] = None,
    timeout_seconds: Annotated[
        Optional[float], Field(description="Timeout in seconds for profiling")
    ] = None,
    load_trace_after: Annotated[
        bool, Field(description="Automatically load the trace after profiling")
    ] = True,
) -> dict:
    """Run rocprof-sys-run to profile an application.

    Executes the specified command with rocprofiler-systems instrumentation
    to capture performance data and generate Perfetto traces.

    Common usage patterns:
    - Basic GPU profiling: profile=True, trace=True, use_rocm=True
    - CPU sampling: sample='cputime', sampling_freq=100
    - GPU metrics: device=True, use_amd_smi=True
    - Focused tracing: enable_categories=['rocm_kernel_dispatch', 'rocm_hip_api']
    - Development build: source_env='/path/to/setup.sh'
    - Custom environment: env_vars={'LD_LIBRARY_PATH': '/custom/path', 'DEBUG': '1'}
    """
    config = ProfilingConfig(
        command=command,
        args=args or [],
        working_dir=working_dir,
        output_dir=output_dir,
        profile=profile,
        trace=trace,
        flat_profile=flat_profile,
        sample=sample,
        sampling_freq=sampling_freq,
        sampling_wait=sampling_wait,
        sampling_duration=sampling_duration,
        host=host,
        device=device,
        trace_wait=trace_wait,
        trace_duration=trace_duration,
        trace_buffer_size=trace_buffer_size,
        trace_fill_policy=trace_fill_policy,
        use_rocm=use_rocm,
        use_sampling=use_sampling,
        use_amd_smi=use_amd_smi,
        use_kokkosp=use_kokkosp,
        use_mpip=use_mpip,
        use_rcclp=use_rcclp,
        enable_categories=enable_categories,
        disable_categories=disable_categories,
        gpus=gpus,
        perfetto_annotations=perfetto_annotations,
        config_file=config_file,
        source_env=source_env,
        env_vars=env_vars or {},
        timeout_seconds=timeout_seconds,
    )

    result = await profiling_runner.run(config)

    response = {
        "success": result.success,
        "return_code": result.return_code,
        "duration_seconds": result.duration_seconds,
    }

    if result.error_message:
        response["error"] = result.error_message

    if result.output_dir:
        response["output_dir"] = str(result.output_dir)

    if result.trace_path:
        response["trace_path"] = str(result.trace_path)

        if load_trace_after:
            session = session_manager.load_trace(
                result.trace_path,
                name=f"profile_{Path(command).stem}",
            )
            response["trace_id"] = session.id
            response["trace_loaded"] = True

    if result.stdout:
        response["stdout"] = result.stdout[:2000]
    if result.stderr:
        response["stderr"] = result.stderr[:2000]

    return response


@mcp.tool()
async def load_trace(
    path: Annotated[str, Field(description="Path to the Perfetto trace file")],
    name: Annotated[
        Optional[str], Field(description="Optional friendly name for the trace")
    ] = None,
    set_active: Annotated[
        bool, Field(description="Set this trace as the active trace")
    ] = True,
) -> dict:
    """Load an existing Perfetto trace file for analysis.

    The trace will be added to the session and can be queried using SQL.
    """
    trace_path = Path(path).expanduser().resolve()

    session = session_manager.load_trace(
        trace_path,
        name=name,
        set_active=set_active,
    )

    analyzer = _get_analyzer(session)
    metadata = await analyzer.get_metadata()

    return {
        "trace_id": session.id,
        "path": str(session.path),
        "name": session.display_name,
        "is_active": session_manager.active_id == session.id,
        "metadata": metadata,
    }


@mcp.tool()
async def query_trace(
    sql: Annotated[str, Field(description="SQL query to execute on the trace")],
    trace_id: Annotated[
        Optional[str], Field(description="Trace ID to query (uses active trace if not specified)")
    ] = None,
    limit: Annotated[
        int, Field(description="Maximum number of rows to return")
    ] = 100,
) -> dict:
    """Execute a SQL query on a loaded trace.

    Uses Perfetto's TraceProcessor SQL interface. Common tables include:
    - slice: Time spans with name, duration, track
    - track: Trace tracks (threads, GPU queues, etc.)
    - thread: Thread information
    - process: Process information
    - counter: Counter values over time

    Example queries:
    - SELECT name, dur FROM slice ORDER BY dur DESC LIMIT 10
    - SELECT t.name, COUNT(*) FROM slice s JOIN track t ON s.track_id = t.id GROUP BY t.name
    """
    if trace_id:
        session = session_manager.get_session(trace_id)
        if session is None:
            raise ValueError(f"Trace not found: {trace_id}")
    else:
        session = _get_active_session()

    analyzer = _get_analyzer(session)

    if not sql.strip().upper().startswith("SELECT"):
        if "LIMIT" not in sql.upper():
            sql = f"{sql.rstrip(';')} LIMIT {limit}"

    result = await analyzer.query(sql)

    if not result.success:
        return {
            "success": False,
            "error": result.error,
        }

    rows = result.rows[:limit] if len(result.rows) > limit else result.rows

    return {
        "success": True,
        "columns": result.columns,
        "rows": rows,
        "row_count": result.row_count,
        "truncated": result.row_count > limit,
    }


@mcp.tool()
async def get_trace_summary(
    trace_id: Annotated[
        Optional[str], Field(description="Trace ID (uses active trace if not specified)")
    ] = None,
) -> dict:
    """Get high-level statistics and summary for a trace.

    Returns duration, slice counts, process/thread counts, and available tables.
    """
    if trace_id:
        session = session_manager.get_session(trace_id)
        if session is None:
            raise ValueError(f"Trace not found: {trace_id}")
    else:
        session = _get_active_session()

    analyzer = _get_analyzer(session)
    summary = await analyzer.get_summary()

    return {
        "trace_id": session.id,
        "trace_name": session.display_name,
        "duration_ns": summary.duration_ns,
        "duration_formatted": _format_duration(summary.duration_ns),
        "slice_count": summary.slice_count,
        "gpu_slice_count": summary.gpu_slice_count,
        "process_count": summary.process_count,
        "thread_count": summary.thread_count,
        "track_count": summary.track_count,
        "tables": summary.tables,
    }


@mcp.tool()
async def get_gpu_kernel_stats(
    trace_id: Annotated[
        Optional[str], Field(description="Trace ID (uses active trace if not specified)")
    ] = None,
    limit: Annotated[
        int, Field(description="Maximum number of kernels to return")
    ] = 20,
) -> dict:
    """Get GPU kernel performance statistics.

    Returns top GPU kernels by total execution time, including call counts,
    average/min/max duration, and percentage of total GPU time.
    """
    if trace_id:
        session = session_manager.get_session(trace_id)
        if session is None:
            raise ValueError(f"Trace not found: {trace_id}")
    else:
        session = _get_active_session()

    analyzer = _get_analyzer(session)
    stats = await analyzer.get_gpu_kernel_stats(limit=limit)

    return {
        "trace_id": session.id,
        "trace_name": session.display_name,
        "kernel_count": len(stats),
        "kernels": [
            {
                "name": s.kernel_name,
                "call_count": s.call_count,
                "total_duration_ns": s.total_duration_ns,
                "total_duration_formatted": _format_duration(s.total_duration_ns),
                "avg_duration_ns": s.avg_duration_ns,
                "avg_duration_formatted": _format_duration(int(s.avg_duration_ns)),
                "min_duration_ns": s.min_duration_ns,
                "max_duration_ns": s.max_duration_ns,
                "percentage_of_total": round(s.percentage_of_total, 2),
            }
            for s in stats
        ],
    }


@mcp.tool()
async def generate_report(
    trace_id: Annotated[
        Optional[str], Field(description="Trace ID (uses active trace if not specified)")
    ] = None,
    format: Annotated[
        str, Field(description="Report format: markdown, json, or text")
    ] = "markdown",
) -> dict:
    """Generate a performance report for a trace.

    Creates a comprehensive report including trace summary and top GPU kernels.
    """
    if trace_id:
        session = session_manager.get_session(trace_id)
        if session is None:
            raise ValueError(f"Trace not found: {trace_id}")
    else:
        session = _get_active_session()

    analyzer = _get_analyzer(session)
    summary = await analyzer.get_summary()
    gpu_stats = await analyzer.get_gpu_kernel_stats()

    try:
        report_format = ReportFormat(format.lower())
    except ValueError:
        report_format = ReportFormat.MARKDOWN

    report = report_generator.generate_summary_report(
        summary=summary,
        gpu_stats=gpu_stats,
        trace_name=session.display_name,
        format=report_format,
    )

    return {
        "trace_id": session.id,
        "format": report_format.value,
        "report": report,
    }


@mcp.tool()
async def compare_traces(
    baseline_id: Annotated[str, Field(description="Trace ID for the baseline")],
    comparison_id: Annotated[str, Field(description="Trace ID to compare against baseline")],
    format: Annotated[
        str, Field(description="Report format: markdown, json, or text")
    ] = "markdown",
    threshold_percent: Annotated[
        float, Field(description="Minimum percentage change to report")
    ] = 5.0,
) -> dict:
    """Compare two traces to identify performance regressions and improvements.

    Compares GPU kernel performance between a baseline and comparison trace,
    reporting significant changes above the threshold.
    """
    baseline_session = session_manager.get_session(baseline_id)
    if baseline_session is None:
        raise ValueError(f"Baseline trace not found: {baseline_id}")

    comparison_session = session_manager.get_session(comparison_id)
    if comparison_session is None:
        raise ValueError(f"Comparison trace not found: {comparison_id}")

    baseline_analyzer = _get_analyzer(baseline_session)
    comparison_analyzer = _get_analyzer(comparison_session)

    baseline_summary = await baseline_analyzer.get_summary()
    baseline_kernels = await baseline_analyzer.get_gpu_kernel_stats()
    comparison_summary = await comparison_analyzer.get_summary()
    comparison_kernels = await comparison_analyzer.get_gpu_kernel_stats()

    comparison = report_generator.compare_traces(
        baseline_summary=baseline_summary,
        baseline_kernels=baseline_kernels,
        comparison_summary=comparison_summary,
        comparison_kernels=comparison_kernels,
        threshold_percent=threshold_percent,
    )

    try:
        report_format = ReportFormat(format.lower())
    except ValueError:
        report_format = ReportFormat.MARKDOWN

    report = report_generator.generate_comparison_report(
        comparison=comparison,
        baseline_name=baseline_session.display_name,
        comparison_name=comparison_session.display_name,
        format=report_format,
    )

    return {
        "baseline_id": baseline_id,
        "comparison_id": comparison_id,
        "duration_change_percent": round(comparison.duration_change_percent, 2),
        "regression_count": len(comparison.kernel_regressions),
        "improvement_count": len(comparison.kernel_improvements),
        "format": report_format.value,
        "report": report,
    }


@mcp.tool()
async def list_traces() -> dict:
    """List all loaded trace sessions.

    Returns information about all traces currently loaded in the session.
    """
    sessions = session_manager.list_sessions()
    active_id = session_manager.active_id

    return {
        "trace_count": len(sessions),
        "active_id": active_id,
        "traces": [
            {
                "id": s.id,
                "name": s.display_name,
                "path": str(s.path),
                "loaded_at": s.loaded_at.isoformat(),
                "is_active": s.id == active_id,
            }
            for s in sessions
        ],
    }


@mcp.tool()
async def get_profiling_options() -> dict:
    """Get available profiling configuration options and categories.

    Returns lists of available trace categories, backends, and common
    profiling configurations to help construct profile_application calls.
    """
    return {
        "trace_categories": {
            "rocm": [
                "rocm",
                "rocm_hip_api",
                "rocm_hip_stream",
                "rocm_hsa_api",
                "rocm_kernel_dispatch",
                "rocm_marker_api",
                "rocm_memory_allocate",
                "rocm_memory_copy",
                "rocm_rccl_api",
                "rocm_counter_collection",
                "rocm_page_migration",
                "rocm_scratch_memory",
            ],
            "sampling": [
                "sampling",
                "timer_sampling",
                "overflow_sampling",
            ],
            "host_metrics": [
                "host",
                "cpu_frequency",
                "numa",
                "process_sampling",
                "process_memory_hwm",
                "process_virtual_memory",
                "process_page_fault",
                "process_context_switch",
                "process_user_cpu_time",
                "process_kernel_cpu_time",
            ],
            "device_metrics": [
                "amd_smi",
                "device_busy_gfx",
                "device_busy_mm",
                "device_busy_umc",
                "device_memory_usage",
                "device_power",
                "device_temp",
                "device_vcn_activity",
                "device_jpeg_activity",
            ],
            "thread_metrics": [
                "pthread",
                "thread_cpu_time",
                "thread_wall_time",
                "thread_peak_memory",
                "thread_page_fault",
                "thread_context_switch",
                "thread_hardware_counter",
            ],
            "other": [
                "kokkos",
                "mpi",
                "rccl",
                "python",
                "user",
                "causal",
            ],
        },
        "backends": {
            "use_rocm": "ROCm API and kernel tracing (HIP, HSA, kernels)",
            "use_sampling": "Statistical call-stack sampling",
            "use_amd_smi": "GPU metrics via AMD SMI (power, temp, memory)",
            "use_kokkosp": "Kokkos performance tools integration",
            "use_mpip": "MPI profiling",
            "use_rcclp": "RCCL (ROCm communication library) profiling",
        },
        "sample_types": {
            "cputime": "Sample based on CPU clock (recommended for compute-bound)",
            "realtime": "Sample based on wall clock (includes idle time)",
        },
        "trace_fill_policies": {
            "discard": "Discard new data when buffer is full",
            "ring_buffer": "Overwrite oldest data when buffer is full",
        },
        "common_configurations": {
            "gpu_kernel_analysis": {
                "description": "Profile GPU kernels and HIP API calls",
                "options": {
                    "trace": True,
                    "use_rocm": True,
                    "enable_categories": ["rocm_kernel_dispatch", "rocm_hip_api"],
                },
            },
            "cpu_hotspot_analysis": {
                "description": "Find CPU hotspots via sampling",
                "options": {
                    "sample": "cputime",
                    "sampling_freq": 100,
                    "use_sampling": True,
                },
            },
            "gpu_power_thermal": {
                "description": "Monitor GPU power and temperature",
                "options": {
                    "device": True,
                    "use_amd_smi": True,
                },
            },
            "memory_analysis": {
                "description": "Track memory allocations and copies",
                "options": {
                    "trace": True,
                    "use_rocm": True,
                    "enable_categories": ["rocm_memory_allocate", "rocm_memory_copy"],
                },
            },
            "full_trace": {
                "description": "Comprehensive profiling (large traces)",
                "options": {
                    "trace": True,
                    "profile": True,
                    "use_rocm": True,
                    "host": True,
                    "device": True,
                },
            },
        },
    }


def _format_duration(ns: int) -> str:
    """Format duration in human-readable form."""
    if ns < 1_000:
        return f"{ns} ns"
    elif ns < 1_000_000:
        return f"{ns / 1_000:.2f} us"
    elif ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    else:
        return f"{ns / 1_000_000_000:.2f} s"


@mcp.resource("trace://{trace_id}/metadata")
async def get_trace_metadata(trace_id: str) -> str:
    """Get metadata for a specific trace."""
    import json

    session = session_manager.get_session(trace_id)
    if session is None:
        return json.dumps({"error": f"Trace not found: {trace_id}"})

    analyzer = _get_analyzer(session)
    metadata = await analyzer.get_metadata()

    return json.dumps(metadata, indent=2)


@mcp.resource("trace://{trace_id}/schema")
async def get_trace_schema(trace_id: str) -> str:
    """Get the database schema for a specific trace."""
    import json

    session = session_manager.get_session(trace_id)
    if session is None:
        return json.dumps({"error": f"Trace not found: {trace_id}"})

    analyzer = _get_analyzer(session)
    schema = await analyzer.get_schema()

    return json.dumps(schema, indent=2)


@mcp.resource("trace://active/info")
async def get_active_trace_info() -> str:
    """Get information about the currently active trace."""
    import json

    session = session_manager.active_session
    if session is None:
        return json.dumps({"error": "No active trace"})

    analyzer = _get_analyzer(session)
    summary = await analyzer.get_summary()

    return json.dumps(
        {
            "id": session.id,
            "name": session.display_name,
            "path": str(session.path),
            "loaded_at": session.loaded_at.isoformat(),
            "duration_ns": summary.duration_ns,
            "slice_count": summary.slice_count,
            "gpu_slice_count": summary.gpu_slice_count,
        },
        indent=2,
    )


@mcp.resource("session://traces")
async def get_all_traces() -> str:
    """Get information about all loaded traces."""
    import json

    sessions = session_manager.list_sessions()
    active_id = session_manager.active_id

    return json.dumps(
        {
            "trace_count": len(sessions),
            "active_id": active_id,
            "traces": [
                {
                    "id": s.id,
                    "name": s.display_name,
                    "path": str(s.path),
                    "loaded_at": s.loaded_at.isoformat(),
                    "is_active": s.id == active_id,
                }
                for s in sessions
            ],
        },
        indent=2,
    )


def run_server():
    """Run the MCP server."""
    mcp.run()
