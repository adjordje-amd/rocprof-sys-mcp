"""Profiling execution using rocprof-sys-run."""

import asyncio
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ProfilingResult:
    """Result of a profiling run."""

    success: bool
    trace_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


@dataclass
class ProfilingConfig:
    """Configuration for profiling run."""

    command: str
    args: list[str] = field(default_factory=list)
    working_dir: Optional[str] = None
    output_dir: Optional[str] = None

    # Basic profiling modes
    profile: bool = True
    trace: bool = True
    flat_profile: bool = False

    # Sampling options
    sample: Optional[str] = None  # "cputime" or "realtime"
    sampling_freq: Optional[float] = None  # interrupts per second
    sampling_wait: Optional[float] = None  # delay before first sample (seconds)
    sampling_duration: Optional[float] = None  # duration of sampling (seconds)

    # Host/Device monitoring
    host: bool = False  # Enable host metrics (CPU freq, memory, etc.)
    device: bool = False  # Enable device metrics (GPU temp, memory, etc.)

    # Trace timing control
    trace_wait: Optional[float] = None  # delay before collecting trace (seconds)
    trace_duration: Optional[float] = None  # duration of trace collection (seconds)
    trace_buffer_size: Optional[int] = None  # trace buffer size in KB
    trace_fill_policy: Optional[str] = None  # "discard" or "ring_buffer"

    # Backend options
    use_rocm: Optional[bool] = None  # Enable ROCm API/kernel tracing
    use_sampling: Optional[bool] = None  # Enable statistical sampling
    use_kokkosp: Optional[bool] = None  # Enable Kokkos Tools support
    use_mpip: Optional[bool] = None  # Enable MPI support
    use_rcclp: Optional[bool] = None  # Enable RCCL support
    use_amd_smi: Optional[bool] = None  # Enable AMD SMI metrics

    # ROCm specific options
    rocm_domains: Optional[list[str]] = None  # ROCm domains to trace
    rocm_events: Optional[str] = None  # ROCm events to capture

    # Category filtering
    enable_categories: Optional[list[str]] = None  # Categories to enable
    disable_categories: Optional[list[str]] = None  # Categories to disable

    # Thread options
    trace_thread_locks: bool = False
    trace_thread_barriers: bool = False
    trace_thread_join: bool = False

    # CPU/GPU selection
    cpus: Optional[list[str]] = None  # CPU IDs for freq sampling
    gpus: Optional[list[str]] = None  # GPU IDs for SMI queries

    # Advanced options
    perfetto_annotations: Optional[bool] = None  # Include debug annotations
    perfetto_backend: Optional[str] = None  # Perfetto backend type
    config_file: Optional[str] = None  # Path to config file

    # Environment and execution
    env_vars: dict[str, str] = field(default_factory=dict)
    timeout_seconds: Optional[float] = None


class ProfilingRunner:
    """Executes rocprof-sys-run for application profiling."""

    DEFAULT_OUTPUT_BASE = Path.home() / ".rocprof-sys-mcp" / "traces"

    def __init__(self, output_base: Optional[Path] = None):
        self.output_base = output_base or self.DEFAULT_OUTPUT_BASE
        self.output_base.mkdir(parents=True, exist_ok=True)

    def _find_rocprof_sys_run(self) -> Optional[str]:
        """Find rocprof-sys-run executable."""
        executable = "/home/amd/work/rocm-systems/projects/rocprofiler-systems/build/debug/bin/rocprof-sys-run"
        #executable = shutil.which("rocprof-sys-run")
        if executable:
            return executable
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        candidate = Path(rocm_path) / "bin" / "rocprof-sys-run"
        if candidate.exists():
            return str(candidate)
        return None

    def _find_trace_file(self, output_dir: Path) -> Optional[Path]:
        """Find the Perfetto trace file in the output directory."""
        for pattern in ["*.perfetto-trace", "*.proto", "perfetto-trace.proto"]:
            traces = list(output_dir.rglob(pattern))
            if traces:
                return traces[0]
        for subdir in output_dir.iterdir():
            if subdir.is_dir():
                for pattern in ["*.perfetto-trace", "*.proto"]:
                    traces = list(subdir.rglob(pattern))
                    if traces:
                        return traces[0]
        return None

    async def run(self, config: ProfilingConfig) -> ProfilingResult:
        """Run profiling on the specified application."""
        import time

        start_time = time.monotonic()

        executable = self._find_rocprof_sys_run()
        if not executable:
            return ProfilingResult(
                success=False,
                error_message="rocprof-sys-run not found in PATH or ROCM_PATH",
                return_code=-1,
            )

        if config.output_dir:
            output_dir = Path(config.output_dir)
        else:
            output_dir = Path(tempfile.mkdtemp(dir=self.output_base, prefix="profile_"))

        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [executable]

        # Basic profiling modes
        if config.profile:
            cmd.append("--profile")
        if config.trace:
            cmd.append("--trace")
        if config.flat_profile:
            cmd.append("--flat-profile")

        # Sampling options
        if config.sample:
            cmd.extend(["--sample", config.sample])
        if config.sampling_freq is not None:
            cmd.extend(["--sampling-freq", str(config.sampling_freq)])
        if config.sampling_wait is not None:
            cmd.extend(["--sampling-wait", str(config.sampling_wait)])
        if config.sampling_duration is not None:
            cmd.extend(["--sampling-duration", str(config.sampling_duration)])

        # Host/Device monitoring
        if config.host:
            cmd.append("--host")
        if config.device:
            cmd.append("--device")

        # Trace timing control
        if config.trace_wait is not None:
            cmd.extend(["--trace-wait", str(config.trace_wait)])
        if config.trace_duration is not None:
            cmd.extend(["--trace-duration", str(config.trace_duration)])
        if config.trace_buffer_size is not None:
            cmd.extend(["--trace-buffer-size", str(config.trace_buffer_size)])
        if config.trace_fill_policy:
            cmd.extend(["--trace-fill-policy", config.trace_fill_policy])

        # Backend options
        if config.use_rocm is not None:
            cmd.extend(["--use-rocm", str(config.use_rocm).lower()])
        if config.use_sampling is not None:
            cmd.extend(["--use-sampling", str(config.use_sampling).lower()])
        if config.use_kokkosp is not None:
            cmd.extend(["--use-kokkosp", str(config.use_kokkosp).lower()])
        if config.use_mpip is not None:
            cmd.extend(["--use-mpip", str(config.use_mpip).lower()])
        if config.use_rcclp is not None:
            cmd.extend(["--use-rcclp", str(config.use_rcclp).lower()])
        if config.use_amd_smi is not None:
            cmd.extend(["--use-amd-smi", str(config.use_amd_smi).lower()])

        # ROCm specific options
        if config.rocm_domains:
            cmd.extend(["--rocm-domains", ",".join(config.rocm_domains)])
        if config.rocm_events:
            cmd.extend(["--rocm-events", config.rocm_events])

        # Category filtering
        if config.enable_categories:
            cmd.extend(["--enable-categories", ",".join(config.enable_categories)])
        if config.disable_categories:
            cmd.extend(["--disable-categories", ",".join(config.disable_categories)])

        # Thread options
        if config.trace_thread_locks:
            cmd.append("--trace-thread-locks")
        if config.trace_thread_barriers:
            cmd.append("--trace-thread-barriers")
        if config.trace_thread_join:
            cmd.append("--trace-thread-join")

        # CPU/GPU selection
        if config.cpus:
            cmd.extend(["--cpus"] + config.cpus)
        if config.gpus:
            cmd.extend(["--gpus"] + config.gpus)

        # Advanced options
        if config.perfetto_annotations is not None:
            cmd.extend(["--perfetto-annotations", str(config.perfetto_annotations).lower()])
        if config.perfetto_backend:
            cmd.extend(["--perfetto-backend", config.perfetto_backend])
        if config.config_file:
            cmd.extend(["--config", config.config_file])

        # Output directory
        cmd.extend(["--output", str(output_dir)])

        cmd.append("--")
        cmd.append(config.command)
        cmd.extend(config.args)

        env = os.environ.copy()
        env.update(config.env_vars)

        working_dir = config.working_dir or os.getcwd()

        logger.info(f"Running profiling: {' '.join(cmd)}")
        logger.info(f"Output directory: {output_dir}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=config.timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ProfilingResult(
                    success=False,
                    output_dir=output_dir,
                    error_message=f"Profiling timed out after {config.timeout_seconds} seconds",
                    return_code=-1,
                    duration_seconds=time.monotonic() - start_time,
                )

            duration = time.monotonic() - start_time
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            if process.returncode != 0:
                return ProfilingResult(
                    success=False,
                    output_dir=output_dir,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    return_code=process.returncode,
                    error_message=f"Profiling failed with return code {process.returncode}",
                    duration_seconds=duration,
                )

            trace_path = self._find_trace_file(output_dir)

            return ProfilingResult(
                success=True,
                trace_path=trace_path,
                output_dir=output_dir,
                stdout=stdout_str,
                stderr=stderr_str,
                return_code=process.returncode,
                duration_seconds=duration,
            )

        except FileNotFoundError:
            return ProfilingResult(
                success=False,
                error_message=f"Command not found: {config.command}",
                return_code=-1,
                duration_seconds=time.monotonic() - start_time,
            )
        except Exception as e:
            logger.exception("Profiling failed")
            return ProfilingResult(
                success=False,
                error_message=str(e),
                return_code=-1,
                duration_seconds=time.monotonic() - start_time,
            )
