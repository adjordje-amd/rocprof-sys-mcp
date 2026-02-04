"""Tests for profiling runner."""

import pytest

from rocprof_sys_mcp.profiling.runner import ProfilingConfig, ProfilingResult, ProfilingRunner


class TestProfilingConfig:
    def test_default_values(self):
        config = ProfilingConfig(command="./app")
        assert config.command == "./app"
        assert config.args == []
        assert config.profile is True
        assert config.trace is True
        assert config.timeout_seconds is None
        assert config.sample is None
        assert config.host is False
        assert config.device is False
        assert config.use_rocm is None

    def test_extended_config(self):
        config = ProfilingConfig(
            command="./app",
            args=["--input", "data.bin"],
            sample="cputime",
            sampling_freq=100.0,
            host=True,
            device=True,
            use_rocm=True,
            use_amd_smi=True,
            enable_categories=["rocm_kernel_dispatch", "rocm_hip_api"],
            trace_duration=60.0,
            gpus=["0", "1"],
        )
        assert config.sample == "cputime"
        assert config.sampling_freq == 100.0
        assert config.host is True
        assert config.device is True
        assert config.use_rocm is True
        assert config.use_amd_smi is True
        assert config.enable_categories == ["rocm_kernel_dispatch", "rocm_hip_api"]
        assert config.trace_duration == 60.0
        assert config.gpus == ["0", "1"]


class TestProfilingResult:
    def test_success_result(self):
        result = ProfilingResult(success=True, return_code=0)
        assert result.success is True
        assert result.error_message is None

    def test_failure_result(self):
        result = ProfilingResult(
            success=False,
            return_code=1,
            error_message="Command failed",
        )
        assert result.success is False
        assert result.error_message == "Command failed"


class TestProfilingRunner:
    def test_find_rocprof_not_in_path(self):
        runner = ProfilingRunner()
        # This may or may not find rocprof-sys-run depending on the system
        # Just verify it doesn't crash
        result = runner._find_rocprof_sys_run()
        assert result is None or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_run_command_not_found(self, tmp_path):
        runner = ProfilingRunner(output_base=tmp_path)
        config = ProfilingConfig(command="/nonexistent/command")

        result = await runner.run(config)

        # Will fail either because rocprof-sys-run not found or command not found
        assert result.success is False
