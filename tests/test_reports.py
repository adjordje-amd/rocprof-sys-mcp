"""Tests for report generation."""

import json

import pytest

from rocprof_sys_mcp.reports.generator import (
    ReportFormat,
    ReportGenerator,
    TraceComparison,
)
from rocprof_sys_mcp.traces.analyzer import GpuKernelStats, TraceSummary


@pytest.fixture
def sample_summary():
    return TraceSummary(
        duration_ns=1_000_000_000,  # 1 second
        slice_count=1000,
        track_count=10,
        process_count=2,
        thread_count=8,
        gpu_slice_count=500,
        tables=["slice", "track", "thread", "process"],
    )


@pytest.fixture
def sample_gpu_stats():
    return [
        GpuKernelStats(
            kernel_name="kernel_a",
            call_count=100,
            total_duration_ns=500_000_000,
            avg_duration_ns=5_000_000.0,
            min_duration_ns=4_000_000,
            max_duration_ns=6_000_000,
            percentage_of_total=50.0,
        ),
        GpuKernelStats(
            kernel_name="kernel_b",
            call_count=50,
            total_duration_ns=300_000_000,
            avg_duration_ns=6_000_000.0,
            min_duration_ns=5_000_000,
            max_duration_ns=7_000_000,
            percentage_of_total=30.0,
        ),
    ]


class TestReportGenerator:
    def test_format_duration_ns(self):
        gen = ReportGenerator()
        assert gen._format_duration(500) == "500 ns"

    def test_format_duration_us(self):
        gen = ReportGenerator()
        assert gen._format_duration(5_000) == "5.00 us"

    def test_format_duration_ms(self):
        gen = ReportGenerator()
        assert gen._format_duration(5_000_000) == "5.00 ms"

    def test_format_duration_s(self):
        gen = ReportGenerator()
        assert gen._format_duration(5_000_000_000) == "5.00 s"

    def test_generate_markdown_summary(self, sample_summary, sample_gpu_stats):
        gen = ReportGenerator()
        report = gen.generate_summary_report(
            summary=sample_summary,
            gpu_stats=sample_gpu_stats,
            trace_name="test_trace",
            format=ReportFormat.MARKDOWN,
        )

        assert "# Performance Report: test_trace" in report
        assert "1.00 s" in report
        assert "kernel_a" in report
        assert "kernel_b" in report

    def test_generate_json_summary(self, sample_summary, sample_gpu_stats):
        gen = ReportGenerator()
        report = gen.generate_summary_report(
            summary=sample_summary,
            gpu_stats=sample_gpu_stats,
            trace_name="test_trace",
            format=ReportFormat.JSON,
        )

        data = json.loads(report)
        assert data["trace_name"] == "test_trace"
        assert data["summary"]["duration_ns"] == 1_000_000_000
        assert len(data["gpu_kernels"]) == 2

    def test_generate_text_summary(self, sample_summary, sample_gpu_stats):
        gen = ReportGenerator()
        report = gen.generate_summary_report(
            summary=sample_summary,
            gpu_stats=sample_gpu_stats,
            trace_name="test_trace",
            format=ReportFormat.TEXT,
        )

        assert "Performance Report: test_trace" in report
        assert "Total Duration: 1.00 s" in report

    def test_compare_traces_detects_regression(self, sample_summary, sample_gpu_stats):
        gen = ReportGenerator()

        # Create comparison with slower kernels
        comparison_stats = [
            GpuKernelStats(
                kernel_name="kernel_a",
                call_count=100,
                total_duration_ns=600_000_000,  # 20% slower
                avg_duration_ns=6_000_000.0,
                min_duration_ns=5_000_000,
                max_duration_ns=7_000_000,
                percentage_of_total=50.0,
            ),
        ]

        comparison = gen.compare_traces(
            baseline_summary=sample_summary,
            baseline_kernels=sample_gpu_stats,
            comparison_summary=sample_summary,
            comparison_kernels=comparison_stats,
            threshold_percent=5.0,
        )

        assert len(comparison.kernel_regressions) == 1
        assert comparison.kernel_regressions[0]["name"] == "kernel_a"
        assert comparison.kernel_regressions[0]["change_percent"] > 0
