"""Performance report generation."""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..traces.analyzer import GpuKernelStats, TraceSummary

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """Supported report formats."""

    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"


@dataclass
class TraceComparison:
    """Comparison between two traces."""

    baseline_summary: TraceSummary
    comparison_summary: TraceSummary
    baseline_kernels: list[GpuKernelStats]
    comparison_kernels: list[GpuKernelStats]
    duration_change_percent: float
    kernel_regressions: list[dict]
    kernel_improvements: list[dict]


class ReportGenerator:
    """Generates performance reports from trace data."""

    def __init__(self):
        pass

    def _format_duration(self, ns: int) -> str:
        """Format duration in human-readable form."""
        if ns < 1_000:
            return f"{ns} ns"
        elif ns < 1_000_000:
            return f"{ns / 1_000:.2f} us"
        elif ns < 1_000_000_000:
            return f"{ns / 1_000_000:.2f} ms"
        else:
            return f"{ns / 1_000_000_000:.2f} s"

    def generate_summary_report(
        self,
        summary: TraceSummary,
        gpu_stats: list[GpuKernelStats],
        trace_name: str,
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """Generate a summary report for a trace."""
        if format == ReportFormat.MARKDOWN:
            return self._generate_markdown_summary(summary, gpu_stats, trace_name)
        elif format == ReportFormat.JSON:
            return self._generate_json_summary(summary, gpu_stats, trace_name)
        else:
            return self._generate_text_summary(summary, gpu_stats, trace_name)

    def _generate_markdown_summary(
        self,
        summary: TraceSummary,
        gpu_stats: list[GpuKernelStats],
        trace_name: str,
    ) -> str:
        """Generate markdown format summary."""
        lines = [
            f"# Performance Report: {trace_name}",
            "",
            f"*Generated: {datetime.now().isoformat()}*",
            "",
            "## Overview",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Duration | {self._format_duration(summary.duration_ns)} |",
            f"| Total Slices | {summary.slice_count:,} |",
            f"| GPU Slices | {summary.gpu_slice_count:,} |",
            f"| Processes | {summary.process_count} |",
            f"| Threads | {summary.thread_count} |",
            f"| Tracks | {summary.track_count} |",
            "",
        ]

        if gpu_stats:
            lines.extend(
                [
                    "## Top GPU Kernels",
                    "",
                    "| Kernel | Calls | Total Time | Avg Time | % of GPU |",
                    "|--------|-------|------------|----------|----------|",
                ]
            )
            for stat in gpu_stats[:10]:
                lines.append(
                    f"| {stat.kernel_name[:40]} | {stat.call_count:,} | "
                    f"{self._format_duration(stat.total_duration_ns)} | "
                    f"{self._format_duration(int(stat.avg_duration_ns))} | "
                    f"{stat.percentage_of_total:.1f}% |"
                )
            lines.append("")

        if summary.tables:
            lines.extend(
                [
                    "## Available Tables",
                    "",
                    "```",
                    ", ".join(summary.tables[:20]),
                    "```" if len(summary.tables) <= 20 else f"... and {len(summary.tables) - 20} more",
                    "",
                ]
            )

        return "\n".join(lines)

    def _generate_json_summary(
        self,
        summary: TraceSummary,
        gpu_stats: list[GpuKernelStats],
        trace_name: str,
    ) -> str:
        """Generate JSON format summary."""
        data = {
            "trace_name": trace_name,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "duration_ns": summary.duration_ns,
                "duration_formatted": self._format_duration(summary.duration_ns),
                "slice_count": summary.slice_count,
                "gpu_slice_count": summary.gpu_slice_count,
                "process_count": summary.process_count,
                "thread_count": summary.thread_count,
                "track_count": summary.track_count,
                "tables": summary.tables,
            },
            "gpu_kernels": [
                {
                    "name": stat.kernel_name,
                    "call_count": stat.call_count,
                    "total_duration_ns": stat.total_duration_ns,
                    "avg_duration_ns": stat.avg_duration_ns,
                    "min_duration_ns": stat.min_duration_ns,
                    "max_duration_ns": stat.max_duration_ns,
                    "percentage_of_total": stat.percentage_of_total,
                }
                for stat in gpu_stats
            ],
        }
        return json.dumps(data, indent=2)

    def _generate_text_summary(
        self,
        summary: TraceSummary,
        gpu_stats: list[GpuKernelStats],
        trace_name: str,
    ) -> str:
        """Generate plain text format summary."""
        lines = [
            f"Performance Report: {trace_name}",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            "",
            "Overview",
            "-" * 40,
            f"  Total Duration: {self._format_duration(summary.duration_ns)}",
            f"  Total Slices:   {summary.slice_count:,}",
            f"  GPU Slices:     {summary.gpu_slice_count:,}",
            f"  Processes:      {summary.process_count}",
            f"  Threads:        {summary.thread_count}",
            f"  Tracks:         {summary.track_count}",
            "",
        ]

        if gpu_stats:
            lines.extend(
                [
                    "Top GPU Kernels",
                    "-" * 40,
                ]
            )
            for i, stat in enumerate(gpu_stats[:10], 1):
                lines.append(
                    f"  {i}. {stat.kernel_name[:35]}"
                    f"\n      Calls: {stat.call_count:,}, "
                    f"Total: {self._format_duration(stat.total_duration_ns)}, "
                    f"Avg: {self._format_duration(int(stat.avg_duration_ns))}"
                )
            lines.append("")

        return "\n".join(lines)

    def generate_comparison_report(
        self,
        comparison: TraceComparison,
        baseline_name: str,
        comparison_name: str,
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> str:
        """Generate a comparison report between two traces."""
        if format == ReportFormat.MARKDOWN:
            return self._generate_markdown_comparison(
                comparison, baseline_name, comparison_name
            )
        elif format == ReportFormat.JSON:
            return self._generate_json_comparison(
                comparison, baseline_name, comparison_name
            )
        else:
            return self._generate_text_comparison(
                comparison, baseline_name, comparison_name
            )

    def _generate_markdown_comparison(
        self,
        comparison: TraceComparison,
        baseline_name: str,
        comparison_name: str,
    ) -> str:
        """Generate markdown comparison report."""
        change_indicator = (
            "slower" if comparison.duration_change_percent > 0 else "faster"
        )
        change_value = abs(comparison.duration_change_percent)

        lines = [
            f"# Trace Comparison Report",
            "",
            f"*Generated: {datetime.now().isoformat()}*",
            "",
            f"**Baseline:** {baseline_name}  ",
            f"**Comparison:** {comparison_name}",
            "",
            "## Summary",
            "",
            f"| Metric | Baseline | Comparison | Change |",
            f"|--------|----------|------------|--------|",
            f"| Duration | {self._format_duration(comparison.baseline_summary.duration_ns)} | "
            f"{self._format_duration(comparison.comparison_summary.duration_ns)} | "
            f"{change_value:.1f}% {change_indicator} |",
            f"| GPU Slices | {comparison.baseline_summary.gpu_slice_count:,} | "
            f"{comparison.comparison_summary.gpu_slice_count:,} | - |",
            "",
        ]

        if comparison.kernel_regressions:
            lines.extend(
                [
                    "## Regressions (slower)",
                    "",
                    "| Kernel | Baseline | Comparison | Change |",
                    "|--------|----------|------------|--------|",
                ]
            )
            for reg in comparison.kernel_regressions[:10]:
                lines.append(
                    f"| {reg['name'][:30]} | {self._format_duration(reg['baseline_ns'])} | "
                    f"{self._format_duration(reg['comparison_ns'])} | "
                    f"+{reg['change_percent']:.1f}% |"
                )
            lines.append("")

        if comparison.kernel_improvements:
            lines.extend(
                [
                    "## Improvements (faster)",
                    "",
                    "| Kernel | Baseline | Comparison | Change |",
                    "|--------|----------|------------|--------|",
                ]
            )
            for imp in comparison.kernel_improvements[:10]:
                lines.append(
                    f"| {imp['name'][:30]} | {self._format_duration(imp['baseline_ns'])} | "
                    f"{self._format_duration(imp['comparison_ns'])} | "
                    f"{imp['change_percent']:.1f}% |"
                )
            lines.append("")

        return "\n".join(lines)

    def _generate_json_comparison(
        self,
        comparison: TraceComparison,
        baseline_name: str,
        comparison_name: str,
    ) -> str:
        """Generate JSON comparison report."""
        data = {
            "generated_at": datetime.now().isoformat(),
            "baseline": {
                "name": baseline_name,
                "duration_ns": comparison.baseline_summary.duration_ns,
                "gpu_slice_count": comparison.baseline_summary.gpu_slice_count,
            },
            "comparison": {
                "name": comparison_name,
                "duration_ns": comparison.comparison_summary.duration_ns,
                "gpu_slice_count": comparison.comparison_summary.gpu_slice_count,
            },
            "duration_change_percent": comparison.duration_change_percent,
            "regressions": comparison.kernel_regressions,
            "improvements": comparison.kernel_improvements,
        }
        return json.dumps(data, indent=2)

    def _generate_text_comparison(
        self,
        comparison: TraceComparison,
        baseline_name: str,
        comparison_name: str,
    ) -> str:
        """Generate text comparison report."""
        change_indicator = (
            "slower" if comparison.duration_change_percent > 0 else "faster"
        )
        change_value = abs(comparison.duration_change_percent)

        lines = [
            "Trace Comparison Report",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            "",
            f"Baseline:   {baseline_name}",
            f"Comparison: {comparison_name}",
            "",
            "Summary",
            "-" * 40,
            f"  Baseline Duration:   {self._format_duration(comparison.baseline_summary.duration_ns)}",
            f"  Comparison Duration: {self._format_duration(comparison.comparison_summary.duration_ns)}",
            f"  Change: {change_value:.1f}% {change_indicator}",
            "",
        ]

        if comparison.kernel_regressions:
            lines.extend(["Regressions (slower)", "-" * 40])
            for reg in comparison.kernel_regressions[:10]:
                lines.append(
                    f"  {reg['name'][:30]}: +{reg['change_percent']:.1f}%"
                )
            lines.append("")

        if comparison.kernel_improvements:
            lines.extend(["Improvements (faster)", "-" * 40])
            for imp in comparison.kernel_improvements[:10]:
                lines.append(
                    f"  {imp['name'][:30]}: {imp['change_percent']:.1f}%"
                )
            lines.append("")

        return "\n".join(lines)

    def compare_traces(
        self,
        baseline_summary: TraceSummary,
        baseline_kernels: list[GpuKernelStats],
        comparison_summary: TraceSummary,
        comparison_kernels: list[GpuKernelStats],
        threshold_percent: float = 5.0,
    ) -> TraceComparison:
        """Compare two traces and identify regressions/improvements."""
        baseline_duration = baseline_summary.duration_ns
        comparison_duration = comparison_summary.duration_ns

        if baseline_duration > 0:
            duration_change = (
                (comparison_duration - baseline_duration) / baseline_duration * 100
            )
        else:
            duration_change = 0.0

        baseline_kernel_map = {k.kernel_name: k for k in baseline_kernels}
        comparison_kernel_map = {k.kernel_name: k for k in comparison_kernels}

        regressions = []
        improvements = []

        for name, comp_kernel in comparison_kernel_map.items():
            if name in baseline_kernel_map:
                base_kernel = baseline_kernel_map[name]
                if base_kernel.avg_duration_ns > 0:
                    change = (
                        (comp_kernel.avg_duration_ns - base_kernel.avg_duration_ns)
                        / base_kernel.avg_duration_ns
                        * 100
                    )
                    if change > threshold_percent:
                        regressions.append(
                            {
                                "name": name,
                                "baseline_ns": int(base_kernel.avg_duration_ns),
                                "comparison_ns": int(comp_kernel.avg_duration_ns),
                                "change_percent": change,
                            }
                        )
                    elif change < -threshold_percent:
                        improvements.append(
                            {
                                "name": name,
                                "baseline_ns": int(base_kernel.avg_duration_ns),
                                "comparison_ns": int(comp_kernel.avg_duration_ns),
                                "change_percent": change,
                            }
                        )

        regressions.sort(key=lambda x: x["change_percent"], reverse=True)
        improvements.sort(key=lambda x: x["change_percent"])

        return TraceComparison(
            baseline_summary=baseline_summary,
            comparison_summary=comparison_summary,
            baseline_kernels=baseline_kernels,
            comparison_kernels=comparison_kernels,
            duration_change_percent=duration_change,
            kernel_regressions=regressions,
            kernel_improvements=improvements,
        )
