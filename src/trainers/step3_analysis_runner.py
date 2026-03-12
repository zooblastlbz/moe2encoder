from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.evaluation.routing_analysis import compare_metrics, compare_routing
from src.utils.io import load_json, save_json, save_text


@dataclass
class Step3Options:
    baseline_dir: str
    post_dir: str
    output_dir: str


class Step3AnalysisRunner:
    def __init__(self, options: Step3Options):
        self.options = options

    @staticmethod
    def _format_top_metric_changes(metric_diff, top_k=10):
        sortable = []
        for k, v in metric_diff.items():
            sortable.append((abs(v["delta"]), k, v))
        sortable.sort(reverse=True)
        lines = []
        for _, name, row in sortable[:top_k]:
            lines.append(
                f"- `{name}`: pre={row['pre']:.6f}, post={row['post']:.6f}, delta={row['delta']:.6f}"
            )
        return lines

    @staticmethod
    def _format_router_changes(route_diff, top_k=10):
        sortable = []
        for k, v in route_diff.items():
            sortable.append((abs(v["expert_distribution_jsd"]), k, v))
        sortable.sort(reverse=True)
        lines = []
        for _, name, row in sortable[:top_k]:
            lines.append(
                "- "
                f"`{name}`: entropy_delta={row['entropy_delta']:.6f}, "
                f"top_share_delta={row['top_expert_share_delta']:.6f}, "
                f"jsd={row['expert_distribution_jsd']:.6f}"
            )
        return lines

    def run(self) -> None:
        baseline_dir = Path(self.options.baseline_dir)
        post_dir = Path(self.options.post_dir)
        output_dir = Path(self.options.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pre_metrics = load_json(baseline_dir / "metrics.json")
        post_metrics = load_json(post_dir / "metrics.json")
        pre_routing = load_json(baseline_dir / "routing_stats.json")
        post_routing = load_json(post_dir / "routing_stats.json")

        metric_diff = compare_metrics(pre_metrics, post_metrics)
        routing_diff = compare_routing(pre_routing, post_routing)

        save_json(output_dir / "metrics_diff.json", metric_diff)
        save_json(output_dir / "routing_diff.json", routing_diff)

        report_lines = []
        report_lines.append("# Step3 Analysis Report")
        report_lines.append("")
        report_lines.append("## Metric Changes (sorted by absolute delta)")
        report_lines.extend(self._format_top_metric_changes(metric_diff, top_k=15))
        report_lines.append("")
        report_lines.append("## Routing Changes (sorted by JSD)")
        report_lines.extend(self._format_router_changes(routing_diff, top_k=15))
        report_lines.append("")
        report_lines.append("## Checks")
        report_lines.append("- Whether representation metrics improve over baseline.")
        report_lines.append("- Whether routing distributions change stably.")
        report_lines.append("- Whether routing changes align with prompt-type analyses.")

        save_text(output_dir / "analysis_report.md", "\n".join(report_lines) + "\n")
        print(f"[Step3] Analysis written to: {output_dir}")


def run_step3(options: Step3Options) -> None:
    Step3AnalysisRunner(options).run()
