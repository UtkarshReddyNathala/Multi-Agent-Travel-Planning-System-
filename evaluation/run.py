"""Command line entry point:  python -m evaluation.run [--mode offline|live] ..."""

import argparse
import json
import sys
from pathlib import Path

from evaluation.dataset import case_ids
from evaluation.runner import run_evaluation

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _print_report(report: dict) -> None:
    print(f"\nTripMate evaluation | mode={report['mode']} | llm={report['llm']} | mcp={report['mcp_tools']}")
    for note in report["notes"]:
        print(f"  NOTE: {note}")

    print(f"\n{'case':28} {'guardrail':10} {'routing':9} {'constraints':12} {'structured':11} {'answer':9} {'sec':>6}")
    for row in report["cases"]:
        def cell(name: str) -> str:
            return row["metrics"].get(name, {}).get("status", "-")

        latency = "-" if row["latency_s"] is None else f"{row['latency_s']:.2f}"
        print(
            f"{row['id']:28} {cell('guardrail'):10} {cell('routing'):9} {cell('constraints'):12} "
            f"{cell('structured_output'):11} {cell('answer_quality'):9} {latency:>6}"
        )
        for name, metric in row["metrics"].items():
            if metric["status"] == "fail":
                print(f"    x {name}: {metric['detail']}")
        if row["error"]:
            print(f"    ! crashed: {row['error']}")

    print("\nSummary (skipped cases are not counted in the pass rate)")
    for name, stats in report["summary"].items():
        if name in ("latency_s", "judge"):
            continue
        rate = "n/a" if stats["pass_rate"] is None else f"{stats['pass_rate']:.0%}"
        print(f"  {name:18} pass {stats['passed']:>2} | fail {stats['failed']:>2} | skipped {stats['skipped']:>2} | rate {rate}")
    if "latency_s" in report["summary"]:
        lat = report["summary"]["latency_s"]
        print(f"  latency (s)        mean {lat['mean']} | p50 {lat['p50']} | p95 {lat['p95']} | max {lat['max']}")
    if "judge" in report["summary"]:
        print(f"  judge (1-5, subjective) {report['summary']['judge']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the TripMate AI workflow.")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--judge", action="store_true", help="LLM-as-a-judge scoring (live mode only)")
    parser.add_argument("--real-mcp", action="store_true", help="live mode: call the real MCP tools")
    parser.add_argument("--case", action="append", choices=case_ids(), help="run only this case (repeatable)")
    parser.add_argument("--no-save", action="store_true", help="do not write a JSON report")
    args = parser.parse_args(argv)

    try:
        report = run_evaluation(args.mode, args.judge, args.real_mcp, args.case)
    except ValueError as exc:
        parser.error(str(exc))

    _print_report(report)

    if not args.no_save:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = report["run_at"].replace(":", "").replace("-", "")[:15]
        path = RESULTS_DIR / f"{args.mode}-{stamp}.json"
        path.write_text(json.dumps(report, indent=2))
        print(f"\nSaved: {path.relative_to(RESULTS_DIR.parent.parent)}")

    failed = any(s["failed"] for k, s in report["summary"].items() if k not in ("latency_s", "judge"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
