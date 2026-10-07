import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from vtn.gen import generate
from vtn.bench import run_benchmarks
from vtn.ingest_pandas import load_vendor_a, load_vendor_b, normalize_schema
from vtn.resolve import build_crosswalk, resolution_report, build_panel
from vtn.text_features import attach_to_panel
from vtn.qc import run_all
from vtn.discrepancy import generate_findings_report


@contextmanager
def _stage(name: str) -> Iterator[None]:
    """Context manager that prints the wall-clock time for a named pipeline stage."""
    t0 = time.perf_counter()
    print(f"[stage] {name} ...")
    try:
        yield
    finally:
        print(f"[stage] {name} done in {time.perf_counter() - t0:.3f}s")


def _run_resolve() -> None:
    """Loads both vendors, resolves entities and writes the normalized panel."""
    df_a = normalize_schema(load_vendor_a(Path("data/raw/vendor_a.parquet")), "vendor_a")
    df_b = normalize_schema(load_vendor_b(Path("data/raw/vendor_b.jsonl")), "vendor_b")
    cw = build_crosswalk(df_a, df_b)
    rep = resolution_report(cw, df_a, df_b)
    panel = build_panel(df_a, df_b, cw)
    print(f"Resolution complete: match_rate={rep['match_rate']}, panel_rows={len(panel)}")


def main() -> None:
    """Entry point for the ``python -m vtn`` CLI."""
    parser = argparse.ArgumentParser(description="VendorTextNormalizer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("gen", help="Generate raw vendor feeds")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    gen_parser.add_argument("--entities", type=int, default=200, help="Number of entities")
    gen_parser.add_argument("--days", type=int, default=500, help="Number of days")

    subparsers.add_parser("ingest", help="Ingest raw feeds")
    subparsers.add_parser("resolve", help="Entity resolution and crosswalk")
    subparsers.add_parser("features", help="Extract text and numeric features")
    subparsers.add_parser("qc", help="Run quality check gate")
    subparsers.add_parser("report", help="Generate findings report")
    subparsers.add_parser("bench", help="Run pandas vs polars benchmarks")

    all_parser = subparsers.add_parser("all", help="Run full pipeline end to end")
    all_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.command == "gen":
        with _stage("gen"):
            res = generate(seed=args.seed, n_entities=args.entities, n_days=args.days)
            print(f"Generated successfully: {res}")
    elif args.command == "ingest":
        with _stage("ingest+bench"):
            run_benchmarks()
    elif args.command == "resolve":
        with _stage("resolve"):
            _run_resolve()
    elif args.command == "features":
        with _stage("features"):
            summary = attach_to_panel()
            print(f"Features complete: {summary}")
    elif args.command == "qc":
        with _stage("qc"):
            passed, checks = run_all()
            for c in checks:
                status = "PASS" if c['passed'] else "FAIL"
                print(f" - {c['name']}: {status} (observed: {c['observed']}, threshold: {c['threshold']})")
        if not passed:
            print("QC Gate FAILED.")
            sys.exit(1)
        print("QC Gate PASSED.")
    elif args.command == "report":
        with _stage("report"):
            path = generate_findings_report()
            print(f"Findings report written to {path}")
    elif args.command == "bench":
        with _stage("bench"):
            res = run_benchmarks()
            print(f"Benchmark results: {res}")
    elif args.command == "all":
        with _stage("gen"):
            generate(seed=args.seed)
        with _stage("bench"):
            run_benchmarks()
        with _stage("resolve"):
            _run_resolve()
        with _stage("features"):
            attach_to_panel()
        with _stage("qc"):
            passed, _ = run_all()
        with _stage("report"):
            generate_findings_report()
        if not passed:
            print("Pipeline completed but QC failed!")
            sys.exit(1)
        print("All steps completed successfully and QC passed.")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
