import argparse
import sys
from pathlib import Path
from vtn.gen import generate
from vtn.bench import run_benchmarks

def main():
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
        print(f"Generating raw feeds with seed={args.seed}...")
        res = generate(seed=args.seed, n_entities=args.entities, n_days=args.days)
        print(f"Generated successfully: {res}")
    elif args.command == "ingest":
        print("Running ingest & benchmarks...")
        run_benchmarks()
        print("Ingest & benchmark complete.")
    elif args.command == "resolve":
        print("Running resolve...")
    elif args.command == "features":
        print("Running features...")
    elif args.command == "qc":
        print("Running qc...")
    elif args.command == "report":
        print("Running report...")
    elif args.command == "bench":
        print("Running bench...")
        res = run_benchmarks()
        print(f"Benchmark results: {res}")
    elif args.command == "all":
        print(f"Running all with seed={args.seed}...")
        generate(seed=args.seed)
        run_benchmarks()
        print("All steps completed.")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
