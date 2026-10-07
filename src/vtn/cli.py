import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="VendorTextNormalizer CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommands
    subparsers.add_parser("gen", help="Generate raw vendor feeds")
    subparsers.add_parser("ingest", help="Ingest raw feeds")
    subparsers.add_parser("resolve", help="Entity resolution and crosswalk")
    subparsers.add_parser("features", help="Extract text and numeric features")
    subparsers.add_parser("qc", help="Run quality check gate")
    subparsers.add_parser("report", help="Generate findings report")
    subparsers.add_parser("bench", help="Run pandas vs polars benchmarks")
    subparsers.add_parser("all", help="Run full pipeline end to end")

    args = parser.parse_args()

    if args.command == "gen":
        print("Running gen...")
    elif args.command == "ingest":
        print("Running ingest...")
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
    elif args.command == "all":
        print("Running all...")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
