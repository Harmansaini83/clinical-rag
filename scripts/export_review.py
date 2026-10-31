"""Export review CSV CLI tool."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sqlmodel import Session, select
from app.db import get_engine
from app.models import Run
from app.services.review import export_review_csv


def main():
    parser = argparse.ArgumentParser(description="Export evaluation runs to CSV for manual review.")
    parser.add_argument("--run-id", type=str, help="Specific Run ID to export.")
    parser.add_argument("--latest", action="store_true", help="Export latest run for each variant (A and C).")
    parser.add_argument("--output", type=str, required=True, help="Destination CSV file path.")

    args = parser.parse_args()
    engine = get_engine()

    run_ids = []
    if args.run_id:
        run_ids.append(args.run_id)
    elif args.latest:
        with Session(engine) as session:
            # Find latest run for Variant A and Variant C
            for variant in ["A", "C"]:
                latest_run = session.exec(
                    select(Run).where(Run.variant == variant).order_by(Run.created_at.desc())
                ).first()
                if latest_run:
                    run_ids.append(latest_run.run_id)

        if not run_ids:
            print("No runs found in database.")
            sys.exit(1)
    else:
        print("Must specify either --run-id <RUN_ID> or --latest")
        sys.exit(1)

    out_path = Path(args.output)
    count = export_review_csv(run_ids, out_path, engine=engine)
    print(f"Exported {count} review rows for run(s) {run_ids} to: {out_path}")


if __name__ == "__main__":
    main()
