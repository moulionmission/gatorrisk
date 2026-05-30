"""
scripts/run_mtsamples.py
========================
Loads the MTSamples Kaggle CSV and runs the GatorRisk pipeline on it.

How to get the data:
    1. Go to https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
    2. Click Download → saves archive.zip
    3. Unzip → you get mtsamples.csv
    4. Move it to: data/mtsamples/mtsamples.csv

Run:
    python scripts/run_mtsamples.py
    python scripts/run_mtsamples.py --specialty "General Medicine"
    python scripts/run_mtsamples.py --all --output data/processed/mtsamples_results.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from modules.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Specialties most likely to contain social history / lifestyle info
SOCIAL_HISTORY_SPECIALTIES = [
    "Consult - History and Phy.",
    "General Medicine",
    "SOAP / Chart / Progress Notes",
    "Discharge Summary",
    "Office Notes",
    "Emergency Room Reports",
    "Cardiovascular / Pulmonary",
    "Endocrinology",
    "Gastroenterology",
    "Nephrology",
    "Psychiatry / Psychology",
]

MTSAMPLES_PATH = Path(__file__).parent.parent / "data" / "mtsamples" / "mtsamples.csv"


def load_mtsamples(specialty: str = None, all_specialties: bool = False) -> pd.DataFrame:
    if not MTSAMPLES_PATH.exists():
        print("\n❌  mtsamples.csv not found!")
        print("    Download it from: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions")
        print(f"    Place it at: {MTSAMPLES_PATH}\n")
        sys.exit(1)

    df = pd.read_csv(MTSAMPLES_PATH)
    df.columns = df.columns.str.strip()
    logger.info(f"Loaded {len(df)} total MTSamples notes")
    logger.info(f"Specialties available: {df['medical_specialty'].nunique()}")

    # Filter
    if all_specialties:
        filtered = df
    elif specialty:
        filtered = df[df["medical_specialty"].str.strip() == specialty]
        logger.info(f"Filtered to '{specialty}': {len(filtered)} notes")
    else:
        filtered = df[df["medical_specialty"].str.strip().isin(SOCIAL_HISTORY_SPECIALTIES)]
        logger.info(f"Filtered to social-history specialties: {len(filtered)} notes")

    # Drop empty transcriptions
    filtered = filtered[filtered["transcription"].notna()]
    filtered = filtered[filtered["transcription"].str.len() > 50]
    logger.info(f"After cleaning: {len(filtered)} notes ready for pipeline")

    return filtered


def run(args):
    df = load_mtsamples(specialty=args.specialty, all_specialties=args.all)

    # Optionally limit for quick testing
    if args.limit:
        df = df.head(args.limit)
        logger.info(f"Limiting to {args.limit} notes (--limit flag)")

    # Convert to pipeline format
    notes = []
    for _, row in df.iterrows():
        notes.append({
            "note_id": f"MTS_{row.name}",
            "text": str(row["transcription"]),
            "specialty": row.get("medical_specialty", ""),
            "description": row.get("description", ""),
        })

    # Run pipeline
    logger.info(f"Running GatorRisk pipeline on {len(notes)} notes...")
    pipeline = Pipeline(use_transformer=False, deidentify=True)
    results = pipeline.run_batch(notes)

    # Summary stats
    scores = [r.risk_profile.composite_score for r in results]
    tiers = [r.risk_profile.composite_tier for r in results]
    from collections import Counter
    tier_counts = Counter(tiers)

    print("\n" + "=" * 55)
    print(f"  GATORRISK — MTSamples Results")
    print("=" * 55)
    print(f"  Notes processed : {len(results)}")
    print(f"  Avg risk score  : {sum(scores)/len(scores):.3f}")
    print(f"  Min / Max       : {min(scores):.2f} / {max(scores):.2f}")
    print(f"  LOW             : {tier_counts.get('LOW', 0)}")
    print(f"  MODERATE        : {tier_counts.get('MODERATE', 0)}")
    print(f"  HIGH            : {tier_counts.get('HIGH', 0)}")
    print(f"  CRITICAL        : {tier_counts.get('CRITICAL', 0)}")
    print("=" * 55)

    # Save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = [r.to_dict() for r in results]
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Results saved → {args.output}")

    # Print sample high-risk notes
    high_risk = [(r, r.risk_profile.composite_score) for r in results
                 if r.risk_profile.composite_tier in ("HIGH", "CRITICAL")]
    if high_risk:
        high_risk.sort(key=lambda x: -x[1])
        print(f"\n  Top 3 highest-risk notes:")
        for r, score in high_risk[:3]:
            print(f"\n  [{r.note_id}] Score: {score:.2f}")
            print(f"  {r.risk_profile.summary()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GatorRisk on MTSamples dataset")
    parser.add_argument("--specialty", type=str, help="Filter to one specialty (exact name)")
    parser.add_argument("--all", action="store_true", help="Run on all specialties")
    parser.add_argument("--limit", type=int, help="Max notes to process (for quick testing)")
    parser.add_argument("--output", type=str, default="data/processed/mtsamples_results.json",
                        help="Output path for JSON results")
    args = parser.parse_args()
    run(args)
