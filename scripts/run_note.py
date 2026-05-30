"""
scripts/run_note.py
===================
Quick CLI to test GatorRisk on any single note you type or paste.

Usage:
    # Interactive mode
    python scripts/run_note.py

    # Pass a note directly
    python scripts/run_note.py --note "Patient smokes 2 ppd. BMI 34. Drinks 3 beers nightly."

    # From a text file
    python scripts/run_note.py --file my_note.txt
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="GatorRisk — single note extractor")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--note", type=str, help="Clinical note text (quoted)")
    group.add_argument("--file", type=str, help="Path to a .txt file with the note")
    parser.add_argument("--json", action="store_true", help="Output full JSON instead of summary")
    args = parser.parse_args()

    # Get note text
    if args.note:
        text = args.note
    elif args.file:
        text = Path(args.file).read_text()
    else:
        # Interactive
        print("GatorRisk — Paste your clinical note below.")
        print("Press Enter twice when done:\n")
        lines = []
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        text = "\n".join(lines)

    if not text.strip():
        print("No note text provided.")
        sys.exit(1)

    # Run
    pipeline = Pipeline(use_transformer=False, deidentify=True)
    result = pipeline.run_note(note_id="SINGLE_NOTE", text=text)

    if args.json:
        print(result.to_json())
    else:
        print("\n" + "=" * 55)
        print(result.risk_profile.summary())
        print("\n--- EXTRACTED PROFILE ---")
        profile = result.normalized_profile.to_dict()
        for factor, data in profile.items():
            if factor in ("note_id", "extraction_warnings"):
                continue
            print(f"\n  {factor.upper()}")
            for k, v in data.items():
                if v not in (None, [], "unknown"):
                    print(f"    {k}: {v}")
        print(f"\n  ⏱  Processed in {result.processing_time_ms}ms")


if __name__ == "__main__":
    main()
