"""
Evaluation Module
=================
Computes Precision, Recall, and F1 per risk factor by comparing
pipeline output against ground-truth annotated notes.

Usage:
    python evaluation/evaluator.py

Metrics:
    - Per-factor P/R/F1 for STATUS classification
    - Per-factor numeric value accuracy (within tolerance)
    - Overall pipeline F1 (macro-averaged)
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.pipeline import Pipeline

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "sample_notes.json"


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class FactorMetrics:
    factor: str
    status_accuracy: float      # % correct status (current/former/never)
    value_accuracy: float       # % numeric values within tolerance
    f1: float                   # composite F1 for this factor
    n: int                      # number of notes evaluated


@dataclass
class EvalReport:
    total_notes: int
    macro_f1: float
    factor_metrics: List[FactorMetrics]

    def print(self):
        print("=" * 65)
        print(f"EVALUATION REPORT  ({self.total_notes} notes)")
        print("=" * 65)
        print(f"{'Factor':<22} {'Status Acc':>12} {'Value Acc':>12} {'F1':>8}")
        print("-" * 65)
        for m in sorted(self.factor_metrics, key=lambda x: -x.f1):
            print(f"  {m.factor:<20} {m.status_accuracy:>11.1%} {m.value_accuracy:>11.1%} {m.f1:>7.3f}")
        print("-" * 65)
        print(f"  {'MACRO AVG':<20} {'':>11} {'':>11} {self.macro_f1:>7.3f}")
        print("=" * 65)


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────

class Evaluator:
    """
    Evaluates pipeline predictions against ground truth annotations.

    Ground truth format (per note):
    {
      "smoking": {"status": "current", "value": 1.5, "unit": "ppd", ...},
      "alcohol": {"status": "never"},
      ...
    }
    """

    FACTORS = ["smoking", "alcohol", "bmi", "physical_activity", "sleep", "diet", "drug_use"]
    NUMERIC_TOLERANCE = 0.15  # 15% tolerance for numeric value comparison

    def __init__(self, pipeline: Optional[Pipeline] = None):
        self.pipeline = pipeline or Pipeline(use_transformer=False)

    def evaluate_file(self, filepath: str = str(DATA_PATH)) -> EvalReport:
        """Load annotated notes from JSON and evaluate."""
        with open(filepath) as f:
            notes = json.load(f)

        # Filter notes that have ground truth
        annotated = [n for n in notes if "ground_truth" in n]
        logger.info(f"Evaluating on {len(annotated)} annotated notes")

        return self.evaluate(annotated)

    def evaluate(self, annotated_notes: List[dict]) -> EvalReport:
        """
        Run evaluation on a list of annotated notes.

        Args:
            annotated_notes: List of dicts with 'note_id', 'text', 'ground_truth'

        Returns:
            EvalReport with per-factor and macro metrics
        """
        # Collect per-factor results
        factor_results: Dict[str, Dict[str, List]] = {
            f: {"status_correct": [], "value_correct": []}
            for f in self.FACTORS
        }

        for note in annotated_notes:
            note_id = note["note_id"]
            text = note["text"]
            gt = note.get("ground_truth", {})

            # Run pipeline
            output = self.pipeline.run_note(note_id=note_id, text=text)
            profile_dict = output.normalized_profile.to_dict()

            # Compare each factor
            for factor in self.FACTORS:
                gt_factor = gt.get(factor, {})
                pred_factor = profile_dict.get(factor, {})

                status_correct = self._compare_status(
                    gt_factor.get("status"), pred_factor.get("status")
                )
                value_correct = self._compare_value(gt_factor, pred_factor, factor)

                factor_results[factor]["status_correct"].append(status_correct)
                factor_results[factor]["value_correct"].append(value_correct)

        # Compute metrics per factor
        factor_metrics = []
        for factor in self.FACTORS:
            sc = factor_results[factor]["status_correct"]
            vc = factor_results[factor]["value_correct"]

            status_acc = sum(sc) / len(sc) if sc else 0.0
            value_acc = sum(vc) / len(vc) if vc else 0.0
            # Simple F1 approximation: harmonic mean of status and value accuracy
            if status_acc + value_acc > 0:
                f1 = 2 * (status_acc * value_acc) / (status_acc + value_acc)
            else:
                f1 = 0.0

            factor_metrics.append(FactorMetrics(
                factor=factor,
                status_accuracy=round(status_acc, 4),
                value_accuracy=round(value_acc, 4),
                f1=round(f1, 4),
                n=len(sc),
            ))

        macro_f1 = round(sum(m.f1 for m in factor_metrics) / len(factor_metrics), 4)

        return EvalReport(
            total_notes=len(annotated_notes),
            macro_f1=macro_f1,
            factor_metrics=factor_metrics,
        )

    # ── Comparison Helpers ─────────────────────────

    def _compare_status(self, gt_status: Optional[str], pred_status: Optional[str]) -> bool:
        """Exact match on status (current/former/never/unknown)."""
        if gt_status is None:
            return True  # Not annotated — skip
        if pred_status is None:
            return False
        return gt_status.lower() == pred_status.lower()

    def _compare_value(self, gt: dict, pred: dict, factor: str) -> bool:
        """
        Numeric value comparison with tolerance.
        For non-numeric factors (diet, physical_activity), use flag overlap.
        """
        if factor == "smoking":
            return self._numeric_close(gt.get("value"), pred.get("ppd"))
        elif factor == "alcohol":
            return self._numeric_close(gt.get("value"), pred.get("drinks_per_day"))
        elif factor == "bmi":
            return self._numeric_close(gt.get("value"), pred.get("value"))
        elif factor == "physical_activity":
            gt_level = (gt.get("level") or "").lower()
            pred_level = (pred.get("level") or "").lower()
            return gt_level == pred_level
        elif factor == "sleep":
            return self._numeric_close(gt.get("value"), pred.get("hours_per_night"))
        elif factor == "diet":
            gt_q = (gt.get("quality") or "").lower()
            pred_q = (pred.get("quality") or "").lower()
            return gt_q == pred_q
        elif factor == "drug_use":
            # Compare substance detection
            gt_subs = set(gt.get("substances", []))
            pred_subs = set(pred.get("substances", []))
            if not gt_subs and not pred_subs:
                return True
            if not gt_subs or not pred_subs:
                return False
            return len(gt_subs & pred_subs) > 0  # at least one substance overlap
        return True

    def _numeric_close(self, gt_val, pred_val, tol: float = None) -> bool:
        """Check if two numeric values are within tolerance."""
        tol = tol or self.NUMERIC_TOLERANCE
        if gt_val is None:
            return True  # Not annotated
        if pred_val is None:
            return False
        try:
            gt_f = float(gt_val)
            pred_f = float(pred_val)
            if gt_f == 0:
                return pred_f == 0
            return abs(gt_f - pred_f) / gt_f <= tol
        except (TypeError, ValueError):
            return False


# ─────────────────────────────────────────────
# Error Analysis
# ─────────────────────────────────────────────

def error_analysis(annotated_notes: List[dict], pipeline: Pipeline):
    """
    Print detailed error analysis: where the pipeline fails and why.
    Useful for identifying which patterns to add to the rule extractor.
    """
    print("\n" + "=" * 65)
    print("ERROR ANALYSIS")
    print("=" * 65)

    for note in annotated_notes:
        note_id = note["note_id"]
        gt = note.get("ground_truth", {})
        output = pipeline.run_note(note_id=note_id, text=note["text"])
        profile = output.normalized_profile.to_dict()

        errors = []
        for factor in ["smoking", "alcohol", "bmi", "sleep"]:
            gt_status = gt.get(factor, {}).get("status")
            pred_status = profile.get(factor, {}).get("status")
            if gt_status and pred_status and gt_status != pred_status:
                errors.append(f"  {factor}: GT={gt_status} PRED={pred_status}")

        if errors:
            print(f"\nNote: {note_id}")
            print(f"Text: {note['text'][:100]}...")
            for e in errors:
                print(e)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pipeline = Pipeline(use_transformer=False)
    evaluator = Evaluator(pipeline=pipeline)

    print("Running evaluation on sample annotated notes...")
    report = evaluator.evaluate_file()
    report.print()

    # Also run error analysis
    with open(DATA_PATH) as f:
        notes = json.load(f)
    annotated = [n for n in notes if "ground_truth" in n]
    error_analysis(annotated, pipeline)
