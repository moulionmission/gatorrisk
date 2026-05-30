"""
Module 5: Risk Scorer
=====================
Converts a NormalizedPatientProfile into a quantitative risk assessment.

For each risk factor, computes:
  - A factor-level risk score (0.0 = no risk, 1.0 = maximum risk)
  - A risk tier: LOW | MODERATE | HIGH | CRITICAL
  - A brief clinical rationale

Then aggregates into a composite risk score and overall tier.

NOTE: This is an ML ENGINEERING tool for NLP extraction validation —
not a certified clinical decision support system. All scores should
be reviewed by a licensed clinician before clinical use.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List
from modules.normalizer import NormalizedPatientProfile

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Risk Tiers
# ─────────────────────────────────────────────

TIERS = {
    (0.0, 0.25): "LOW",
    (0.25, 0.50): "MODERATE",
    (0.50, 0.75): "HIGH",
    (0.75, 1.01): "CRITICAL",
}

def score_to_tier(score: float) -> str:
    for (lo, hi), tier in TIERS.items():
        if lo <= score < hi:
            return tier
    return "UNKNOWN"


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class FactorRisk:
    factor: str
    score: float          # 0.0 – 1.0
    tier: str             # LOW | MODERATE | HIGH | CRITICAL
    rationale: str        # one-line clinical explanation
    weight: float         # contribution weight in composite score


@dataclass
class RiskProfile:
    note_id: str
    composite_score: float
    composite_tier: str
    factors: List[FactorRisk] = field(default_factory=list)
    disclaimer: str = (
        "RESEARCH USE ONLY. Not a certified clinical decision support tool. "
        "All risk assessments require clinician review."
    )

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Note: {self.note_id}",
            f"Composite Risk: {self.composite_score:.2f} ({self.composite_tier})",
            "-" * 40,
        ]
        for f in sorted(self.factors, key=lambda x: -x.score):
            lines.append(f"  {f.factor:<22} {f.score:.2f}  {f.tier:<10}  {f.rationale}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Risk Scorer
# ─────────────────────────────────────────────

class RiskScorer:
    """
    Scores a NormalizedPatientProfile across all lifestyle risk factors.

    Weights are configurable (see configs/config.yaml).
    Default weights are loosely aligned with cardiovascular disease
    risk factor literature.
    """

    DEFAULT_WEIGHTS = {
        "smoking": 0.25,
        "alcohol": 0.15,
        "bmi": 0.20,
        "physical_activity": 0.15,
        "sleep": 0.10,
        "diet": 0.10,
        "drug_use": 0.05,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total:.2f}, not 1.0. Normalizing.")
            self.weights = {k: v / total for k, v in self.weights.items()}

    def score(self, profile: NormalizedPatientProfile) -> RiskProfile:
        """
        Compute risk scores for all factors and composite score.

        Args:
            profile: NormalizedPatientProfile from Normalizer

        Returns:
            RiskProfile with per-factor and composite scores
        """
        factors = []

        factors.append(self._score_smoking(profile.smoking))
        factors.append(self._score_alcohol(profile.alcohol))
        factors.append(self._score_bmi(profile.bmi))
        factors.append(self._score_activity(profile.physical_activity))
        factors.append(self._score_sleep(profile.sleep))
        factors.append(self._score_diet(profile.diet))
        factors.append(self._score_drug_use(profile.drug_use))

        composite = sum(f.score * f.weight for f in factors)
        composite = round(min(composite, 1.0), 4)

        return RiskProfile(
            note_id=profile.note_id,
            composite_score=composite,
            composite_tier=score_to_tier(composite),
            factors=factors,
        )

    # ── Factor Scorers ─────────────────────────────

    def _score_smoking(self, s) -> FactorRisk:
        w = self.weights["smoking"]

        if s.status == "never":
            return FactorRisk("smoking", 0.0, "LOW", "Never smoker.", w)
        if s.status == "former":
            score = 0.25
            rationale = "Former smoker — residual risk remains."
            if s.pack_years and s.pack_years >= 20:
                score = 0.40
                rationale = f"Former smoker with {s.pack_years} pack-years. Significant residual risk."
            return FactorRisk("smoking", score, score_to_tier(score), rationale, w)
        if s.status == "current":
            score = 0.60
            rationale = "Current smoker."
            if s.ppd:
                if s.ppd >= 2.0:
                    score = 1.0
                    rationale = f"Heavy smoker: {s.ppd} ppd."
                elif s.ppd >= 1.0:
                    score = 0.80
                    rationale = f"Moderate-heavy smoker: {s.ppd} ppd."
                else:
                    score = 0.65
                    rationale = f"Light smoker: {s.ppd} ppd."
            if s.pack_years and s.pack_years >= 30:
                score = min(score + 0.15, 1.0)
                rationale += f" {s.pack_years} pack-year history."
            return FactorRisk("smoking", round(score, 2), score_to_tier(score), rationale, w)

        return FactorRisk("smoking", 0.0, "LOW", "Smoking status unknown/not mentioned.", w)

    def _score_alcohol(self, a) -> FactorRisk:
        w = self.weights["alcohol"]

        if a.status == "never":
            return FactorRisk("alcohol", 0.0, "LOW", "No alcohol use reported.", w)
        if a.status == "former":
            return FactorRisk("alcohol", 0.15, "LOW", "Former alcohol user — in recovery.", w)
        if a.status == "current":
            dpd = a.drinks_per_day
            dpw = a.drinks_per_week or (dpd * 7 if dpd else None)

            if dpw is None:
                score = 0.20
                rationale = "Alcohol use reported, quantity unknown."
                if a.pattern == "heavy":
                    score = 0.70
                    rationale = "Heavy alcohol use reported."
                elif a.pattern == "social":
                    score = 0.10
                    rationale = "Social/occasional drinker."
            else:
                # NIAAA thresholds: >14 drinks/week (men) or >7 (women) = heavy
                if dpw > 21:
                    score, rationale = 1.0, f"Very heavy drinking: {dpw:.1f} drinks/week."
                elif dpw > 14:
                    score, rationale = 0.80, f"Heavy drinking: {dpw:.1f} drinks/week."
                elif dpw > 7:
                    score, rationale = 0.55, f"Moderate-heavy: {dpw:.1f} drinks/week."
                elif dpw > 3:
                    score, rationale = 0.30, f"Moderate: {dpw:.1f} drinks/week."
                else:
                    score, rationale = 0.10, f"Light: {dpw:.1f} drinks/week."

            return FactorRisk("alcohol", round(score, 2), score_to_tier(score), rationale, w)

        return FactorRisk("alcohol", 0.0, "LOW", "Alcohol status not documented.", w)

    def _score_bmi(self, b) -> FactorRisk:
        w = self.weights["bmi"]

        bmi_class_scores = {
            "underweight": (0.50, "Underweight — nutritional risk."),
            "normal": (0.0, "BMI in normal range."),
            "overweight": (0.30, "Overweight — elevated cardiometabolic risk."),
            "obese_I": (0.55, "Class I obesity."),
            "obese_II": (0.75, "Class II obesity — significant risk."),
            "obese_III": (1.0, "Class III (morbid) obesity — critical risk."),
        }

        if b.value is not None:
            if b.value < 18.5:
                cls = "underweight"
            elif b.value < 25.0:
                cls = "normal"
            elif b.value < 30.0:
                cls = "overweight"
            elif b.value < 35.0:
                cls = "obese_I"
            elif b.value < 40.0:
                cls = "obese_II"
            else:
                cls = "obese_III"
            score, rationale = bmi_class_scores[cls]
            rationale = f"BMI {b.value} — {rationale}"
        elif b.bmi_class and b.bmi_class in bmi_class_scores:
            score, rationale = bmi_class_scores[b.bmi_class]
        else:
            return FactorRisk("bmi", 0.0, "LOW", "BMI not documented.", w)

        return FactorRisk("bmi", round(score, 2), score_to_tier(score), rationale, w)

    def _score_activity(self, a) -> FactorRisk:
        w = self.weights["physical_activity"]

        level_scores = {
            "sedentary": (0.85, "Sedentary lifestyle — major cardiovascular risk factor."),
            "low": (0.55, "Low physical activity."),
            "moderate": (0.20, "Moderate activity — meets partial guidelines."),
            "high": (0.0, "High activity — meets/exceeds recommendations."),
        }

        if a.level and a.level in level_scores:
            score, rationale = level_scores[a.level]
            # Bonus context from minutes/week
            if a.days_per_week and a.minutes_per_session:
                total_min = a.days_per_week * a.minutes_per_session
                if total_min >= 150:
                    score = max(0.0, score - 0.15)
                    rationale += f" ({int(total_min)} min/week meets guidelines.)"
            return FactorRisk("physical_activity", round(score, 2), score_to_tier(score), rationale, w)

        return FactorRisk("physical_activity", 0.0, "LOW", "Physical activity not documented.", w)

    def _score_sleep(self, s) -> FactorRisk:
        w = self.weights["sleep"]

        score = 0.0
        notes = []

        # Hours-based scoring
        if s.hours_per_night is not None:
            if s.hours_per_night < 5:
                score += 0.60
                notes.append(f"Severely insufficient sleep ({s.hours_per_night}h).")
            elif s.hours_per_night < 6:
                score += 0.40
                notes.append(f"Insufficient sleep ({s.hours_per_night}h).")
            elif s.hours_per_night < 7:
                score += 0.20
                notes.append(f"Below recommended sleep ({s.hours_per_night}h).")
            elif s.hours_per_night <= 9:
                notes.append(f"Adequate sleep ({s.hours_per_night}h).")
            else:
                score += 0.15
                notes.append(f"Hypersomnia suspected ({s.hours_per_night}h).")

        # Condition-based scoring
        if s.condition == "OSA":
            osa_add = 0.40 if s.osa_status == "confirmed" else 0.25
            score += osa_add
            notes.append(f"OSA {s.osa_status or 'mentioned'}.")
        elif s.condition == "insomnia":
            score += 0.30
            notes.append("Insomnia reported.")

        if s.on_cpap:
            score = max(0.0, score - 0.15)
            notes.append("On CPAP therapy.")

        score = round(min(score, 1.0), 2)
        rationale = " ".join(notes) if notes else "Sleep not documented."
        return FactorRisk("sleep", score, score_to_tier(score), rationale, w)

    def _score_diet(self, d) -> FactorRisk:
        w = self.weights["diet"]

        quality_scores = {
            "poor": 0.75,
            "moderate": 0.35,
            "good": 0.05,
        }

        score = quality_scores.get(d.quality, 0.0)

        # Adjust for specific flags
        high_risk_flags = {"high_sodium", "high_fat", "high_sugar", "fast_food", "skips_meals"}
        flag_hits = len(set(d.flags) & high_risk_flags)
        score = min(score + flag_hits * 0.05, 1.0)

        rationale = f"Diet quality: {d.quality or 'unknown'}."
        if d.flags:
            rationale += f" Flags: {', '.join(d.flags[:5])}."

        return FactorRisk("diet", round(score, 2), score_to_tier(score), rationale, w)

    def _score_drug_use(self, d) -> FactorRisk:
        w = self.weights["drug_use"]

        if d.status == "never":
            return FactorRisk("drug_use", 0.0, "LOW", "No illicit drug use reported.", w)
        if d.status == "former":
            score = 0.20
            rationale = "Former drug use — in recovery."
            return FactorRisk("drug_use", score, score_to_tier(score), rationale, w)
        if d.status == "current":
            high_risk = {"opioid/heroin", "cocaine", "methamphetamine", "IVDU"}
            if any(s in high_risk for s in d.substances):
                score = 1.0
                rationale = f"High-risk substance use: {', '.join(d.substances)}."
            else:
                score = 0.55
                rationale = f"Illicit drug use: {', '.join(d.substances) or 'unspecified'}."
            if d.route == "IV":
                score = 1.0
                rationale += " IV route — additional infection risk."
            return FactorRisk("drug_use", round(score, 2), score_to_tier(score), rationale, w)

        return FactorRisk("drug_use", 0.0, "LOW", "Drug use status not documented.", w)


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO)

    from modules.preprocessor import ClinicalPreprocessor
    from modules.ner_extractor import NERExtractor
    from modules.relation_extractor import RelationExtractor
    from modules.normalizer import Normalizer

    text = """Patient smokes 1.5 packs per day for the past 30 years.
Drinks 3 beers nightly.
BMI 34.2, consistent with class I obesity.
Sedentary lifestyle with no regular exercise.
Sleeps 4-5 hours per night; loud snoring noted, OSA suspected.
Diet is poor — high sodium, frequent fast food.
Denies illicit drug use."""

    note = ClinicalPreprocessor().process("TEST_001", text)
    ner_result = NERExtractor().extract("TEST_001", note.sentences)
    rel_result = RelationExtractor().extract("TEST_001", ner_result)
    profile = Normalizer().normalize(rel_result)
    risk = RiskScorer().score(profile)

    print("=" * 60)
    print(risk.summary())
    print(f"\n⚠️  {risk.disclaimer}")
