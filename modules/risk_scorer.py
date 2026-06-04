"""
Module 5: Risk Scorer
=====================
Quantitative risk assessment engine — converts normalized lifestyle factors
into individual risk scores and a composite health risk profile.

Based on established epidemiological literature:
  - Smoking: Major risk factor (all-cause mortality, cancer, CVD)
  - Alcohol: Dose-dependent risk (cancer, liver disease)
  - BMI: Non-linear risk (underweight, overweight, obesity)
  - Physical Activity: Protective factor (inverse relationship)
  - Sleep: U-shaped risk (too little OR too much is harmful)
  - Diet: Protective when good, harmful when poor
  - Drug Use: Major risk factor (overdose, disease, all-cause mortality)

Risk Tiers:
  LOW       (0–20):   Excellent/good health profile
  MODERATE  (21–40):  Some modifiable risk factors
  HIGH      (41–70):  Significant risk, intervention recommended
  CRITICAL  (71–100): Critical risk, urgent intervention needed

Usage:
    from modules.risk_scorer import RiskScorer, RiskProfile
    from modules.normalizer import NormalizedPatientProfile

    scorer = RiskScorer(weights={...})
    risk_profile = scorer.score(normalized_profile)
    print(risk_profile.summary())
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from modules.normalizer import NormalizedPatientProfile

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Risk Score Details per Factor
# ─────────────────────────────────────────────

@dataclass
class FactorRiskScore:
    """Risk assessment for a single lifestyle factor."""
    factor: str  # smoking, alcohol, bmi, physical_activity, sleep, diet, drug_use
    score: float  # 0–100
    tier: str  # LOW | MODERATE | HIGH | CRITICAL
    explanation: str  # Human-readable summary
    contributing_factors: List[str] = field(default_factory=list)  # What drove the score
    recommendations: List[str] = field(default_factory=list)  # Action items


# ─────────────────────────────────────────────
# Risk Profile Output
# ─────────────────────────────────────────────

@dataclass
class RiskProfile:
    """Complete risk assessment for a patient note."""
    note_id: str
    composite_score: float = 0.0  # Weighted average of all factors, 0–100
    composite_tier: str = "UNKNOWN"  # LOW | MODERATE | HIGH | CRITICAL
    individual_scores: Dict[str, FactorRiskScore] = field(default_factory=dict)
    top_risk_factors: List[tuple] = field(default_factory=list)  # (factor, score) sorted
    protective_factors: List[tuple] = field(default_factory=list)  # (factor, score) sorted
    summary_statement: str = ""
    clinical_flags: List[str] = field(default_factory=list)  # Red flags for clinician
    estimated_mortality_risk: Optional[str] = None  # LOW|MODERATE|HIGH relative to population
    disclaimer: str = "RESEARCH USE ONLY"

    @property
    def factors(self) -> List[FactorRiskScore]:
        """Return list of FactorRiskScore objects (for app.py compatibility)."""
        return list(self.individual_scores.values())

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "note_id": self.note_id,
            "composite_score": round(self.composite_score, 1),
            "composite_tier": self.composite_tier,
            "individual_scores": {
                k: {
                    "score": round(v.score, 1),
                    "tier": v.tier,
                    "explanation": v.explanation,
                    "contributing_factors": v.contributing_factors,
                    "recommendations": v.recommendations,
                }
                for k, v in self.individual_scores.items()
            },
            "top_risk_factors": self.top_risk_factors,
            "protective_factors": self.protective_factors,
            "summary_statement": self.summary_statement,
            "clinical_flags": self.clinical_flags,
            "estimated_mortality_risk": self.estimated_mortality_risk,
            "disclaimer": self.disclaimer,
        }

    def summary(self) -> str:
        """Human-readable risk summary for display."""
        lines = [
            "=" * 70,
            f"LIFESTYLE RISK ASSESSMENT — {self.note_id}",
            "=" * 70,
            f"\nCOMPOSITE RISK SCORE: {self.composite_score:.1f}/100 [{self.composite_tier}]",
            f"Summary: {self.summary_statement}",
            f"\nESTIMATED MORTALITY RISK: {self.estimated_mortality_risk}",
        ]

        if self.top_risk_factors:
            lines.append("\n--- TOP RISK FACTORS ---")
            for factor, score in self.top_risk_factors[:3]:
                tier = self.individual_scores[factor].tier
                lines.append(f"  • {factor.upper()}: {score:.1f}/100 [{tier}]")

        if self.protective_factors:
            lines.append("\n--- PROTECTIVE FACTORS ---")
            for factor, score in self.protective_factors[:2]:
                lines.append(f"  ✓ {factor.upper()}: {score:.1f}/100 [LOW RISK]")

        if self.clinical_flags:
            lines.append("\n--- ⚠️  CLINICAL FLAGS ---")
            for flag in self.clinical_flags:
                lines.append(f"  ⚠️  {flag}")

        lines.append("\n--- DETAILED FACTOR ASSESSMENTS ---")
        for factor in sorted(self.individual_scores.keys()):
            score_obj = self.individual_scores[factor]
            lines.append(f"\n{factor.upper()} — {score_obj.tier}")
            lines.append(f"  Score: {score_obj.score:.1f}/100")
            lines.append(f"  {score_obj.explanation}")
            if score_obj.recommendations:
                lines.append("  Recommendations:")
                for rec in score_obj.recommendations:
                    lines.append(f"    → {rec}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Risk Scorer
# ─────────────────────────────────────────────

class RiskScorer:
    """
    Quantitative lifestyle risk assessment.

    Scoring logic based on epidemiological evidence:
      - Each factor scored 0–100 (0=lowest risk, 100=highest risk)
      - Composite = weighted average across all factors
      - Weights (default) reflect public health impact:
        * Smoking: 30% (leading modifiable mortality risk)
        * Alcohol: 15% (dose-dependent, substantial risk)
        * Drug Use: 20% (high-impact mortality risk)
        * BMI: 15% (complex, non-linear relationship)
        * Sleep: 10% (U-shaped risk)
        * Diet: 5% (preventive)
        * Physical Activity: 5% (protective)

    Args:
        weights: Dict mapping factor → weight (must sum to 1.0)
    """

    # Default epidemiological weights
    DEFAULT_WEIGHTS = {
        "smoking": 0.30,
        "drug_use": 0.20,
        "alcohol": 0.15,
        "bmi": 0.15,
        "sleep": 0.10,
        "diet": 0.05,
        "physical_activity": 0.05,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        # Validate weights sum to ~1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total:.2f}, not 1.0 — normalizing")
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(f"RiskScorer initialized | weights: {self.weights}")

    # ── Public API ──────────────────────────────

    def score(self, profile: NormalizedPatientProfile) -> RiskProfile:
        """
        Score all lifestyle factors and generate risk profile.

        Args:
            profile: NormalizedPatientProfile from normalizer

        Returns:
            RiskProfile with composite and individual factor scores
        """
        risk = RiskProfile(note_id=profile.note_id)

        # Score each factor
        risk.individual_scores["smoking"] = self._score_smoking(profile.smoking)
        risk.individual_scores["alcohol"] = self._score_alcohol(profile.alcohol)
        risk.individual_scores["bmi"] = self._score_bmi(profile.bmi)
        risk.individual_scores["physical_activity"] = self._score_activity(profile.physical_activity)
        risk.individual_scores["sleep"] = self._score_sleep(profile.sleep)
        risk.individual_scores["diet"] = self._score_diet(profile.diet)
        risk.individual_scores["drug_use"] = self._score_drug_use(profile.drug_use)

        # Compute composite score
        composite = sum(
            self.individual_scores["smoking"].score * self.weights["smoking"],
            # ...weighted sum
        ) if False else self._compute_composite(risk.individual_scores)
        
        risk.composite_score = composite
        risk.composite_tier = self._score_to_tier(composite)

        # Identify top and protective factors
        scores_list = [(f, s.score) for f, s in risk.individual_scores.items()]
        risk.top_risk_factors = sorted(scores_list, key=lambda x: x[1], reverse=True)[:3]
        risk.protective_factors = sorted(scores_list, key=lambda x: x[1])[:2]

        # Generate summary
        risk.summary_statement = self._generate_summary(risk)

        # Identify clinical flags
        risk.clinical_flags = self._identify_flags(profile, risk)

        # Estimate mortality risk
        risk.estimated_mortality_risk = self._estimate_mortality_tier(risk.composite_score)

        logger.info(
            f"[{profile.note_id}] Risk scored: composite={composite:.1f} "
            f"({risk.composite_tier}) | top factors: "
            f"{', '.join([f'{f}={s:.0f}' for f, s in risk.top_risk_factors])}"
        )

        return risk

    # ── Factor Scoring Methods ──────────────────

    def _score_smoking(self, record) -> FactorRiskScore:
        """
        Smoking risk scoring.

        Status priority:
          - never: 0 (no risk from smoking)
          - former: 5–15 (residual risk decreases over time)
          - current: 40–100 (dose-dependent)
          - unknown: 20 (uncertainty penalty)

        Dose modifiers:
          - PPD < 0.5: +10
          - 0.5–1 PPD: +20
          - 1–2 PPD: +35
          - 2+ PPD: +50
          - 20+ pack-years: +15 (cumulative damage)
        """
        score = FactorRiskScore(
            factor="smoking",
            score=0.0,
            tier="LOW",
            explanation="",
            contributing_factors=[],
            recommendations=[],
        )

        status = record.status.lower() if record.status else "unknown"

        if status == "never":
            score.score = 0
            score.explanation = "Patient reports never smoking — no tobacco risk."
            score.recommendations = ["Continue smoking avoidance. Counsel on secondhand smoke exposure."]

        elif status == "former":
            # Former smoker: 5–15 depending on recency (estimated)
            score.score = 10
            score.explanation = "Former smoker. Residual cardiovascular and cancer risk remains but declines over time."
            score.contributing_factors = ["Former smoker status"]
            score.recommendations = ["Monitor for smoking relapse. Counsel on cessation maintenance."]

        elif status == "current":
            # Base: 40 for current status
            score.score = 40
            score.contributing_factors = ["Current smoking status"]

            # Add dose escalation
            if record.ppd is not None:
                ppd = record.ppd
                if ppd < 0.5:
                    score.score += 10
                    score.contributing_factors.append(f"Light smoking ({ppd} PPD)")
                elif ppd < 1:
                    score.score += 20
                    score.contributing_factors.append(f"Low-moderate smoking ({ppd} PPD)")
                elif ppd < 2:
                    score.score += 35
                    score.contributing_factors.append(f"Moderate smoking ({ppd} PPD)")
                else:
                    score.score += 50
                    score.contributing_factors.append(f"Heavy smoking ({ppd}+ PPD)")

            # Add cumulative damage penalty
            if record.pack_years is not None and record.pack_years >= 20:
                score.score += 15
                score.contributing_factors.append(f"Cumulative exposure: {record.pack_years} pack-years")

            score.score = min(score.score, 100)  # Cap at 100
            score.explanation = (
                f"Active smoker ({record.ppd or '?'} PPD). "
                f"Smoking is the leading modifiable risk factor for cancer, cardiovascular disease, and COPD. "
                f"High priority for cessation intervention."
            )
            score.recommendations = [
                "Provide smoking cessation counseling (5 A's: Ask, Advise, Assess, Assist, Arrange)",
                "Consider pharmacotherapy (NRT, varenicline, bupropion)",
                "Refer to tobacco quitline (1-800-QUIT-NOW in US)",
                "Set quit date and monitor closely",
            ]

        else:  # unknown
            score.score = 20
            score.explanation = "Smoking status unknown. Cannot assess tobacco-related risk."
            score.contributing_factors = ["Unknown status"]
            score.recommendations = ["Clarify smoking history at next visit"]

        score.tier = self._score_to_tier(score.score)
        return score

    def _score_alcohol(self, record) -> FactorRiskScore:
        """
        Alcohol risk scoring.

        Status:
          - never/abstainer: 0
          - rare: 5 (minimal risk)
          - moderate: 15–25 (safe limits: ≤1 drink/day for women, ≤2 for men)
          - social: 10–15
          - heavy/binge: 50–80+ (dose-dependent cancer, liver, accident risk)
          - unknown: 15 (uncertainty)

        Dose modifiers:
          - ≤1 drink/day (women), ≤2 (men): Low risk
          - 3–4 drinks/day: Moderate risk
          - 5+ drinks/day: High risk
          - Binge pattern: +20
        """
        score = FactorRiskScore(
            factor="alcohol",
            score=0.0,
            tier="LOW",
            explanation="",
            contributing_factors=[],
            recommendations=[],
        )

        status = record.status.lower() if record.status else "unknown"

        if status in ("never", "abstainer"):
            score.score = 0
            score.explanation = "Patient denies alcohol use — no alcohol-related risk."
            score.recommendations = ["Continue alcohol avoidance. Counsel on alcohol risks."]

        elif status == "rare":
            score.score = 5
            score.explanation = "Rare/minimal alcohol use — very low risk."
            score.contributing_factors = ["Minimal consumption"]
            score.recommendations = ["Continue moderate consumption or abstinence."]

        elif status == "social":
            score.score = 10
            score.explanation = "Social drinking. Low risk if within safe limits (≤1 drink/day for women, ≤2 for men)."
            score.contributing_factors = ["Social drinking pattern"]
            score.recommendations = ["Maintain moderate intake. Screen for escalation at future visits."]

        elif status == "moderate":
            score.score = 20
            score.explanation = "Moderate alcohol consumption. Within guideline limits but some cardiovascular/cancer risk."
            score.contributing_factors = ["Moderate consumption"]
            score.recommendations = ["Monitor intake and screen for problematic drinking patterns."]

        elif status in ("heavy", "heavy_use", "very_heavy"):
            score.score = 60
            score.contributing_factors = ["Heavy alcohol use"]

            # Dose escalation
            if record.drinks_per_day is not None:
                dpd = record.drinks_per_day
                if dpd >= 5:
                    score.score += 20
                    score.contributing_factors.append(f"Very high daily intake ({dpd} drinks/day)")
                elif dpd >= 3:
                    score.score += 10
                    score.contributing_factors.append(f"High daily intake ({dpd} drinks/day)")
                elif dpd >= 1:
                    score.score += 5
                    score.contributing_factors.append(f"Elevated daily intake ({dpd} drinks/day)")

            if record.pattern == "binge":
                score.score += 15
                score.contributing_factors.append("Binge drinking pattern")

            score.score = min(score.score, 100)
            score.explanation = (
                f"Heavy alcohol use ({record.drinks_per_day or record.drinks_per_week} drinks). "
                f"Significant risk for liver disease, cancer, accidents, and mortality. "
                f"Urgent intervention needed."
            )
            score.recommendations = [
                "Screen for alcohol use disorder (AUDIT, CAGE)",
                "Refer to addiction specialist or AA/SMART Recovery",
                "Consider pharmacotherapy (naltrexone, acamprosate)",
                "Assess for liver disease (AST, ALT, INR)",
                "Monitor closely for withdrawal risk",
            ]

        else:  # unknown
            score.score = 15
            score.explanation = "Alcohol use status unknown."
            score.contributing_factors = ["Unknown status"]
            score.recommendations = ["Clarify alcohol use at next visit using validated screening (AUDIT)"]

        score.tier = self._score_to_tier(score.score)
        return score

    def _score_bmi(self, record) -> FactorRiskScore:
        """
        BMI risk scoring (U-shaped curve).

        Classes (WHO):
          - Underweight (BMI <18.5): Risk 20–30 (malnutrition, weak immunity, fractures)
          - Normal (18.5–25): Risk 0–5 (optimal)
          - Overweight (25–30): Risk 10–20 (modest disease risk)
          - Obese I (30–35): Risk 30–45 (significant CVD, diabetes risk)
          - Obese II (35–40): Risk 50–70 (high disease risk)
          - Obese III (≥40): Risk 70–100 (very high mortality risk)

        Missing BMI: +15 penalty (unknown risk)
        """
        score = FactorRiskScore(
            factor="bmi",
            score=0.0,
            tier="LOW",
            explanation="",
            contributing_factors=[],
            recommendations=[],
        )

        if record.value is None:
            score.score = 15
            score.explanation = "BMI not documented. Cannot assess weight-related risk."
            score.contributing_factors = ["Missing BMI data"]
            score.recommendations = [
                "Calculate BMI at next visit: weight (lbs) / height² (inches) × 703",
                "If weight/height given, calculate now",
            ]
            score.tier = "MODERATE"
            return score

        bmi = record.value
        bmi_class = record.bmi_class or self._classify_bmi(bmi)

        if bmi_class == "underweight":
            score.score = 25
            score.contributing_factors = [f"Underweight (BMI {bmi:.1f})"]
            score.explanation = (
                f"Underweight (BMI {bmi:.1f}). Risk from malnutrition, weak immunity, "
                f"osteoporosis, and increased infection risk."
            )
            score.recommendations = [
                "Evaluate cause (malabsorption, malignancy, inadequate intake)",
                "Nutritional support and dietician referral",
                "Monitor bone density (DEXA) if age >50",
            ]

        elif bmi_class == "normal":
            score.score = 5
            score.contributing_factors = [f"Normal weight (BMI {bmi:.1f})"]
            score.explanation = f"Normal BMI ({bmi:.1f}). Low weight-related disease risk."
            score.recommendations = ["Maintain current weight through balanced diet and exercise."]

        elif bmi_class == "overweight":
            score.score = 15
            score.contributing_factors = [f"Overweight (BMI {bmi:.1f})"]
            score.explanation = (
                f"Overweight (BMI {bmi:.1f}). Modest increased risk for Type 2 diabetes and CVD. "
                f"Weight loss of 5–10% improves health outcomes."
            )
            score.recommendations = [
                "5–10% weight loss through diet + exercise recommended",
                "Increase physical activity to ≥150 min/week moderate intensity",
                "Dietary counseling (Mediterranean or DASH diet)",
            ]

        elif bmi_class == "obese_I":
            score.score = 35
            score.contributing_factors = [f"Obese Class I (BMI {bmi:.1f})"]
            score.explanation = (
                f"Obesity Class I (BMI {bmi:.1f}). Significant increased risk for Type 2 diabetes, "
                f"hypertension, CVD, sleep apnea, and certain cancers."
            )
            score.recommendations = [
                "Aggressive weight loss: target 5–10% reduction",
                "Structured diet program (calorie deficit 500–750 kcal/day)",
                "Exercise ≥150 min/week moderate + strength training",
                "Consider GLP-1 agonists or weight loss medications if BMI + comorbidities",
                "Screen for sleep apnea (STOP-BANG), hypertension, diabetes",
            ]

        elif bmi_class == "obese_II":
            score.score = 55
            score.contributing_factors = [f"Obese Class II (BMI {bmi:.1f})"]
            score.explanation = (
                f"Obesity Class II (BMI {bmi:.1f}). High risk for obesity-related comorbidities. "
                f"Intervention essential to prevent progression."
            )
            score.recommendations = [
                "Urgent weight loss intervention: target 10–15% reduction",
                "Comprehensive diabetes/CVD screening",
                "Consider bariatric surgery if BMI >35 with comorbidities or >40 regardless",
                "Pharmacotherapy (GLP-1, orlistat, phentermine)",
                "Intensive behavioral therapy and structured program enrollment",
            ]

        elif bmi_class == "obese_III":
            score.score = 80
            score.contributing_factors = [f"Severe Obesity (BMI {bmi:.1f})"]
            score.explanation = (
                f"Severe obesity (BMI {bmi:.1f}). Very high risk for premature mortality "
                f"from diabetes, CVD, sleep apnea, cancers. Urgent comprehensive intervention."
            )
            score.recommendations = [
                "URGENT: Refer for bariatric surgery evaluation",
                "Comprehensive metabolic/comorbidity screening (diabetes, HTN, sleep apnea, GERD)",
                "Multidisciplinary weight management program",
                "Consider intensive GLP-1 therapy, other pharmacotherapy",
                "Monitor for sleep apnea, depression, substance use (risk factors)",
                "Orthopaedic, hepatology, cardiology assessment as needed",
            ]

        score.tier = self._score_to_tier(score.score)
        return score

    def _score_activity(self, record) -> FactorRiskScore:
        """
        Physical activity risk scoring (INVERSE — low activity = high risk).

        Guidelines (WHO):
          - Adults: ≥150 min/week moderate OR ≥75 min/week vigorous
          - Sedentary: 0 min/week

        Scoring (inverse):
          - Sedentary (0 activity): 60–80 (high risk)
          - Low (1–2 days/week): 40–50
          - Moderate (3–4 days/week): 15–25 (meets guidelines)
          - High (5+ days/week): 5–10 (protective)
        """
        score = FactorRiskScore(
            factor="physical_activity",
            score=0.0,
            tier="LOW",
            explanation="",
            contributing_factors=[],
            recommendations=[],
        )

        level = record.level or "unknown"
        days = record.days_per_week

        if level == "sedentary" or days == 0:
            score.score = 70
            score.contributing_factors = ["Sedentary lifestyle (0 activity)"]
            score.explanation = (
                "Sedentary lifestyle. Physical inactivity increases risk for cardiovascular disease, "
                "Type 2 diabetes, cancer, obesity, depression, and all-cause mortality by 20–30%."
            )
            score.recommendations = [
                "Start with 10–15 min walks daily, gradually increase to ≥150 min/week moderate",
                "Reduce sitting time (stand/move every 30 min)",
                "Add strength training 2 days/week",
                "Consider supervised exercise program or physical therapy",
                "Assess barriers (pain, fear, depression) and address",
            ]

        elif level == "low" or (days and days <= 2):
            score.score = 45
            score.contributing_factors = [f"Low activity ({days or '~1'} days/week)"]
            score.explanation = (
                f"Low activity level ({days or '~1'} days/week). Below WHO guidelines. "
                f"Increased risk for chronic disease."
            )
            score.recommendations = [
                "Increase to ≥150 min/week moderate activity (e.g., brisk walking)",
                "Spread activity across ≥3 days/week",
                "Add strength/flexibility training 1–2 days/week",
                "Gradually build endurance; set realistic milestones",
            ]

        elif level == "moderate" or (days and 3 <= days <= 4):
            score.score = 20
            score.contributing_factors = [f"Moderate activity ({days or '3–4'} days/week)"]
            score.explanation = (
                f"Moderate activity level ({days or '3–4'} days/week). Meets WHO guidelines. "
                f"Good cardiovascular protection."
            )
            score.recommendations = [
                "Maintain current activity level",
                "Consider increasing to vigorous intensity for additional benefits",
                "Ensure variety: cardio, strength, flexibility",
            ]

        elif level == "high" or (days and days >= 5):
            score.score = 8
            score.contributing_factors = [f"High activity ({days or '5+'} days/week)"]
            score.explanation = (
                f"High activity level ({days or '5+'} days/week). Excellent cardiovascular fitness "
                f"and protective effect against chronic disease."
            )
            score.recommendations = [
                "Maintain high activity as part of lifestyle",
                "Vary intensity and types to prevent overuse injury",
                "Continue strength and flexibility training",
            ]

        else:  # unknown
            score.score = 30
            score.explanation = "Physical activity level unknown."
            score.contributing_factors = ["Unknown activity"]
            score.recommendations = [
                "Assess current exercise habits using validated questionnaire (IPAQ)",
                "Discuss barriers and goals at next visit",
            ]

        score.tier = self._score_to_tier(score.score)
        return score

    def _score_sleep(self, record) -> FactorRiskScore:
        """
        Sleep risk scoring (U-shaped: too little OR too much is harmful).

        Optimal: 7–9 hours/night for adults
          - <5 hours: Risk 50–70 (chronic sleep deprivation)
          - 5–6 hours: Risk 25–35 (suboptimal)
          - 7–9 hours: Risk 0–10 (optimal)
          - 9–10 hours: Risk 15–25 (oversleeping may indicate depression/illness)
          - >10 hours: Risk 40–60 (associated with mortality)

        OSA: +20–30 if suspected/confirmed (untreated → high mortality)
        Insomnia: +15–20
        Snoring: +10 (suggests possible OSA)
        CPAP use: -10 (mitigates OSA risk)
        """
        score = FactorRiskScore(
            factor="sleep",
            score=0.0,
            tier="LOW",
            explanation="",
            contributing_factors=[],
            recommendations=[],
        )

        hours = record.hours_per_night
        condition = record.condition or "none"
        osa_status = record.osa_status
        cpap = record.on_cpap

        # Base score from sleep duration
        if hours is None:
            base_score = 20
            score.contributing_factors.append("Sleep duration not documented")
        elif hours < 5:
            base_score = 60
            score.contributing_factors.append(f"Severe sleep deprivation ({hours} hours/night)")
        elif hours < 6:
            base_score = 30
            score.contributing_factors.append(f"Suboptimal sleep ({hours} hours/night)")
        elif hours <= 9:
            base_score = 5
            score.contributing_factors.append(f"Optimal sleep duration ({hours} hours/night)")
        elif hours < 10:
            base_score = 20
            score.contributing_factors.append(f"Prolonged sleep ({hours} hours/night)")
        else:  # >10
            base_score = 50
            score.contributing_factors.append(f"Excessive sleep ({hours} hours/night)")

        score.score = base_score

        # Condition modifiers
        if condition == "OSA" or osa_status == "confirmed":
            if cpap:
                score.score += 15
                score.contributing_factors.append("Sleep apnea (confirmed, treated with CPAP)")
            else:
                score.score += 30
                score.contributing_factors.append("Sleep apnea (confirmed, UNTREATED — high risk)")

        elif osa_status == "suspected":
            if cpap:
                score.score += 10
                score.contributing_factors.append("Sleep apnea (suspected, on CPAP)")
            else:
                score.score += 20
                score.contributing_factors.append("Sleep apnea (suspected, needs evaluation)")

        elif condition == "insomnia":
            score.score += 15
            score.contributing_factors.append("Insomnia")

        if record.snoring and osa_status is None:
            score.score += 10
            score.contributing_factors.append("Snoring (may indicate occult sleep apnea)")

        score.score = min(score.score, 100)

        # Generate explanation
        if score.score >= 70:
            score.explanation = (
                f"High sleep risk ({score.score:.0f}/100). "
                f"{'Sleep apnea significantly increases risk for hypertension, MI, stroke, sudden death.' if 'apnea' in str(score.contributing_factors).lower() else 'Sleep disorder or poor sleep hygiene substantially impacts health.'} "
                f"Urgent evaluation and treatment needed."
            )
        elif score.score >= 40:
            score.explanation = (
                f"Moderate sleep concern ({score.score:.0f}/100). "
                f"{'Sleep apnea or suboptimal sleep duration associated with cardiovascular and metabolic risk.' if hours and hours < 6 else 'Sleep disorder or excessive/insufficient sleep impacts health.'} "
                f"Evaluation and intervention recommended."
            )
        else:
            score.explanation = f"Sleep appears adequate ({score.score:.0f}/100 risk). Monitor and maintain sleep hygiene."

        # Recommendations
        score.recommendations = []
        if osa_status == "confirmed" and not cpap:
            score.recommendations.append("URGENT: Initiate CPAP or alternative OSA therapy (oral appliance, positional, surgery)")
        elif osa_status == "suspected":
            score.recommendations.append("Refer for sleep study (polysomnography) to rule out sleep apnea")

        if hours and hours < 7:
            score.recommendations.append("Increase sleep duration to 7–9 hours/night through sleep hygiene")
        if hours and hours > 10:
            score.recommendations.append("Evaluate for depression, hypothyroidism, sleep apnea causing excessive sleep")
        if condition == "insomnia":
            score.recommendations.append("Consider CBT-I (cognitive-behavioral therapy for insomnia) before medication")
            score.recommendations.append("Avoid long-term benzodiazepines; consider melatonin or antidepressants if needed")

        score.recommendations.extend([
            "Maintain consistent sleep schedule (same bedtime/wake time)",
            "Avoid screens 1 hour before bed; limit caffeine after 2 PM",
        ])

        score.tier = self._score_to_tier(score.score)
        return score

    def _score_diet(self, record) -> FactorRiskScore:
        """
        Diet quality risk scoring.

        Quality levels:
          - Good (Mediterranean, DASH, MIND diet): 5–10 (protective)
          - Moderate: 15–25 (mixed foods, some processed)
          - Poor (high sodium, saturated fat, processed): 40–60 (harmful)
          - Unknown: 20

        Flags (increase risk if present):
          - "high_sodium": +10
          - "high_sugar": +10
          - "high_saturated_fat": +10
          - "frequent_fast_food": +15
          - "low_fiber": +10
        """
        score = FactorRiskScore(
            factor="diet",
            score=0.0,
            tier="LOW",
            explanation="",
            contributing_factors=[],
            recommendations=[],
        )

        quality = record.quality or "unknown"

        if quality == "good":
            score.score = 8
            score.contributing_factors = ["Healthy diet pattern"]
            score.explanation = (
                "Healthy diet (Mediterranean/DASH pattern). Protective against cardiovascular disease, "
                "diabetes, cancer, and obesity."
            )
            score.recommendations = ["Continue current healthy eating pattern."]

        elif quality == "moderate":
            score.score = 20
            score.contributing_factors = ["Moderate diet quality"]
            score.explanation = (
                "Moderate diet quality. Mixed foods with some healthful and some less healthy choices. "
                "Opportunity for improvement in disease prevention."
            )
            score.recommendations = [
                "Increase vegetables, fruits, whole grains, lean protein",
                "Reduce processed foods, added sugars, saturated fat",
                "Consult dietician for personalized nutrition plan",
            ]

        elif quality == "poor":
            score.score = 50
            score.contributing_factors = ["Poor diet quality"]
            score.explanation = (
                "Poor diet quality (high sodium, saturated fat, processed foods). "
                "Substantially increases risk for hypertension, dyslipidemia, diabetes, CVD, and cancer."
            )
            score.recommendations = [
                "Urgent dietary intervention: transition to Mediterranean or DASH diet",
                "Refer to registered dietician (Medicare covers with diagnosis codes)",
                "Reduce sodium intake to <2300 mg/day (target <1500 for HTN)",
                "Eliminate sugary beverages; drink water",
                "Prepare meals at home; limit restaurant/fast food to <1x/week",
                "Increase fiber (whole grains, legumes, vegetables) gradually",
            ]

        else:  # unknown
            score.score = 20
            score.explanation = "Diet quality not documented."
            score.contributing_factors = ["Unknown diet"]
            score.recommendations = [
                "Perform dietary assessment at next visit (24-hour recall or FFQ)",
                "Screen for food insecurity if relevant",
            ]

        # Apply flag penalties
        harmful_flags = [f for f in record.flags if any(
            bad in f for bad in ["high_sodium", "high_sugar", "high_saturated_fat", "frequent_fast_food", "low_fiber"]
        )]
        for flag in harmful_flags:
            if "high_sodium" in flag:
                score.score += 10
                score.contributing_factors.append("High sodium intake")
            elif "high_sugar" in flag:
                score.score += 10
                score.contributing_factors.append("High sugar consumption")
            elif "high_saturated_fat" in flag:
                score.score += 10
                score.contributing_factors.append("High saturated fat")
            elif "frequent_fast_food" in flag:
                score.score += 15
                score.contributing_factors.append("Frequent fast food consumption")
            elif "low_fiber" in flag:
                score.score += 10
                score.contributing_factors.append("Low dietary fiber")

        score.score = min(score.score, 100)
        score.tier = self._score_to_tier(score.score)
        return score

    def _score_drug_use(self, record) -> FactorRiskScore:
        """
        Drug use risk scoring.

        Status:
          - Never/denies: 0 (no risk)
          - Former: 10–20 (residual risk + relapse potential)
          - Current: 60–100 (high risk for overdose, infection, mortality)
          - Unknown: 25

        Substance modifiers (if documented):
          - Opioids: +30 (overdose risk, especially if non-prescribed)
          - Cocaine/methamphetamine: +25 (cardiac, mortality)
          - Cannabis: +5 (low direct risk, but dependence potential)
          - IVDU (IV route): +20 (infection, endocarditis, HIV)

        Aggregate substances increase risk cumulatively.
        """
        score = FactorRiskScore(
            factor="drug_use",
            score=0.0,
            tier="LOW",
            explanation="",
            contributing_factors=[],
            recommendations=[],
        )

        status = record.status.lower() if record.status else "unknown"

        if status in ("never", "denies"):
            score.score = 0
            score.explanation = "Patient denies illicit drug use — no substance use risk."
            score.recommendations = ["Counsel on drug risks and overdose prevention. Provide naloxone if opioid-exposed."]

        elif status == "former":
            score.score = 15
            score.contributing_factors = ["Former drug use"]
            score.explanation = (
                "Former substance use. Risk of relapse and persistent social/medical sequelae "
                "(hepatitis C, HIV, endocarditis, PTSD)."
            )
            score.recommendations = [
                "Screen for relapse risk and triggers",
                "Refer to addiction counselor if signs of relapse",
                "Screen for bloodborne infections (HIV, Hep C, Hep B)",
                "Assess for medication-assisted therapy maintenance if opioid use disorder",
            ]

        elif status == "current":
            score.score = 70
            score.contributing_factors = ["Current drug use"]

            # Substance-specific escalation
            substances = record.substances or []
            for sub in substances:
                sub_lower = sub.lower()
                if any(x in sub_lower for x in ["opioid", "heroin", "fentanyl", "oxycodone", "hydrocodone"]):
                    score.score += 25
                    score.contributing_factors.append(f"Opioid use ({sub}) — overdose risk")
                elif any(x in sub_lower for x in ["cocaine", "crack", "methamphetamine", "meth"]):
                    score.score += 20
                    score.contributing_factors.append(f"Stimulant use ({sub}) — cardiac risk")
                elif "cannabis" in sub_lower or "marijuana" in sub_lower:
                    score.score += 5
                    score.contributing_factors.append("Cannabis use")

            # IV route escalation
            if record.route == "IV" or "IVDU" in substances:
                score.score += 20
                score.contributing_factors.append("Intravenous route — infection/endocarditis/HIV risk")

            score.score = min(score.score, 100)
            score.explanation = (
                f"Active substance use ({', '.join(substances) if substances else 'unspecified'}). "
                f"Very high risk for overdose, infectious disease (HIV, hepatitis, endocarditis), "
                f"legal consequences, and premature mortality."
            )
            score.recommendations = [
                "Provide naloxone (Narcan) prescription and overdose education",
                "Refer to addiction specialist, OUD treatment (MAT: methadone, buprenorphine, naltrexone)",
                "Screen for co-occurring mental health (depression, PTSD, anxiety)",
                "Refer to needle exchange/syringe services if IVDU",
                "Screen for HIV, Hep B, Hep C, TB, endocarditis (blood cultures)",
                "Consider harm reduction counseling",
                "Establish trust and offer regular monitoring; avoid stigma",
            ]

        else:  # unknown
            score.score = 25
            score.explanation = "Substance use status unknown."
            score.contributing_factors = ["Unknown status"]
            score.recommendations = [
                "Screen for substance use using validated tool (DAST-10, NIDA Quick Screen)",
                "Use non-judgmental, confidential approach",
            ]

        score.tier = self._score_to_tier(score.score)
        return score

    # ── Composite & Helper Methods ──────────────

    def _compute_composite(self, individual_scores: Dict[str, FactorRiskScore]) -> float:
        """Weighted average of all factor scores."""
        total = 0.0
        for factor, weight in self.weights.items():
            if factor in individual_scores:
                total += individual_scores[factor].score * weight
        return round(total, 1)

    def _score_to_tier(self, score: float) -> str:
        """Convert numeric score to tier."""
        if score < 20:
            return "LOW"
        elif score < 40:
            return "MODERATE"
        elif score < 70:
            return "HIGH"
        else:
            return "CRITICAL"

    def _classify_bmi(self, bmi: float) -> str:
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25:
            return "normal"
        elif bmi < 30:
            return "overweight"
        elif bmi < 35:
            return "obese_I"
        elif bmi < 40:
            return "obese_II"
        else:
            return "obese_III"

    def _generate_summary(self, risk: RiskProfile) -> str:
        """Generate 1–2 sentence clinical summary."""
        tier = risk.composite_tier
        if tier == "LOW":
            return (
                "Excellent lifestyle profile with minimal health risk. "
                "Continue current practices and maintain annual screening."
            )
        elif tier == "MODERATE":
            top_factors = ", ".join([f[0].replace("_", " ") for f in risk.top_risk_factors[:2]])
            return (
                f"Moderate overall risk driven primarily by {top_factors}. "
                f"Targeted interventions in these areas could significantly improve health."
            )
        elif tier == "HIGH":
            top_factors = ", ".join([f[0].replace("_", " ") for f in risk.top_risk_factors[:2]])
            return (
                f"Significant health risk from multiple lifestyle factors ({top_factors}). "
                f"Urgent, comprehensive intervention needed to prevent chronic disease."
            )
        else:  # CRITICAL
            return (
                f"Critical health risk requiring immediate, intensive intervention. "
                f"Refer for multidisciplinary care (primary care, cardiology, oncology, addiction, mental health as indicated)."
            )

    def _identify_flags(self, profile: NormalizedPatientProfile, risk: RiskProfile) -> List[str]:
        """Identify clinically important red flags."""
        flags = []

        # Smoking + high pack-years
        if profile.smoking.status == "current" and profile.smoking.ppd and profile.smoking.ppd >= 2:
            flags.append("Heavy current smoker — very high lung cancer and CVD risk")

        # Untreated severe OSA
        if profile.sleep.osa_status == "confirmed" and not profile.sleep.on_cpap:
            flags.append("⚠️ Untreated sleep apnea — high risk for sudden cardiac death, stroke")

        # Severe obesity + diabetes risk
        if profile.bmi.bmi_class in ("obese_II", "obese_III"):
            flags.append(f"Severe obesity (BMI {profile.bmi.value:.1f}) — recommend bariatric surgery evaluation")

        # Active IVDU
        if profile.drug_use.status == "current" and profile.drug_use.route == "IV":
            flags.append("Active IVDU — prescribe naloxone, refer to harm reduction + MAT services")

        # Heavy alcohol + poor nutrition
        if profile.alcohol.status in ("heavy", "heavy_use", "very_heavy"):
            if profile.diet.quality == "poor":
                flags.append("Heavy alcohol + poor diet — screen for liver disease and malnutrition")

        # Multiple modifiable risks
        high_risk_count = sum(1 for s in risk.individual_scores.values() if s.score >= 50)
        if high_risk_count >= 3:
            flags.append(f"Polyaddiction/polydisease pattern ({high_risk_count} high-risk factors) — multidisciplinary care essential")

        return flags

    def _estimate_mortality_tier(self, composite_score: float) -> str:
        """Estimate relative mortality risk tier based on composite score."""
        # Rough calibration:
        # Composite <20: 0.8–1.0x population mortality
        # Composite 20–40: 1.0–1.5x
        # Composite 40–70: 1.5–3.0x
        # Composite 70+: 3.0–5.0x+
        if composite_score < 20:
            return "LOW (near population average)"
        elif composite_score < 40:
            return "MODERATE (1–1.5x population risk)"
        elif composite_score < 70:
            return "HIGH (1.5–3x population risk)"
        else:
            return "CRITICAL (3–5x+ population risk — urgent intervention)"


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

    # Sample clinical note
    text = """Patient is a 52-year-old male smoker.
Smokes 2 packs per day for 30 years (60 pack-years).
Drinks 5-6 beers nightly. Prior detox attempt failed.
BMI 38.2, class II obesity. Weight 285 lbs.
Sedentary, no exercise. Works desk job.
Sleeps 4-5 hours, loud snoring, likely sleep apnea.
Diet poor — high sodium, frequent fast food.
Denies illicit drug use but history of cocaine abuse 10 years ago, now in recovery.
"""

    print("=" * 70)
    print("RISK SCORER TEST")
    print("=" * 70)

    # Run full pipeline
    preprocessor = ClinicalPreprocessor()
    processed = preprocessor.process("TEST_001", text)

    ner = NERExtractor(use_transformer=False)
    ner_result = ner.extract("TEST_001", processed.sentences)

    rel_extractor = RelationExtractor()
    rel_result = rel_extractor.extract("TEST_001", ner_result)

    normalizer = Normalizer()
    rel_result.sentences = processed.sentences
    profile = normalizer.normalize(rel_result)

    scorer = RiskScorer()
    risk_profile = scorer.score(profile)

    print(risk_profile.summary())
