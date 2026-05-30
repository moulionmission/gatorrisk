"""
Module 4: Normalizer
====================
Converts raw Relation objects into standardized, structured JSON records.

Handles:
  - Unit normalization (e.g., cigarettes/day → ppd)
  - Range collapsing (e.g., "4-5 hours" → avg 4.5)
  - Status consolidation (multiple mentions → single canonical status)
  - Missing value inference from context flags
  - Output schema enforcement per risk factor
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from modules.relation_extractor import Relation, RelationResult
from modules.bmi_calculator import BMICalculator

logger = logging.getLogger(__name__)

# Singleton BMI calculator
_bmi_calc = BMICalculator()


# ─────────────────────────────────────────────
# Output Schemas (one per risk factor)
# ─────────────────────────────────────────────

@dataclass
class SmokingRecord:
    status: str = "unknown"          # current | former | never | unknown
    ppd: Optional[float] = None      # packs per day
    pack_years: Optional[float] = None
    cigarettes_per_day: Optional[float] = None
    temporal: Optional[str] = None


@dataclass
class AlcoholRecord:
    status: str = "unknown"
    drinks_per_day: Optional[float] = None
    drinks_per_week: Optional[float] = None
    pattern: Optional[str] = None    # social | heavy | binge | rare
    temporal: Optional[str] = None


@dataclass
class BMIRecord:
    status: str = "current"
    value: Optional[float] = None    # numeric BMI
    unit: str = "kg/m2"
    bmi_class: Optional[str] = None  # underweight|normal|overweight|obese_I|II|III
    weight: Optional[float] = None
    weight_unit: Optional[str] = None


@dataclass
class PhysicalActivityRecord:
    status: str = "current"
    level: Optional[str] = None      # sedentary|low|moderate|high
    days_per_week: Optional[int] = None
    minutes_per_session: Optional[float] = None
    activity_types: List[str] = field(default_factory=list)


@dataclass
class SleepRecord:
    status: str = "current"
    hours_per_night: Optional[float] = None
    condition: Optional[str] = None  # OSA|insomnia|hypersomnia|none
    osa_status: Optional[str] = None # suspected|confirmed
    on_cpap: bool = False
    snoring: bool = False


@dataclass
class DietRecord:
    status: str = "current"
    quality: Optional[str] = None    # poor|moderate|good
    flags: List[str] = field(default_factory=list)


@dataclass
class DrugUseRecord:
    status: str = "unknown"
    substances: List[str] = field(default_factory=list)
    route: Optional[str] = None      # oral|IV|smoked
    temporal: Optional[str] = None


@dataclass
class NormalizedPatientProfile:
    """Final structured output for a single patient note."""
    note_id: str
    smoking: SmokingRecord = field(default_factory=SmokingRecord)
    alcohol: AlcoholRecord = field(default_factory=AlcoholRecord)
    bmi: BMIRecord = field(default_factory=BMIRecord)
    physical_activity: PhysicalActivityRecord = field(default_factory=PhysicalActivityRecord)
    sleep: SleepRecord = field(default_factory=SleepRecord)
    diet: DietRecord = field(default_factory=DietRecord)
    drug_use: DrugUseRecord = field(default_factory=DrugUseRecord)
    extraction_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────
# Normalizer Class
# ─────────────────────────────────────────────

class Normalizer:
    """
    Normalizes a RelationResult into a NormalizedPatientProfile.

    For each risk factor, consolidates multiple Relations (one per
    sentence) into a single canonical record using priority rules.
    """

    def normalize(self, rel_result: RelationResult) -> NormalizedPatientProfile:
        """
        Convert relations into a structured patient profile.

        Args:
            rel_result: Output from RelationExtractor

        Returns:
            NormalizedPatientProfile — ready for risk scoring or API output
        """
        profile = NormalizedPatientProfile(note_id=rel_result.note_id)

        profile.smoking = self._normalize_smoking(rel_result.by_factor("smoking"))
        profile.alcohol = self._normalize_alcohol(rel_result.by_factor("alcohol"))
        profile.bmi = self._normalize_bmi(rel_result.by_factor("bmi"), raw_sentences=rel_result.sentences)
        profile.physical_activity = self._normalize_activity(rel_result.by_factor("physical_activity"))
        profile.sleep = self._normalize_sleep(rel_result.by_factor("sleep"))
        profile.diet = self._normalize_diet(rel_result.by_factor("diet"))
        profile.drug_use = self._normalize_drug_use(rel_result.by_factor("drug_use"))

        logger.info(f"[{rel_result.note_id}] Normalized profile created")
        return profile

    # ── Factor Normalizers ────────────────────────

    def _normalize_smoking(self, relations: List[Relation]) -> SmokingRecord:
        rec = SmokingRecord()
        if not relations:
            return rec

        # Status: "never" > "former" > "current"
        rec.status = self._consolidate_status([r.status for r in relations])

        for r in relations:
            if r.unit == "ppd" and r.value is not None:
                rec.ppd = r.value
            if r.unit == "cigarettes/day" and r.value is not None:
                rec.cigarettes_per_day = r.value
                rec.ppd = rec.ppd or round(r.value / 20, 2)
            if r.temporal:
                rec.temporal = r.temporal
            # Extract pack_years from flags like "pack_years:45"
            for flag in r.flags:
                if flag.startswith("pack_years:"):
                    try:
                        rec.pack_years = float(flag.split(":")[1])
                    except ValueError:
                        pass

        return rec

    def _normalize_alcohol(self, relations: List[Relation]) -> AlcoholRecord:
        rec = AlcoholRecord()
        if not relations:
            return rec

        rec.status = self._consolidate_status([r.status for r in relations])

        for r in relations:
            if r.value is not None:
                if r.unit == "drinks/day":
                    rec.drinks_per_day = r.value
                    rec.drinks_per_week = rec.drinks_per_week or round(r.value * 7, 1)
                elif r.unit == "drinks/week":
                    rec.drinks_per_week = r.value
                    rec.drinks_per_day = rec.drinks_per_day or round(r.value / 7, 2)

            if "social_drinker" in r.flags:
                rec.pattern = "social"
            if "heavy_use" in r.flags:
                rec.pattern = "heavy"
            if r.temporal:
                rec.temporal = r.temporal

        # Infer pattern from quantity if not explicitly stated
        if rec.pattern is None and rec.drinks_per_week is not None:
            if rec.drinks_per_week <= 1:
                rec.pattern = "rare"
            elif rec.drinks_per_week <= 7:
                rec.pattern = "moderate"
            elif rec.drinks_per_week <= 14:
                rec.pattern = "heavy"
            else:
                rec.pattern = "very_heavy"

        return rec

    def _normalize_bmi(self, relations: List[Relation], raw_sentences: List[str] = None) -> BMIRecord:
        rec = BMIRecord()

        # Step 1: Try to get explicit BMI value from NER relations
        for r in relations:
            if r.unit == "kg/m2" and r.value is not None:
                rec.value = r.value
                rec.bmi_class = r.condition or self._classify_bmi(r.value)
            elif r.unit in ("lbs", "kg") and r.value is not None:
                rec.weight = r.value
                rec.weight_unit = r.unit

            # Pick up class from flags
            for flag in r.flags:
                if flag in ("obese_I", "obese_II", "obese_III", "overweight", "underweight", "normal"):
                    rec.bmi_class = rec.bmi_class or flag

        # Step 2: If no explicit BMI found, compute from height+weight in raw text
        if rec.value is None and raw_sentences:
            bmi_extraction = _bmi_calc.extract_from_sentences(raw_sentences)
            if bmi_extraction and bmi_extraction.bmi:
                rec.value = bmi_extraction.bmi
                rec.bmi_class = bmi_extraction.bmi_class
                rec.weight = bmi_extraction.weight_lbs
                rec.weight_unit = "lbs"
                rec.unit = "kg/m2"
                computed = bmi_extraction.computed_from
                logger.debug(f"BMI computed from height/weight: {rec.value} ({rec.bmi_class}) [{computed}]")

        return rec

    def _normalize_activity(self, relations: List[Relation]) -> PhysicalActivityRecord:
        rec = PhysicalActivityRecord()
        if not relations:
            return rec

        for r in relations:
            if r.condition:
                rec.level = r.condition
            if r.value and r.unit == "min/session":
                rec.minutes_per_session = r.value
            for flag in r.flags:
                if flag == "sedentary":
                    rec.level = "sedentary"
                elif flag.startswith("days_per_week:"):
                    try:
                        rec.days_per_week = int(flag.split(":")[1])
                    except ValueError:
                        pass
                elif flag.startswith("type:"):
                    activity = flag.split(":")[1]
                    if activity not in rec.activity_types:
                        rec.activity_types.append(activity)
                elif flag.startswith("miles:"):
                    rec.activity_types.append(f"running_{flag.split(':')[1]}mi")

        # Infer level from days/week if not set
        if rec.level is None and rec.days_per_week is not None:
            if rec.days_per_week == 0:
                rec.level = "sedentary"
            elif rec.days_per_week <= 2:
                rec.level = "low"
            elif rec.days_per_week <= 4:
                rec.level = "moderate"
            else:
                rec.level = "high"

        return rec

    def _normalize_sleep(self, relations: List[Relation]) -> SleepRecord:
        rec = SleepRecord()
        if not relations:
            return rec

        for r in relations:
            # Average range values (e.g., 4-5 hours → 4.5)
            if r.value is not None:
                if r.value2 is not None:
                    rec.hours_per_night = round((r.value + r.value2) / 2, 1)
                else:
                    rec.hours_per_night = r.value

            if r.condition:
                rec.condition = r.condition

            for flag in r.flags:
                if "osa_suspected" in flag or "osa_likely" in flag or "osa_probable" in flag:
                    rec.osa_status = "suspected"
                elif "osa_confirmed" in flag or "osa_diagnosed" in flag:
                    rec.osa_status = "confirmed"
                if "cpap" in flag:
                    rec.on_cpap = True
                if "snoring" in flag:
                    rec.snoring = True

        return rec

    def _normalize_diet(self, relations: List[Relation]) -> DietRecord:
        rec = DietRecord()
        if not relations:
            return rec

        all_flags = []
        quality_votes = {"poor": 0, "moderate": 0, "good": 0}

        for r in relations:
            if r.condition in quality_votes:
                quality_votes[r.condition] += 1
            all_flags.extend(r.flags)

        rec.quality = max(quality_votes, key=quality_votes.get)
        # Deduplicate and clean flags
        rec.flags = list(dict.fromkeys(all_flags))  # preserves order, removes dups

        return rec

    def _normalize_drug_use(self, relations: List[Relation]) -> DrugUseRecord:
        rec = DrugUseRecord()
        if not relations:
            return rec

        rec.status = self._consolidate_status([r.status for r in relations])

        substances = []
        for r in relations:
            if r.substance:
                for sub in r.substance.split(", "):
                    if sub and sub not in substances:
                        substances.append(sub)
            if "IVDU" in substances:
                rec.route = "IV"
            if r.temporal:
                rec.temporal = r.temporal

        rec.substances = substances
        return rec

    # ── Helpers ───────────────────────────────────

    def _consolidate_status(self, statuses: List[Optional[str]]) -> str:
        """
        Resolve conflicting statuses using clinical priority:
        "never" > "former" > "current" > "unknown"
        """
        statuses = [s for s in statuses if s]
        if "never" in statuses:
            return "never"
        if "former" in statuses:
            return "former"
        if "current" in statuses:
            return "current"
        return "unknown"

    def _classify_bmi(self, bmi: float) -> str:
        if bmi < 18.5:
            return "underweight"
        elif bmi < 25.0:
            return "normal"
        elif bmi < 30.0:
            return "overweight"
        elif bmi < 35.0:
            return "obese_I"
        elif bmi < 40.0:
            return "obese_II"
        else:
            return "obese_III"


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO)

    from modules.preprocessor import ClinicalPreprocessor
    from modules.ner_extractor import NERExtractor
    from modules.relation_extractor import RelationExtractor

    text = """Patient smokes 1.5 packs per day for the past 30 years.
Drinks 3 beers nightly.
BMI 34.2, consistent with class I obesity.
Sedentary lifestyle with no regular exercise.
Sleeps 4-5 hours per night; loud snoring noted, OSA suspected.
Diet is poor — high sodium, frequent fast food.
Denies illicit drug use."""

    preprocessor = ClinicalPreprocessor()
    processed = preprocessor.process("TEST_001", text)

    ner = NERExtractor(use_transformer=False)
    ner_result = ner.extract("TEST_001", processed.sentences)

    rel_extractor = RelationExtractor()
    rel_result = rel_extractor.extract("TEST_001", ner_result)

    normalizer = Normalizer()
    profile = normalizer.normalize(rel_result)

    print("=" * 60)
    print("NORMALIZED PATIENT PROFILE")
    print(json.dumps(profile.to_dict(), indent=2))
