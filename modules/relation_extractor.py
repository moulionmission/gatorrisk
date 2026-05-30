"""
Module 3: Relation Extractor
============================
Links extracted entities to their associated values, quantities,
statuses, and temporal information within the same sentence context.

Example:
  Input entities: [smoking_trigger: "smokes"], [smoking_ppd: "1.5 packs per day"]
  Output relation: smoking → {trigger: "smokes", quantity: 1.5, unit: "ppd"}

This module works at the SENTENCE level — entities in the same
sentence are candidates for relation linking.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from modules.ner_extractor import Entity, NERResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Relation:
    """A linked entity-value pair for a single risk factor mention."""
    factor: str                         # e.g., "smoking"
    raw_text: str                       # source sentence
    status: Optional[str] = None       # "current", "former", "never"
    value: Optional[float] = None      # numeric quantity
    unit: Optional[str] = None         # unit of measurement
    value2: Optional[float] = None     # for ranges: "4-5 hours" → value=4, value2=5
    substance: Optional[str] = None    # for drug_use
    condition: Optional[str] = None    # for sleep: OSA, insomnia, etc.
    temporal: Optional[str] = None     # "for 30 years", "since 2018", "quit 2019"
    flags: List[str] = field(default_factory=list)  # for diet
    confidence: float = 1.0
    source_entities: List[str] = field(default_factory=list)


@dataclass
class RelationResult:
    """All extracted relations from a single note."""
    note_id: str
    relations: List[Relation] = field(default_factory=list)
    sentences: List[str] = field(default_factory=list)

    def by_factor(self, factor: str) -> List[Relation]:
        return [r for r in self.relations if r.factor == factor]


# ─────────────────────────────────────────────
# Relation Extractor
# ─────────────────────────────────────────────

class RelationExtractor:
    """
    Extracts structured relations from NER entities grouped by sentence.

    For each risk factor found in a sentence, it:
      1. Determines status (current / former / never)
      2. Extracts quantities and units
      3. Extracts temporal expressions
      4. Extracts substance type (for drug use)
      5. Builds a Relation object
    """

    # Status patterns (shared across factors)
    STATUS_PATTERNS = {
        "never": re.compile(
            r'\b(never|denies?|no\s+history\s+of|non[-\s]?smoker|nonsmoker|drug[-\s]free|abstain|teetotal)\b', re.I
        ),
        "former": re.compile(
            r'\b(former|ex[-\s]?|quit|stopped|ceased|used\s+to|previously|in\s+recovery|sober|sobriety)\b', re.I
        ),
    }

    # Temporal extraction
    TEMPORAL_PATTERN = re.compile(
        r'\b(for\s+(?:the\s+)?(?:past\s+)?\d+\s+years?|since\s+\d{4}|quit\s+in\s+\d{4}|'
        r'quit\s+\d+\s+years?\s+ago|\d+[-\s]year\s+history)\b', re.I
    )

    # Numeric value extractor (handles "1.5", "3", "4-5")
    NUMERIC_RANGE = re.compile(r'(\d+\.?\d*)\s*[-–to]+\s*(\d+\.?\d*)')
    NUMERIC_SINGLE = re.compile(r'(\d+\.?\d*)')

    def extract(self, note_id: str, ner_result: NERResult) -> RelationResult:
        """
        Extract relations from NER results.

        Args:
            note_id: Note identifier
            ner_result: Output of the NER extractor

        Returns:
            RelationResult with structured Relation objects
        """
        # Collect unique sentences from entities
        all_sentences = list(dict.fromkeys(e.sentence for e in ner_result.entities))
        result = RelationResult(note_id=note_id, sentences=all_sentences)

        # Group entities by sentence
        sentence_groups = self._group_by_sentence(ner_result.entities)

        for sentence, entities in sentence_groups.items():
            # Group entities by factor within the sentence
            factor_groups = self._group_by_factor(entities)
            for factor, factor_entities in factor_groups.items():
                relation = self._build_relation(factor, sentence, factor_entities)
                if relation:
                    result.relations.append(relation)

        logger.debug(f"[{note_id}] RelationExtractor: {len(result.relations)} relations")
        return result

    # ── Private Methods ──────────────────────────

    def _group_by_sentence(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        groups: Dict[str, List[Entity]] = {}
        for e in entities:
            groups.setdefault(e.sentence, []).append(e)
        return groups

    def _group_by_factor(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        groups: Dict[str, List[Entity]] = {}
        for e in entities:
            groups.setdefault(e.label, []).append(e)
        return groups

    def _build_relation(self, factor: str, sentence: str, entities: List[Entity]) -> Optional[Relation]:
        """Dispatch to factor-specific relation builder."""
        builders = {
            "smoking": self._build_smoking,
            "alcohol": self._build_alcohol,
            "bmi": self._build_bmi,
            "physical_activity": self._build_activity,
            "sleep": self._build_sleep,
            "diet": self._build_diet,
            "drug_use": self._build_drug_use,
        }
        builder = builders.get(factor)
        if builder:
            return builder(sentence, entities)
        return None

    def _determine_status(self, sentence: str, entities: List[Entity]) -> str:
        """Determine current/former/never status from sentence context."""
        sub_labels = {e.sub_label for e in entities}

        # Check entity sub-labels first (most precise)
        for sub in sub_labels:
            if "never" in sub:
                return "never"
            if "former" in sub:
                return "former"
            if "current" in sub:
                return "current"

        # Fall back to sentence-level pattern matching
        if self.STATUS_PATTERNS["never"].search(sentence):
            return "never"
        if self.STATUS_PATTERNS["former"].search(sentence):
            return "former"
        return "current"

    def _extract_temporal(self, sentence: str) -> Optional[str]:
        m = self.TEMPORAL_PATTERN.search(sentence)
        return m.group(0) if m else None

    def _extract_numeric(self, text: str) -> Tuple[Optional[float], Optional[float]]:
        """Extract single value or range. Returns (value, value2)."""
        range_match = self.NUMERIC_RANGE.search(text)
        if range_match:
            return float(range_match.group(1)), float(range_match.group(2))
        single_match = self.NUMERIC_SINGLE.search(text)
        if single_match:
            return float(single_match.group(1)), None
        return None, None

    # ── Factor-Specific Builders ─────────────────

    def _build_smoking(self, sentence: str, entities: List[Entity]) -> Relation:
        rel = Relation(
            factor="smoking",
            raw_text=sentence,
            status=self._determine_status(sentence, entities),
            temporal=self._extract_temporal(sentence),
            source_entities=[e.sub_label for e in entities],
        )
        for e in entities:
            if e.sub_label == "smoking_ppd":
                rel.value, _ = self._extract_numeric(e.text)
                rel.unit = "ppd"
            elif e.sub_label == "smoking_pack_years":
                m = self.NUMERIC_SINGLE.search(e.text)
                if m:
                    rel.flags.append(f"pack_years:{m.group(1)}")
            elif e.sub_label == "smoking_cigarettes_day":
                rel.value, _ = self._extract_numeric(e.text)
                rel.unit = "cigarettes/day"
                if rel.value:
                    rel.flags.append(f"ppd_approx:{round(rel.value/20, 2)}")
        return rel

    def _build_alcohol(self, sentence: str, entities: List[Entity]) -> Relation:
        rel = Relation(
            factor="alcohol",
            raw_text=sentence,
            status=self._determine_status(sentence, entities),
            temporal=self._extract_temporal(sentence),
            source_entities=[e.sub_label for e in entities],
        )
        for e in entities:
            if e.sub_label == "alcohol_quantity":
                rel.value, _ = self._extract_numeric(e.text)
                # Determine unit from text
                if re.search(r'(per|a|/)\s*week|weekly', e.text, re.I):
                    rel.unit = "drinks/week"
                elif re.search(r'(per|a|/)\s*(day|night)|nightly|daily', e.text, re.I):
                    rel.unit = "drinks/day"
            elif e.sub_label == "alcohol_social":
                rel.flags.append("social_drinker")
            elif e.sub_label == "alcohol_heavy":
                rel.flags.append("heavy_use")
        return rel

    def _build_bmi(self, sentence: str, entities: List[Entity]) -> Relation:
        rel = Relation(
            factor="bmi",
            raw_text=sentence,
            status="current",
            source_entities=[e.sub_label for e in entities],
        )
        for e in entities:
            if e.sub_label == "bmi_value":
                rel.value, _ = self._extract_numeric(e.text)
                rel.unit = "kg/m2"
                if rel.value:
                    rel.condition = self._classify_bmi(rel.value)
            elif e.sub_label == "bmi_class":
                # Extract class from text if no numeric value yet
                text_lower = e.text.lower()
                if "morbidly obese" in text_lower or "class iii" in text_lower or "class 3" in text_lower:
                    rel.flags.append("obese_III")
                elif "class ii" in text_lower or "class 2" in text_lower:
                    rel.flags.append("obese_II")
                elif "class i" in text_lower or "class 1" in text_lower:
                    rel.flags.append("obese_I")
                elif "overweight" in text_lower:
                    rel.flags.append("overweight")
                elif "underweight" in text_lower:
                    rel.flags.append("underweight")
            elif e.sub_label == "bmi_weight":
                rel.value, _ = self._extract_numeric(e.text)
                if re.search(r'(lbs?|pounds?)', e.text, re.I):
                    rel.unit = "lbs"
                elif re.search(r'(kg|kilograms?)', e.text, re.I):
                    rel.unit = "kg"
        return rel

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

    def _build_activity(self, sentence: str, entities: List[Entity]) -> Relation:
        rel = Relation(
            factor="physical_activity",
            raw_text=sentence,
            status="current",
            source_entities=[e.sub_label for e in entities],
        )
        for e in entities:
            if e.sub_label == "activity_sedentary":
                rel.condition = "sedentary"
                rel.flags.append("sedentary")
            elif e.sub_label == "activity_active":
                rel.condition = "active"
            elif e.sub_label == "activity_frequency":
                val, _ = self._extract_numeric(e.text)
                if val:
                    rel.flags.append(f"days_per_week:{int(val)}")
            elif e.sub_label == "activity_duration":
                val, _ = self._extract_numeric(e.text)
                if val:
                    rel.value = val
                    rel.unit = "min/session"
            elif e.sub_label == "activity_distance":
                val, _ = self._extract_numeric(e.text)
                if val:
                    rel.flags.append(f"miles:{val}")
            elif e.sub_label == "activity_type":
                rel.flags.append(f"type:{e.text.lower()}")
        return rel

    def _build_sleep(self, sentence: str, entities: List[Entity]) -> Relation:
        rel = Relation(
            factor="sleep",
            raw_text=sentence,
            status="current",
            source_entities=[e.sub_label for e in entities],
        )
        for e in entities:
            if e.sub_label == "sleep_hours":
                val, val2 = self._extract_numeric(e.text)
                rel.value = val
                rel.value2 = val2
                rel.unit = "hours/night"
                if val and val2:
                    rel.flags.append(f"avg_hours:{round((val+val2)/2, 1)}")
            elif e.sub_label in ("sleep_apnea", "sleep_osa_status"):
                rel.condition = "OSA"
                osa_status = re.search(r'(suspected|likely|probable|confirmed|diagnosed)', sentence, re.I)
                if osa_status:
                    rel.flags.append(f"osa_{osa_status.group(1).lower()}")
            elif e.sub_label == "sleep_insomnia":
                rel.condition = "insomnia"
            elif e.sub_label == "sleep_cpap":
                rel.flags.append("cpap_user")
            elif e.sub_label == "sleep_snoring":
                rel.flags.append("snoring")
        return rel

    def _build_diet(self, sentence: str, entities: List[Entity]) -> Relation:
        rel = Relation(
            factor="diet",
            raw_text=sentence,
            status="current",
            source_entities=[e.sub_label for e in entities],
        )
        quality_votes = {"poor": 0, "moderate": 0, "good": 0}

        for e in entities:
            if e.sub_label == "diet_quality_poor":
                quality_votes["poor"] += 1
                rel.flags.append(e.text.lower().replace(" ", "_"))
            elif e.sub_label == "diet_quality_good":
                quality_votes["good"] += 1
            elif e.sub_label == "diet_sodium":
                rel.flags.append("high_sodium" if "high" in e.text.lower() else "low_sodium")
            elif e.sub_label == "diet_macro":
                rel.flags.append(e.text.lower().replace(" ", "_"))
            elif e.sub_label == "diet_type":
                rel.flags.append(e.text.lower())
            elif e.sub_label == "diet_therapeutic":
                rel.flags.append(e.text.lower().replace(" ", "_"))
            elif e.sub_label == "diet_behavior":
                rel.flags.append(e.text.lower().replace(" ", "_"))

        rel.condition = max(quality_votes, key=quality_votes.get)
        return rel

    def _build_drug_use(self, sentence: str, entities: List[Entity]) -> Relation:
        rel = Relation(
            factor="drug_use",
            raw_text=sentence,
            status=self._determine_status(sentence, entities),
            temporal=self._extract_temporal(sentence),
            source_entities=[e.sub_label for e in entities],
        )
        substance_map = {
            "drug_marijuana": "marijuana",
            "drug_cocaine": "cocaine",
            "drug_opioid": "opioid/heroin",
            "drug_stimulant": "methamphetamine",
            "drug_ivdu": "IVDU",
        }
        substances = []
        for e in entities:
            sub = substance_map.get(e.sub_label)
            if sub and sub not in substances:
                substances.append(sub)
        rel.substance = ", ".join(substances) if substances else None
        return rel


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO)

    from modules.preprocessor import ClinicalPreprocessor
    from modules.ner_extractor import NERExtractor

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

    print("=" * 60)
    print(f"Note: {rel_result.note_id}")
    print(f"Relations found: {len(rel_result.relations)}")
    for r in rel_result.relations:
        print(f"\n  [{r.factor.upper()}]")
        print(f"    Status  : {r.status}")
        print(f"    Value   : {r.value} {r.unit or ''}")
        if r.value2:
            print(f"    Value2  : {r.value2}")
        if r.condition:
            print(f"    Condition: {r.condition}")
        if r.substance:
            print(f"    Substance: {r.substance}")
        if r.temporal:
            print(f"    Temporal: {r.temporal}")
        if r.flags:
            print(f"    Flags   : {r.flags}")
