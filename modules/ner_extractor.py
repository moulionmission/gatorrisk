"""
Module 2: NER Extractor
=======================
Named Entity Recognition for lifestyle risk factors.

Architecture:
  - Primary:  Rule-based extractor (regex + ontology) — fast, interpretable, no GPU needed
  - Secondary: Transformer-based NER (BioBERT/GatorTron) — higher recall, GPU recommended
  - Strategy:  Rule-based runs first; transformer fills gaps where rules miss

On HiPerGator:
  Set model_name = "uf-health/gatortron-base" or "uf-health/gatortron-og"
  and use_transformer = True.
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

ONTOLOGY_PATH = Path(__file__).parent.parent / "data" / "ontologies" / "risk_terms.json"


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Entity:
    """A single extracted entity span from a clinical note."""
    text: str               # Raw matched text
    label: str              # Risk factor category (e.g., "smoking", "bmi")
    sub_label: str          # Finer label (e.g., "smoking_trigger", "bmi_value")
    start: int              # Character start offset in the sentence
    end: int                # Character end offset
    sentence: str           # Source sentence
    confidence: float       # 0.0–1.0 (rule-based = 1.0 by convention)
    source: str             # "rule" or "transformer"


@dataclass
class NERResult:
    """All entities extracted from a single note."""
    note_id: str
    entities: List[Entity] = field(default_factory=list)

    def by_factor(self, factor: str) -> List[Entity]:
        return [e for e in self.entities if e.label == factor]

    def summary(self) -> Dict[str, int]:
        from collections import Counter
        return dict(Counter(e.label for e in self.entities))


# ─────────────────────────────────────────────
# Rule-Based Extractor
# ─────────────────────────────────────────────

class RuleBasedExtractor:
    """
    Fast regex + ontology-based entity extraction.

    Patterns are compiled once at init for performance.
    Covers the most common clinical phrasings for each risk factor.
    """

    def __init__(self, ontology: dict):
        self.ontology = ontology
        self._patterns = self._compile_patterns()
        logger.info(f"RuleBasedExtractor: compiled patterns for {list(self._patterns.keys())}")

    def _compile_patterns(self) -> Dict[str, List[Tuple[re.Pattern, str]]]:
        """Build (pattern, sub_label) tuples per risk factor."""
        patterns = {}

        # ── Smoking ────────────────────────────────────
        patterns["smoking"] = [
            (re.compile(r'(\d+\.?\d*)\s*(pack[s]?\s*per\s*day|ppd)', re.I), "smoking_ppd"),
            (re.compile(r'(\d+\.?\d*)\s*pack[-\s]?year[s]?', re.I), "smoking_pack_years"),
            (re.compile(r'(\d+)\s*cigarette[s]?\s*(per|a)\s*day', re.I), "smoking_cigarettes_day"),
            # Former / quit patterns — expanded
            (re.compile(r'\b(former\s+smoker|ex[-\s]smoker|stopped\s+smoking)\b', re.I), "smoking_former"),
            (re.compile(r'\b(history\s+of\s+tobacco\s+use|history\s+of\s+smoking)\b', re.I), "smoking_former"),
            (re.compile(r'\b(tobacco\s+use[,.]?\s*which\s+he\s+quit|tobacco\s+use[,.]?\s*which\s+she\s+quit)\b', re.I), "smoking_former"),
            (re.compile(r'\b(quit\s+(?:at\s+(?:the\s+)?age|tobacco))\b', re.I), "smoking_former"),
            (re.compile(r'\b(used\s+to\s+smoke|previously\s+smoked|no\s+longer\s+smokes?)\b', re.I), "smoking_former"),
            # Never patterns — expanded
            (re.compile(r'\b(never\s+smoked|non[-\s]smoker|nonsmoker|denies\s+smoking)\b', re.I), "smoking_never"),
            (re.compile(r'\b(negative\s+for\s+tobacco|no\s+tobacco|no\s+history\s+of\s+smoking)\b', re.I), "smoking_never"),
            (re.compile(r'\bnever\s+(?:used\s+)?tobacco\b', re.I), "smoking_never"),
            # Current patterns
            (re.compile(r'\b(current|active|still)\s+smoker\b', re.I), "smoking_current"),
            (re.compile(r'\bsmokes?\b', re.I), "smoking_trigger"),
            (re.compile(r'\btobacco\b', re.I), "smoking_trigger"),
            (re.compile(r'\b(vaping|e-cigarette|vape|chewing\s+tobacco|chew|dip)\b', re.I), "smoking_vaping"),
            # Less than X cigarettes patterns
            (re.compile(r'\b(less\s+than\s+\d+\s+cigarettes?\s+(?:a|per)\s+day)\b', re.I), "smoking_light"),
        ]

        # ── Alcohol ────────────────────────────────────
        patterns["alcohol"] = [
            (re.compile(r'(\d+\.?\d*)\s*(drink[s]?|beer[s]?|glass(?:es)?)\s*(per|a|/)\s*(day|night|week)', re.I), "alcohol_quantity"),
            (re.compile(r'(\d+\.?\d*)\s*(drink[s]?|beer[s]?|glass(?:es)?)\s*(nightly|daily|weekly)', re.I), "alcohol_quantity"),
            # "one alcoholic drink per day" — common phrasing
            (re.compile(r'(one|two|three|four|five|six)\s+alcoholic\s+drink[s]?\s*(per|a)\s*(day|week|night)', re.I), "alcohol_quantity_words"),
            (re.compile(r'\bone\s+drink\s+(?:per|a)\s+day\b', re.I), "alcohol_quantity"),
            (re.compile(r'\b(social\s+drinker|drinks\s+socially|occasional\s+(?:alcohol|drink|drinker)|rarely\s+(?:consumes|drinks|uses)(?:\s+alcohol|\s+etoh)?)\b', re.I), "alcohol_social"),
            (re.compile(r'\b(heavy\s+drinker|heavy\s+alcohol\s+use|alcohol\s+abuse|alcoholism)\b', re.I), "alcohol_heavy"),
            (re.compile(r'\b(sober|sobriety|in\s+recovery|abstains?\s+from\s+alcohol|quit\s+drinking)\b', re.I), "alcohol_former"),
            (re.compile(r'\b(denies\s+alcohol|no\s+alcohol|teetotal|never\s+drinks|negative\s+for\s+alcohol)\b', re.I), "alcohol_never"),
            (re.compile(r'\bnegative\s+for\s+(?:illicit\s+drugs[,\s]+alcohol|alcohol)\b', re.I), "alcohol_never"),
            (re.compile(r'\b(etoh|ethanol)\b', re.I), "alcohol_trigger"),
            # "alcoholic drink" without quantity — still signals use
            (re.compile(r'\balcoholic\s+drink[s]?\b', re.I), "alcohol_trigger"),
            (re.compile(r'\bdrinks?\s+alcohol\b', re.I), "alcohol_trigger"),
        ]

        # ── BMI ────────────────────────────────────────
        patterns["bmi"] = [
            (re.compile(r'\bbmi\s*(?:of|is|=|:)?\s*(\d{2,3}\.?\d*)\b', re.I), "bmi_value"),
            (re.compile(r'\bbody\s+mass\s+index\s*(?:of|is|=|:)?\s*(\d{2,3}\.?\d*)\b', re.I), "bmi_value"),
            (re.compile(r'\b(morbidly\s+obese|obese\s+class\s+(?:I{1,3}|1|2|3)|class\s+(?:I{1,3}|1|2|3)\s+obes\w+)\b', re.I), "bmi_class"),
            (re.compile(r'\b(overweight|obese|obesity|morbid\s+obesity)\b', re.I), "bmi_class"),
            (re.compile(r'\b(underweight|normal\s+weight|ideal\s+body\s+weight)\b', re.I), "bmi_class"),
            # Weight in lbs/kg — very common in clinical notes
            (re.compile(r'\bweigh[st]?\s+(\d{2,3})\s*(lbs?|kg|pounds?|kilograms?)\b', re.I), "bmi_weight"),
            (re.compile(r'\bweight\s+(?:is\s+|was\s+|of\s+)?(\d{2,3})\s*(lbs?|kg|pounds?)\b', re.I), "bmi_weight"),
            (re.compile(r'\b(\d{2,3})\s*(lbs?|pounds?|kg)\b', re.I), "bmi_weight"),
            # "weighs X pounds" / "weight X kg"
            (re.compile(r'\bweighs?\s+(\d{2,3}\.?\d*)\s*(lbs?|pounds?|kg|kilograms?)\b', re.I), "bmi_weight"),
            # Weight loss context
            (re.compile(r'\bweight\s+loss\b', re.I), "bmi_weight_loss"),
            (re.compile(r'\b(gained?|lost?)\s+\d+\s+(lbs?|pounds?|kg)\b', re.I), "bmi_weight_change"),
        ]

        # ── Physical Activity ──────────────────────────
        patterns["physical_activity"] = [
            (re.compile(r'\b(\d+)\s*(day[s]?|time[s]?)\s*(per|a|/)\s*week\b.*?(exercise|walk|run|gym|workout)', re.I), "activity_frequency"),
            (re.compile(r'\b(\d+)\s*minute[s]?\s*(per|a|/)\s*(day|session)\b', re.I), "activity_duration"),
            (re.compile(r'\b(\d+[,.]?\d*)\s*mile[s]?\s*(per|a|/)\s*(day|week)\b', re.I), "activity_distance"),
            (re.compile(r'\b(sedentary\s+lifestyle|no\s+regular\s+exercise|does\s+not\s+exercise|physically\s+inactive)\b', re.I), "activity_sedentary"),
            (re.compile(r'\b(not\s+(?:very\s+)?active|no\s+physical\s+activity|inactive)\b', re.I), "activity_sedentary"),
            (re.compile(r'\b(physically\s+active|exercises\s+regularly|active\s+lifestyle)\b', re.I), "activity_active"),
            (re.compile(r'\b(walks?|runs?|jogs?|swims?|cycles?|gyms?|workouts?|aerobic|cardio)\b', re.I), "activity_type"),
            # "does cardio" / "does exercise at home" — common in MTSamples
            (re.compile(r'\bdoes?\s+(cardio|exercise[s]?|workout[s]?)\b', re.I), "activity_active"),
            (re.compile(r'\bexercises?\s+(at\s+home|regularly|daily|three\s+times|twice)\b', re.I), "activity_active"),
            # Difficulty with activity — signals low activity
            (re.compile(r'\bdifficulty\s+(walking|climbing\s+stairs|exercising)\b', re.I), "activity_sedentary"),
        ]

        # ── Sleep ──────────────────────────────────────
        patterns["sleep"] = [
            (re.compile(r'(\d+\.?\d*)\s*(?:-\s*(\d+\.?\d*))?\s*hour[s]?\s*(?:of\s+)?sleep', re.I), "sleep_hours"),
            (re.compile(r'sleep[s]?\s+(\d+\.?\d*)\s*(?:-\s*(\d+\.?\d*))?\s*hour[s]?', re.I), "sleep_hours"),
            (re.compile(r'\b(obstructive\s+sleep\s+apnea|osa|sleep\s+apnea)\b', re.I), "sleep_apnea"),
            (re.compile(r'\b(insomnia|difficulty\s+sleeping|can\'t\s+sleep|trouble\s+sleeping)\b', re.I), "sleep_insomnia"),
            (re.compile(r'\b(cpap|bipap)\b', re.I), "sleep_cpap"),
            (re.compile(r'\b(snoring|snores|loud\s+snoring|difficulty\s+snoring)\b', re.I), "sleep_snoring"),
            (re.compile(r'\b(hypersomnia|narcolepsy|excessive\s+daytime\s+sleepiness)\b', re.I), "sleep_hypersomnia"),
            (re.compile(r'\bosa\s+(suspected|likely|probable|confirmed|diagnosed)\b', re.I), "sleep_osa_status"),
            # Fatigue as sleep proxy
            (re.compile(r'\b(chronic\s+fatigue|excessive\s+fatigue|daytime\s+fatigue)\b', re.I), "sleep_fatigue"),
            # Non-restorative sleep
            (re.compile(r'\b(non[-\s]restorative\s+sleep|poor\s+sleep|wakes?\s+(?:up\s+)?frequently)\b', re.I), "sleep_insomnia"),
        ]

        # ── Diet ───────────────────────────────────────
        patterns["diet"] = [
            (re.compile(r'\b(high\s+sodium|high\s+salt|low\s+sodium|low\s+salt)\b', re.I), "diet_sodium"),
            (re.compile(r'\b(high\s+fat|low\s+fat|high\s+carb|low\s+carb)\b', re.I), "diet_macro"),
            (re.compile(r'\b(fast\s+food|processed\s+food|junk\s+food)\b', re.I), "diet_quality_poor"),
            (re.compile(r'\b(balanced\s+diet|healthy\s+diet|mediterranean\s+diet|well[-\s]balanced)\b', re.I), "diet_quality_good"),
            (re.compile(r'\b(poor\s+diet|unhealthy\s+diet|poor\s+nutrition)\b', re.I), "diet_quality_poor"),
            (re.compile(r'\b(vegetarian|vegan|plant[-\s]based)\b', re.I), "diet_type"),
            (re.compile(r'\b(diabetic\s+diet|cardiac\s+diet|renal\s+diet|low[-\s]carb\s+diet)\b', re.I), "diet_therapeutic"),
            (re.compile(r'\b(skips?\s+meals?|poor\s+appetite|decreased\s+appetite)\b', re.I), "diet_behavior"),
            # "eating and drinking well" — positive signal
            (re.compile(r'\b(eating\s+(?:and\s+drinking\s+)?well|good\s+appetite)\b', re.I), "diet_quality_good"),
            # Soft drinks / fruit drinks — dietary flag
            (re.compile(r'\b(soft\s+drinks?|fruit\s+drinks?|soda|sugary\s+drinks?)\b', re.I), "diet_quality_poor"),
            # Big portions / overeating
            (re.compile(r'\b(big\s+portions?|overeating|emotional\s+eater|binge\s+eating)\b', re.I), "diet_behavior"),
            # Weight loss program context
            (re.compile(r'\b(weight\s+watchers?|calorie\s+restrict|dieting)\b', re.I), "diet_therapeutic"),
        ]

        # ── Drug Use ───────────────────────────────────
        patterns["drug_use"] = [
            (re.compile(r'\b(marijuana|cannabis|weed|pot)\b', re.I), "drug_marijuana"),
            (re.compile(r'\b(cocaine|crack|coke)\b', re.I), "drug_cocaine"),
            (re.compile(r'\b(heroin|opioid[s]?|fentanyl)\b.*?(use|use[rs]?|abuse|addict)', re.I), "drug_opioid"),
            (re.compile(r'\b(heroin|fentanyl)\b', re.I), "drug_opioid"),
            (re.compile(r'\b(methamphetamine|meth|crystal\s+meth|amphetamine)\b', re.I), "drug_stimulant"),
            (re.compile(r'\b(ivdu|intravenous\s+drug\s+use[r]?|iv\s+drug)\b', re.I), "drug_ivdu"),
            (re.compile(r'\b(denies\s+(?:illicit\s+)?drug|no\s+(?:illicit\s+)?drug|drug[-\s]free)\b', re.I), "drug_never"),
            (re.compile(r'\bnegative\s+for\s+illicit\s+drugs?\b', re.I), "drug_never"),
            # "got addicted to drugs" — MTSamples real pattern
            (re.compile(r'\b(addict(?:ed)?\s+to\s+drugs?|drug\s+addict)\b', re.I), "drug_trigger"),
            (re.compile(r'\b(illicit\s+drug|recreational\s+drug|substance\s+abuse|substance\s+use)\b', re.I), "drug_trigger"),
            (re.compile(r'\b(no\s+history\s+of\s+(?:drug|substance)\s+use|denies\s+(?:drug|substance)\s+use)\b', re.I), "drug_never"),
        ]

        return patterns

    def extract(self, note_id: str, sentences: List[str]) -> NERResult:
        """
        Run rule-based extraction over all sentences.

        Args:
            note_id: Identifier for the note
            sentences: Preprocessed sentence list

        Returns:
            NERResult with all found entities
        """
        result = NERResult(note_id=note_id)

        for sentence in sentences:
            for factor, pattern_list in self._patterns.items():
                for pattern, sub_label in pattern_list:
                    for match in pattern.finditer(sentence):
                        entity = Entity(
                            text=match.group(0),
                            label=factor,
                            sub_label=sub_label,
                            start=match.start(),
                            end=match.end(),
                            sentence=sentence,
                            confidence=1.0,
                            source="rule",
                        )
                        result.entities.append(entity)

        logger.debug(f"[{note_id}] Rule extractor found {len(result.entities)} entities: {result.summary()}")
        return result


# ─────────────────────────────────────────────
# Transformer-Based Extractor (Stub for HiPerGator)
# ─────────────────────────────────────────────

class TransformerNERExtractor:
    """
    BioBERT / GatorTron-based NER extractor.

    This stub loads a HuggingFace NER model and runs inference.
    On a local machine without GPU, this is slow — use rule-based instead.
    On HiPerGator: set model_name="uf-health/gatortron-base" for best clinical performance.

    The transformer runs in ADDITION to rules, filling in entities
    that the rule system misses (lower confidence entities, novel phrasings).
    """

    def __init__(self, model_name: str = "d4data/biomedical-ner-all", confidence_threshold: float = 0.65):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self._pipeline = None
        logger.info(f"TransformerNERExtractor configured: model={model_name}")

    def _load(self):
        """Lazy-load model on first use."""
        if self._pipeline is None:
            try:
                from transformers import pipeline
                logger.info(f"Loading transformer model: {self.model_name} ...")
                self._pipeline = pipeline("ner", model=self.model_name, aggregation_strategy="simple")
                logger.info("Transformer model loaded.")
            except Exception as e:
                logger.error(f"Failed to load transformer model: {e}")
                self._pipeline = None

    def extract(self, note_id: str, sentences: List[str]) -> NERResult:
        """Run transformer NER over sentences. Falls back gracefully if model unavailable."""
        self._load()
        result = NERResult(note_id=note_id)

        if self._pipeline is None:
            logger.warning("Transformer not available — returning empty result.")
            return result

        for sentence in sentences:
            try:
                predictions = self._pipeline(sentence)
                for pred in predictions:
                    if pred["score"] < self.confidence_threshold:
                        continue
                    # Map generic NER labels to our risk factor taxonomy
                    label = self._map_label(pred["entity_group"], sentence)
                    if label is None:
                        continue
                    entity = Entity(
                        text=pred["word"],
                        label=label,
                        sub_label=f"transformer_{pred['entity_group'].lower()}",
                        start=pred["start"],
                        end=pred["end"],
                        sentence=sentence,
                        confidence=round(pred["score"], 4),
                        source="transformer",
                    )
                    result.entities.append(entity)
            except Exception as e:
                logger.warning(f"Transformer inference error on sentence: {e}")

        return result

    def _map_label(self, entity_group: str, context: str) -> Optional[str]:
        """
        Map generic biomedical NER labels to our risk factor categories.
        This is a simplistic mapping — extend based on the specific model's label set.
        """
        label_map = {
            "SIGN_SYMPTOM": None,      # not directly a risk factor
            "DISEASE_DISORDER": None,
            "MEDICATION": None,
            "SUBSTANCE": self._infer_substance(context),
            "BEHAVIOR": self._infer_behavior(context),
            "LABORATORY_OR_TEST_RESULT": "bmi",
        }
        return label_map.get(entity_group)

    def _infer_substance(self, context: str) -> Optional[str]:
        context_lower = context.lower()
        if any(w in context_lower for w in ["smok", "cigarette", "tobacco", "nicotine"]):
            return "smoking"
        if any(w in context_lower for w in ["alcohol", "beer", "wine", "etoh", "drink"]):
            return "alcohol"
        if any(w in context_lower for w in ["marijuana", "cannabis", "cocaine", "heroin", "meth"]):
            return "drug_use"
        return None

    def _infer_behavior(self, context: str) -> Optional[str]:
        context_lower = context.lower()
        if any(w in context_lower for w in ["exercise", "sedentary", "active", "walk", "run"]):
            return "physical_activity"
        if any(w in context_lower for w in ["sleep", "insomnia", "apnea", "snor"]):
            return "sleep"
        if any(w in context_lower for w in ["diet", "eat", "food", "sodium", "nutrition"]):
            return "diet"
        return None


# ─────────────────────────────────────────────
# Combined NER Extractor
# ─────────────────────────────────────────────

class NERExtractor:
    """
    Orchestrates rule-based and transformer extractors.

    Strategy:
      1. Run rules (always) — fast, high precision
      2. Run transformer (if enabled) — higher recall
      3. Merge, deduplicating overlapping spans
    """

    def __init__(self, use_transformer: bool = False, model_name: str = "d4data/biomedical-ner-all",
                 confidence_threshold: float = 0.65):
        # Load ontology
        with open(ONTOLOGY_PATH) as f:
            self.ontology = json.load(f)

        self.rule_extractor = RuleBasedExtractor(self.ontology)
        self.use_transformer = use_transformer

        if use_transformer:
            self.transformer_extractor = TransformerNERExtractor(model_name, confidence_threshold)

        logger.info(f"NERExtractor ready | transformer={'ON' if use_transformer else 'OFF'}")

    def extract(self, note_id: str, sentences: List[str]) -> NERResult:
        """Run full NER extraction pipeline."""
        # Step 1: Rule-based
        rule_result = self.rule_extractor.extract(note_id, sentences)

        if not self.use_transformer:
            return rule_result

        # Step 2: Transformer
        transformer_result = self.transformer_extractor.extract(note_id, sentences)

        # Step 3: Merge (prefer rule entities; add transformer only if non-overlapping)
        merged = NERResult(note_id=note_id)
        merged.entities = rule_result.entities.copy()
        existing_spans = {(e.sentence[:30], e.start, e.end) for e in merged.entities}

        for entity in transformer_result.entities:
            key = (entity.sentence[:30], entity.start, entity.end)
            if key not in existing_spans:
                merged.entities.append(entity)

        logger.info(f"[{note_id}] Merged NER: {merged.summary()}")
        return merged


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sentences = [
        "Patient smokes 1.5 packs per day for 30 years.",
        "Drinks 3 beers nightly.",
        "BMI 34.2, consistent with class I obesity.",
        "Sedentary lifestyle with no regular exercise.",
        "Sleeps 4-5 hours per night; loud snoring noted, OSA suspected.",
        "Diet is poor — high sodium, frequent fast food.",
        "Denies illicit drug use.",
    ]

    extractor = NERExtractor(use_transformer=False)
    result = extractor.extract("TEST_001", sentences)

    print("=" * 60)
    print(f"Note: {result.note_id}")
    print(f"Entity Summary: {result.summary()}")
    print("\nAll Entities:")
    for e in result.entities:
        print(f"  [{e.label:<20}] ({e.sub_label:<30}) → '{e.text}'")
