"""
Unit Tests — Lifestyle Risk Factor Extractor
============================================
Tests each module independently and the end-to-end pipeline.

Run:
    pytest tests/test_pipeline.py -v
    pytest tests/test_pipeline.py -v -k "test_smoking"   # single test
"""

import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.preprocessor import ClinicalPreprocessor
from modules.ner_extractor import NERExtractor
from modules.relation_extractor import RelationExtractor
from modules.normalizer import Normalizer
from modules.risk_scorer import RiskScorer
from modules.pipeline import Pipeline


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    return Pipeline(use_transformer=False, deidentify=True)

@pytest.fixture(scope="module")
def preprocessor():
    return ClinicalPreprocessor(deidentify=True)

@pytest.fixture(scope="module")
def ner():
    return NERExtractor(use_transformer=False)

@pytest.fixture(scope="module")
def full_chain():
    """Returns a helper that runs the full chain up to normalization."""
    pre = ClinicalPreprocessor()
    ner = NERExtractor(use_transformer=False)
    rel = RelationExtractor()
    norm = Normalizer()
    scorer = RiskScorer()

    def run(note_id: str, text: str):
        processed = pre.process(note_id, text)
        ner_result = ner.extract(note_id, processed.sentences)
        rel_result = rel.extract(note_id, ner_result)
        profile = norm.normalize(rel_result)
        risk = scorer.score(profile)
        return profile, risk

    return run


# ─────────────────────────────────────────────
# Module 1: Preprocessor Tests
# ─────────────────────────────────────────────

class TestPreprocessor:

    def test_basic_processing(self, preprocessor):
        text = "Patient smokes 2 ppd. BMI 32.1."
        result = preprocessor.process("T001", text)
        assert result.note_id == "T001"
        assert result.cleaned_text
        assert len(result.sentences) >= 1

    def test_deidentification_mrn(self, preprocessor):
        text = "MRN: 1234567 Patient smokes."
        result = preprocessor.process("T002", text)
        assert "1234567" not in result.cleaned_text
        assert "[MRN]" in result.cleaned_text

    def test_deidentification_dates(self, preprocessor):
        text = "Seen on 03/15/2024. Patient smokes 1 ppd."
        result = preprocessor.process("T003", text)
        assert "[DATE]" in result.cleaned_text

    def test_deidentification_provider(self, preprocessor):
        text = "Dr. Johnson reviewed the case. Patient denies smoking."
        result = preprocessor.process("T004", text)
        assert "Dr. [PROVIDER]" in result.cleaned_text

    def test_sentence_splitting(self, preprocessor):
        text = "Smokes 2 ppd.\nDrinks 3 beers daily.\nBMI 34."
        result = preprocessor.process("T005", text)
        assert len(result.sentences) >= 2

    def test_empty_line_filtering(self, preprocessor):
        text = "Smokes 2 ppd.\n\n\n\nBMI 34."
        result = preprocessor.process("T006", text)
        for s in result.sentences:
            assert s.strip()

    def test_batch_processing(self, preprocessor):
        notes = [
            {"note_id": "B001", "text": "Smokes 2 ppd."},
            {"note_id": "B002", "text": "Nonsmoker. BMI 22."},
        ]
        results = preprocessor.process_batch(notes)
        assert len(results) == 2
        assert results[0].note_id == "B001"


# ─────────────────────────────────────────────
# Module 2: NER Extractor Tests
# ─────────────────────────────────────────────

class TestNERExtractor:

    def test_smoking_ppd_detected(self, ner):
        sentences = ["Patient smokes 1.5 packs per day."]
        result = ner.extract("T010", sentences)
        smoking = result.by_factor("smoking")
        assert len(smoking) > 0
        ppd_entities = [e for e in smoking if e.sub_label == "smoking_ppd"]
        assert len(ppd_entities) > 0

    def test_smoking_never(self, ner):
        sentences = ["Patient is a nonsmoker."]
        result = ner.extract("T011", sentences)
        smoking = result.by_factor("smoking")
        never = [e for e in smoking if "never" in e.sub_label]
        assert len(never) > 0

    def test_alcohol_quantity(self, ner):
        sentences = ["Drinks 3 beers per day."]
        result = ner.extract("T012", sentences)
        alcohol = result.by_factor("alcohol")
        assert len(alcohol) > 0

    def test_bmi_value(self, ner):
        sentences = ["BMI 34.2, class I obesity."]
        result = ner.extract("T013", sentences)
        bmi = result.by_factor("bmi")
        assert len(bmi) > 0

    def test_sleep_hours_range(self, ner):
        sentences = ["Sleeps 4-5 hours per night."]
        result = ner.extract("T014", sentences)
        sleep = result.by_factor("sleep")
        hours = [e for e in sleep if e.sub_label == "sleep_hours"]
        assert len(hours) > 0

    def test_osa_detected(self, ner):
        sentences = ["Loud snoring noted, OSA suspected."]
        result = ner.extract("T015", sentences)
        sleep = result.by_factor("sleep")
        assert len(sleep) > 0

    def test_drug_marijuana(self, ner):
        sentences = ["Admits to daily marijuana use."]
        result = ner.extract("T016", sentences)
        drugs = result.by_factor("drug_use")
        assert len(drugs) > 0

    def test_drug_ivdu(self, ner):
        sentences = ["Known IVDU — heroin use, last use this morning."]
        result = ner.extract("T017", sentences)
        drugs = result.by_factor("drug_use")
        ivdu = [e for e in drugs if "ivdu" in e.sub_label]
        assert len(ivdu) > 0

    def test_summary_counts(self, ner):
        sentences = ["Smokes 2 ppd.", "BMI 34.", "Drinks 3 beers nightly."]
        result = ner.extract("T018", sentences)
        summary = result.summary()
        assert "smoking" in summary
        assert "bmi" in summary
        assert "alcohol" in summary


# ─────────────────────────────────────────────
# Module 3-5: Integration Tests (Normalization + Scoring)
# ─────────────────────────────────────────────

class TestNormalizationAndScoring:

    def test_smoking_current_status(self, full_chain):
        profile, _ = full_chain("T020", "Patient smokes 1.5 packs per day.")
        assert profile.smoking.status == "current"
        assert profile.smoking.ppd == 1.5

    def test_smoking_never_status(self, full_chain):
        profile, _ = full_chain("T021", "Patient is a nonsmoker. Never smoked.")
        assert profile.smoking.status == "never"

    def test_smoking_former_status(self, full_chain):
        profile, _ = full_chain("T022", "Former smoker, quit in 2018.")
        assert profile.smoking.status == "former"

    def test_bmi_numeric_extracted(self, full_chain):
        profile, _ = full_chain("T023", "BMI is 34.2.")
        assert profile.bmi.value == 34.2
        assert profile.bmi.bmi_class == "obese_I"

    def test_bmi_class_boundaries(self, full_chain):
        cases = [
            ("BMI 17.5.", "underweight"),
            ("BMI 22.0.", "normal"),
            ("BMI 27.5.", "overweight"),
            ("BMI 31.0.", "obese_I"),
            ("BMI 37.0.", "obese_II"),
            ("BMI 42.0.", "obese_III"),
        ]
        for text, expected_class in cases:
            profile, _ = full_chain(f"BMI_{expected_class}", text)
            assert profile.bmi.bmi_class == expected_class, f"Failed for: {text}"

    def test_sleep_hours_averaged(self, full_chain):
        profile, _ = full_chain("T024", "Patient sleeps 4-5 hours per night.")
        assert profile.sleep.hours_per_night == 4.5

    def test_sleep_osa_detected(self, full_chain):
        profile, _ = full_chain("T025", "Loud snoring, OSA suspected.")
        assert profile.sleep.condition == "OSA"
        assert profile.sleep.osa_status == "suspected"

    def test_alcohol_drinks_per_day(self, full_chain):
        profile, _ = full_chain("T026", "Drinks 3 beers nightly.")
        assert profile.alcohol.status == "current"
        assert profile.alcohol.drinks_per_day == 3.0

    def test_alcohol_never(self, full_chain):
        profile, _ = full_chain("T027", "Denies alcohol use.")
        assert profile.alcohol.status == "never"

    def test_drug_use_current(self, full_chain):
        profile, _ = full_chain("T028", "Admits to daily marijuana use and cocaine.")
        assert profile.drug_use.status == "current"
        assert len(profile.drug_use.substances) > 0

    def test_drug_use_never(self, full_chain):
        profile, _ = full_chain("T029", "Denies illicit drug use.")
        assert profile.drug_use.status == "never"

    def test_sedentary_activity(self, full_chain):
        profile, _ = full_chain("T030", "Sedentary lifestyle with no regular exercise.")
        assert profile.physical_activity.level == "sedentary"

    def test_diet_poor_flags(self, full_chain):
        profile, _ = full_chain("T031", "Diet is poor — high sodium, frequent fast food.")
        assert profile.diet.quality == "poor"
        assert len(profile.diet.flags) > 0

    def test_pipeline_refinements(self, full_chain):
        # 1. Past-tense denies
        profile, _ = full_chain("T_REF_1", "Denied tobacco/illicit drug use.")
        assert profile.smoking.status == "never"
        assert profile.drug_use.status == "never"

        # 2. Coordinated list negation
        profile, _ = full_chain("T_REF_2", "Negative for illicit drugs, alcohol, and tobacco.")
        assert profile.smoking.status == "never"
        assert profile.drug_use.status == "never"
        assert profile.alcohol.status == "never"

        # 3. Social drinker for rarely consumes
        profile, _ = full_chain("T_REF_3", "Rarely consumes ETOH.")
        assert profile.alcohol.pattern == "social"

        # 4. Overweight guard
        profile, _ = full_chain("T_REF_4", "He currently weighs 312 pounds. Ideal weight is 170 pounds. He is 142 pounds overweight.")
        assert profile.bmi.bmi_class is None


# ─────────────────────────────────────────────
# Risk Scorer Tests
# ─────────────────────────────────────────────

class TestRiskScorer:

    def test_heavy_smoker_high_risk(self, full_chain):
        _, risk = full_chain("T040", "Patient smokes 3 packs per day.")
        smoking_factor = next(f for f in risk.factors if f.factor == "smoking")
        assert smoking_factor.score >= 75
        assert smoking_factor.tier in ("HIGH", "CRITICAL")

    def test_never_smoker_low_risk(self, full_chain):
        _, risk = full_chain("T041", "Non-smoker, never smoked.")
        smoking_factor = next(f for f in risk.factors if f.factor == "smoking")
        assert smoking_factor.score == 0.0
        assert smoking_factor.tier == "LOW"

    def test_morbid_obesity_critical(self, full_chain):
        _, risk = full_chain("T042", "BMI 42.5, morbidly obese.")
        bmi_factor = next(f for f in risk.factors if f.factor == "bmi")
        assert bmi_factor.score == 80.0
        assert bmi_factor.tier == "CRITICAL"

    def test_normal_bmi_low_risk(self, full_chain):
        _, risk = full_chain("T043", "BMI 22.4, normal weight.")
        bmi_factor = next(f for f in risk.factors if f.factor == "bmi")
        assert bmi_factor.score == 5.0

    def test_composite_score_range(self, full_chain):
        _, risk = full_chain("T044", "Smokes 2 ppd. BMI 38. Drinks 4 beers daily. Sedentary.")
        assert 0.0 <= risk.composite_score <= 100.0

    def test_healthy_patient_low_composite(self, full_chain):
        _, risk = full_chain("T045",
            "Non-smoker. BMI 21. Exercises 5 days per week, 45 minutes each. "
            "Sleeps 8 hours. Healthy diet. Denies alcohol and drug use."
        )
        assert risk.composite_score < 30.0
        assert risk.composite_tier in ("LOW", "MODERATE")

    def test_high_risk_patient_critical(self, full_chain):
        _, risk = full_chain("T046",
            "Smokes 2 packs per day. BMI 42, morbidly obese. "
            "Drinks 6 beers nightly. Sedentary. Sleeps 3-4 hours. "
            "Poor diet, high sodium. Active heroin IVDU."
        )
        assert risk.composite_score >= 60.0
        assert risk.composite_tier in ("HIGH", "CRITICAL")

    def test_score_has_disclaimer(self, full_chain):
        _, risk = full_chain("T047", "Smokes 1 ppd.")
        assert "RESEARCH USE ONLY" in risk.disclaimer


# ─────────────────────────────────────────────
# End-to-End Pipeline Tests
# ─────────────────────────────────────────────

class TestPipeline:

    def test_single_note_runs(self, pipeline):
        result = pipeline.run_note("E001", "Patient smokes 2 ppd. BMI 34.")
        assert result.note_id == "E001"
        assert result.processing_time_ms > 0
        assert result.risk_profile.composite_score >= 0

    def test_empty_note_handled(self, pipeline):
        result = pipeline.run_note("E002", "Patient presents for routine checkup.")
        assert result.note_id == "E002"
        # Should not crash on sparse notes

    def test_batch_runs(self, pipeline):
        notes = [
            {"note_id": "B001", "text": "Smokes 2 ppd. BMI 34."},
            {"note_id": "B002", "text": "Non-smoker. Exercises daily. BMI 22."},
        ]
        results = pipeline.run_batch(notes)
        assert len(results) == 2

    def test_output_serializable(self, pipeline):
        result = pipeline.run_note("E003", "BMI 28. Drinks 2 beers nightly.")
        d = result.to_dict()
        json_str = result.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "risk_profile" in parsed

    def test_note_file_processing(self, pipeline):
        notes_path = Path(__file__).parent.parent / "data" / "sample_notes.json"
        results = pipeline.run_from_file(str(notes_path))
        assert len(results) == 5  # 5 notes in sample file
        for r in results:
            assert r.risk_profile.composite_score >= 0.0
