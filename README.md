# 🐊 GatorRisk
### Clinical NLP Pipeline for Lifestyle Risk Factor Extraction
**Built at the University of Florida — extending the CTSI NLP Core GatorTron smoking extractor**

---

## What It Does

GatorRisk reads unstructured clinical notes and automatically extracts
**7 quantitative lifestyle risk factors**, scoring each on a 0–1 risk scale.

| Risk Factor | Example Extraction |
|---|---|
| Smoking | `{status: current, ppd: 1.5, pack_years: 45}` |
| Alcohol | `{status: current, drinks_per_day: 3.0, pattern: heavy}` |
| BMI | `{value: 34.2, class: obese_I}` |
| Physical Activity | `{level: sedentary}` |
| Sleep | `{hours: 4.5, condition: OSA, osa_status: suspected}` |
| Diet | `{quality: poor, flags: [high_sodium, fast_food]}` |
| Drug Use | `{status: current, substances: [marijuana]}` |

---

## Project Structure

```
gatorrisk/
│
├── modules/                   # Core pipeline (import these)
│   ├── preprocessor.py        # Text cleaning + de-identification
│   ├── ner_extractor.py       # Rule-based NER + transformer stub
│   ├── relation_extractor.py  # Entity-value linking
│   ├── normalizer.py          # Structured output schemas
│   ├── risk_scorer.py         # 0-1 scoring + risk tiers
│   └── pipeline.py            # End-to-end orchestrator
│
├── api/
│   └── server.py              # FastAPI REST endpoint
│
├── evaluation/
│   └── evaluator.py           # Precision / Recall / F1
│
├── tests/
│   └── test_pipeline.py       # 42 unit tests
│
├── scripts/
│   ├── run_note.py            # Quick CLI for a single note
│   └── run_mtsamples.py       # Run on MTSamples Kaggle dataset
│
├── notebooks/
│   └── demo.ipynb             # Full walkthrough notebook
│
├── data/
│   ├── sample_notes.json      # 5 annotated synthetic notes
│   ├── generated_notes.json   # 200 generated training notes
│   ├── ontologies/
│   │   └── risk_terms.json    # Clinical term lists per factor
│   ├── mtsamples/             # ← place mtsamples.csv here
│   ├── raw/                   # ← raw downloaded data
│   └── processed/             # ← pipeline outputs
│
└── configs/
    └── config.yaml            # All tunable parameters
```

---

## Quickstart

```bash
# Install
pip install -r requirements.txt

# Run on a single note
python scripts/run_note.py --note "Patient smokes 1.5 ppd. BMI 34.2. Sedentary."

# Run on 200 generated notes
python modules/pipeline.py --file data/generated_notes.json

# Run on MTSamples (download CSV from Kaggle first)
python scripts/run_mtsamples.py --limit 100

# Start the API
uvicorn api.server:app --reload

# Run all tests
pytest tests/ -v

# Run evaluation
python evaluation/evaluator.py
```

---

## Data Sources

| Dataset | Notes | Access |
|---|---|---|
| `data/sample_notes.json` | 5 annotated synthetic notes | Included |
| `data/generated_notes.json` | 200 generated notes | Run `python data/generate_dataset.py` |
| MTSamples | 4,314 real transcribed notes | [Kaggle — free](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) |
| MIMIC-III | 2M+ real ICU patient notes | [PhysioNet — 1 week credentialing](https://physionet.org) |
| UF Health EHR | GatorTron training data | Via CTSI collaboration |

---

## Architecture

```
Raw Clinical Note
      ↓
[1] Preprocessor       clean text, de-identify PHI, split sentences
      ↓
[2] NER Extractor      rule-based patterns + optional transformer (GatorTron/BioBERT)
      ↓
[3] Relation Extractor link entity spans to values and quantities
      ↓
[4] Normalizer         typed output schema per risk factor
      ↓
[5] Risk Scorer        0.0–1.0 score + LOW/MODERATE/HIGH/CRITICAL tier
      ↓
[API] FastAPI Server   POST /extract for UF Health integration
```

---

## Results (200 generated notes)

| Metric | Value |
|---|---|
| Avg processing time | ~1ms/note |
| Macro F1 (annotated set) | 0.777 |
| BMI extraction F1 | 1.000 |
| Sleep extraction F1 | 1.000 |
| Drug use F1 | 0.889 |

---

## Upgrade Path (HiPerGator)

```python
# Swap in GatorTron on UF's supercomputer
pipeline = Pipeline(
    use_transformer=True,
    model_name="uf-health/gatortron-base",  # request via UFRC
)
```

---

## Contact & Collaboration

Built as an extension of the **UF CTSI NLP Core** work.

- NLP Core Director: Dr. Yonghui Wu — yonghui.wu@ufl.edu
- UF CTSI NLP Core: https://www.ctsi.ufl.edu/research/laboratory-services/nlp-core/

> ⚠️ **RESEARCH USE ONLY** — Not a certified clinical decision support system.
> All risk assessments require review by a licensed clinician.
