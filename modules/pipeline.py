"""
Module 6: Pipeline Orchestrator
================================
End-to-end runner — wires all five modules together.

Usage:
    # Single note
    python modules/pipeline.py --note "Patient smokes 2 ppd..."

    # From JSON file
    python modules/pipeline.py --file data/sample_notes.json

    # Programmatic
    from modules.pipeline import Pipeline
    pipeline = Pipeline()
    result = pipeline.run_note(note_id="N001", text="...")
"""

import json
import logging
import argparse
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path

from modules.preprocessor import ClinicalPreprocessor, ProcessedNote
from modules.ner_extractor import NERExtractor, NERResult
from modules.relation_extractor import RelationExtractor, RelationResult
from modules.normalizer import Normalizer, NormalizedPatientProfile
from modules.risk_scorer import RiskScorer, RiskProfile

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Pipeline Output
# ─────────────────────────────────────────────

@dataclass
class PipelineOutput:
    """Complete output for a single note — all pipeline stages."""
    note_id: str
    processing_time_ms: float
    processed_note: ProcessedNote
    ner_result: NERResult
    relation_result: RelationResult
    normalized_profile: NormalizedPatientProfile
    risk_profile: RiskProfile
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "note_id": self.note_id,
            "processing_time_ms": self.processing_time_ms,
            "sentences": self.processed_note.sentences,
            "entities_found": self.ner_result.summary(),
            "normalized_profile": self.normalized_profile.to_dict(),
            "risk_profile": self.risk_profile.to_dict(),
            "errors": self.errors,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ─────────────────────────────────────────────
# Pipeline Class
# ─────────────────────────────────────────────

class Pipeline:
    """
    End-to-end Lifestyle Risk Factor Extraction Pipeline.

    Modules:
        1. ClinicalPreprocessor  — clean + sentence-split
        2. NERExtractor          — entity recognition
        3. RelationExtractor     — entity-value linking
        4. Normalizer            — structured schema output
        5. RiskScorer            — quantitative risk assessment

    Args:
        use_transformer:  Enable transformer-based NER (requires GPU for speed)
        model_name:       HuggingFace model ID for NER
        deidentify:       Apply lightweight PHI removal
        confidence_threshold: Min confidence for transformer entities
        weights:          Dict of factor weights for risk scoring
    """

    def __init__(
        self,
        use_transformer: bool = False,
        model_name: str = "d4data/biomedical-ner-all",
        deidentify: bool = True,
        confidence_threshold: float = 0.65,
        weights: Optional[Dict[str, float]] = None,
    ):
        logger.info("Initializing pipeline...")

        self.preprocessor = ClinicalPreprocessor(deidentify=deidentify)
        self.ner_extractor = NERExtractor(
            use_transformer=use_transformer,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
        )
        self.relation_extractor = RelationExtractor()
        self.normalizer = Normalizer()
        self.risk_scorer = RiskScorer(weights=weights)

        logger.info(
            f"Pipeline ready | transformer={'ON' if use_transformer else 'OFF (rule-based)'} | "
            f"deidentify={deidentify}"
        )

    # ── Public API ──────────────────────────────

    def run_note(self, note_id: str, text: str, metadata: Optional[dict] = None) -> PipelineOutput:
        """
        Run the full pipeline on a single clinical note.

        Args:
            note_id:  Unique identifier for the note
            text:     Raw clinical note text
            metadata: Optional dict (patient_id, note_type, etc.)

        Returns:
            PipelineOutput with all intermediate and final results
        """
        start = time.time()
        errors = []

        try:
            # Stage 1: Preprocess
            processed = self.preprocessor.process(note_id, text, metadata)
            logger.debug(f"[{note_id}] Preprocessed: {len(processed.sentences)} sentences")

            # Stage 2: NER
            ner_result = self.ner_extractor.extract(note_id, processed.sentences)
            logger.debug(f"[{note_id}] NER: {ner_result.summary()}")

            # Stage 3: Relation extraction
            rel_result = self.relation_extractor.extract(note_id, ner_result)
            logger.debug(f"[{note_id}] Relations: {len(rel_result.relations)}")

            # Stage 4: Normalize
            profile = self.normalizer.normalize(rel_result)

            # Stage 5: Risk score
            risk = self.risk_scorer.score(profile)
            logger.info(f"[{note_id}] ✓ Risk: {risk.composite_score:.2f} ({risk.composite_tier})")

        except Exception as e:
            logger.error(f"[{note_id}] Pipeline error: {e}", exc_info=True)
            errors.append(str(e))
            # Return partial result on error
            processed = processed if 'processed' in dir() else self.preprocessor.process(note_id, text)
            ner_result = ner_result if 'ner_result' in dir() else NERResult(note_id=note_id)
            rel_result = rel_result if 'rel_result' in dir() else RelationResult(note_id=note_id)
            from modules.normalizer import NormalizedPatientProfile
            profile = profile if 'profile' in dir() else NormalizedPatientProfile(note_id=note_id)
            from modules.risk_scorer import RiskProfile
            risk = risk if 'risk' in dir() else RiskProfile(note_id=note_id, composite_score=0.0, composite_tier="UNKNOWN")

        elapsed = round((time.time() - start) * 1000, 2)

        return PipelineOutput(
            note_id=note_id,
            processing_time_ms=elapsed,
            processed_note=processed,
            ner_result=ner_result,
            relation_result=rel_result,
            normalized_profile=profile,
            risk_profile=risk,
            errors=errors,
        )

    def run_batch(self, notes: List[dict]) -> List[PipelineOutput]:
        """
        Run pipeline on a list of notes.

        Args:
            notes: List of dicts with keys: note_id, text, (optional) metadata

        Returns:
            List of PipelineOutput objects
        """
        results = []
        logger.info(f"Running batch pipeline on {len(notes)} notes...")

        for i, note in enumerate(notes):
            note_id = note.get("note_id", f"NOTE_{i:04d}")
            text = note.get("text", "")
            metadata = {k: v for k, v in note.items() if k not in ("note_id", "text")}
            output = self.run_note(note_id=note_id, text=text, metadata=metadata)
            results.append(output)

        total_ms = sum(r.processing_time_ms for r in results)
        logger.info(f"Batch complete: {len(results)} notes in {total_ms:.0f}ms "
                    f"(avg {total_ms/len(results):.0f}ms/note)")
        return results

    def run_from_file(self, filepath: str) -> List[PipelineOutput]:
        """Load notes from JSON file and run batch pipeline."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Notes file not found: {filepath}")

        with open(path) as f:
            notes = json.load(f)

        logger.info(f"Loaded {len(notes)} notes from {filepath}")
        return self.run_batch(notes)


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────

def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Lifestyle Risk Factor Extractor — Clinical NLP Pipeline"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--note", type=str, help="Raw clinical note text (quoted string)")
    source.add_argument("--file", type=str, help="Path to JSON file of notes")

    parser.add_argument("--note-id", default="CLI_NOTE_001", help="Note ID (for --note mode)")
    parser.add_argument("--transformer", action="store_true", help="Enable transformer NER (slower)")
    parser.add_argument("--no-deidentify", action="store_true", help="Skip PHI de-identification")
    parser.add_argument("--output", type=str, help="Save JSON output to file")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    _setup_logging(args.verbose)

    pipeline = Pipeline(
        use_transformer=args.transformer,
        deidentify=not args.no_deidentify,
    )

    if args.note:
        result = pipeline.run_note(note_id=args.note_id, text=args.note)
        results = [result]
        print("\n" + "=" * 60)
        print(result.risk_profile.summary())
        print("\n--- NORMALIZED PROFILE ---")
        print(json.dumps(result.normalized_profile.to_dict(), indent=2))

    else:
        results = pipeline.run_from_file(args.file)
        print(f"\nProcessed {len(results)} notes:\n")
        for r in results:
            print(r.risk_profile.summary())
            print()

    if args.output:
        output_data = [r.to_dict() for r in results]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
