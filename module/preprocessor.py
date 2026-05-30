"""
Module 1: Preprocessor
======================
Handles text cleaning, sentence splitting, and lightweight de-identification
before any NLP processing.

On HiPerGator with real MIMIC data, swap the deidentify() method for
a full PhiDIA or MIST de-identification pipeline.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Fuzzy matching — import lazily so missing rapidfuzz doesn't break anything
try:
    from modules.fuzzy_matcher import correct_sentences, fuzzy_available
    _FUZZY_ENABLED = fuzzy_available()
except ImportError:
    _FUZZY_ENABLED = False


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class ProcessedNote:
    """Output of the preprocessor — clean, sentence-split clinical note."""
    note_id: str
    original_text: str
    cleaned_text: str
    sentences: List[str]
    deidentified: bool
    metadata: dict = field(default_factory=dict)
    fuzzy_corrections: list = field(default_factory=list)


# ─────────────────────────────────────────────
# PHI Patterns (lightweight, not a full de-ID)
# ─────────────────────────────────────────────

PHI_PATTERNS = {
    # Dates
    "date": r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b",
    # MRN-like numbers
    "mrn": r"\bMRN[:\s#]*\d{4,10}\b",
    # Phone numbers
    "phone": r"\b(\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4})\b",
    # SSN
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    # Doctor names (simplistic: "Dr. Smith" pattern)
    "provider_name": r"\bDr\.?\s+[A-Z][a-z]+\b",
    # Patient names after common headers
    "patient_header": r"(Patient Name|Name):\s+[A-Z][a-z]+\s+[A-Z][a-z]+",
}

PHI_REPLACEMENTS = {
    "date": "[DATE]",
    "mrn": "[MRN]",
    "phone": "[PHONE]",
    "ssn": "[SSN]",
    "provider_name": "Dr. [PROVIDER]",
    "patient_header": r"\1: [PATIENT_NAME]",
}

# Sentence boundary patterns for clinical text
# Clinical notes use newlines, periods, semicolons as sentence breaks
SENTENCE_SPLIT_PATTERN = re.compile(
    r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s|(?<=\n)'
)


# ─────────────────────────────────────────────
# Preprocessor Class
# ─────────────────────────────────────────────

class ClinicalPreprocessor:
    """
    Cleans and prepares clinical notes for downstream NLP.

    Steps:
        1. Normalize whitespace and encoding artifacts
        2. Optional: de-identify PHI patterns
        3. Split into sentences
        4. Filter out noise/empty sentences

    Usage:
        preprocessor = ClinicalPreprocessor(deidentify=True)
        processed = preprocessor.process(note_id="N001", text="Patient smokes 2 ppd...")
    """

    def __init__(self, deidentify: bool = True, min_sentence_length: int = 5, fuzzy_correct: bool = True):
        self.deidentify = deidentify
        self.min_sentence_length = min_sentence_length
        self.fuzzy_correct = fuzzy_correct and _FUZZY_ENABLED
        logger.info(f"ClinicalPreprocessor initialized | deidentify={deidentify} | fuzzy={self.fuzzy_correct}")

    # ── Public API ──────────────────────────────

    def process(self, note_id: str, text: str, metadata: Optional[dict] = None) -> ProcessedNote:
        """
        Full preprocessing pipeline for a single clinical note.

        Args:
            note_id: Unique identifier for the note
            text: Raw clinical note text
            metadata: Optional dict (patient_id, note_type, etc.)

        Returns:
            ProcessedNote with cleaned text and sentence list
        """
        logger.debug(f"Processing note: {note_id}")
        original = text

        # Step 1: Normalize encoding and whitespace
        text = self._normalize(text)

        # Step 2: De-identify (optional)
        if self.deidentify:
            text = self._deidentify(text)

        # Step 3: Sentence splitting
        sentences = self._split_sentences(text)

        # Step 4: Filter noise
        sentences = self._filter_sentences(sentences)

        # Step 5: Fuzzy spelling correction (catches typos like "smokss" → "smokes")
        fuzzy_corrections = []
        if self.fuzzy_correct:
            sentences, fuzzy_corrections = correct_sentences(sentences)
            if fuzzy_corrections:
                logger.debug(f"[{note_id}] Fuzzy corrections: {len(fuzzy_corrections)}")
                for c in fuzzy_corrections:
                    logger.debug(f"  '{c['original']}' → '{c['corrected']}' [{c['factor']}] ({c['similarity']}%)")

        return ProcessedNote(
            note_id=note_id,
            original_text=original,
            cleaned_text=text,
            sentences=sentences,
            deidentified=self.deidentify,
            metadata=metadata or {},
            fuzzy_corrections=fuzzy_corrections,
        )

    def process_batch(self, notes: List[dict]) -> List[ProcessedNote]:
        """
        Process a list of notes.

        Args:
            notes: List of dicts with keys: note_id, text, (optional) metadata

        Returns:
            List of ProcessedNote objects
        """
        results = []
        for note in notes:
            processed = self.process(
                note_id=note["note_id"],
                text=note["text"],
                metadata={k: v for k, v in note.items() if k not in ("note_id", "text")},
            )
            results.append(processed)
        logger.info(f"Batch processed {len(results)} notes")
        return results

    # ── Private Methods ──────────────────────────

    def _normalize(self, text: str) -> str:
        """Fix encoding artifacts, collapse whitespace, normalize line endings."""
        # Replace Windows line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse multiple blank lines into one
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Fix common OCR/encoding artifacts
        text = text.replace("\x00", "").replace("\ufffd", "")
        # Normalize multiple spaces
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _deidentify(self, text: str) -> str:
        """
        Lightweight PHI removal using regex patterns.

        NOTE: For production use with real patient data, replace this with
        a certified de-identification tool (PhiDIA, MIST, or Amazon Comprehend Medical).
        This is a development-grade implementation only.
        """
        for phi_type, pattern in PHI_PATTERNS.items():
            replacement = PHI_REPLACEMENTS.get(phi_type, f"[{phi_type.upper()}]")
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split clinical note into sentences.
        Clinical notes often use bullet points, numbered lists, and newlines.
        """
        sentences = []

        # First split on newlines (common in clinical notes)
        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Further split long lines on sentence boundaries
            # Avoid splitting on common abbreviations (e.g., "Dr.", "approx.", "pt.")
            # Split on ". Capital" but avoid splitting on common abbreviations
            parts = re.split(r'(?<=\.)\s+(?=[A-Z])', line)
            # Re-join parts that were incorrectly split on abbreviations
            ABBREVS = {"Dr", "Mr", "Mrs", "Ms", "approx", "pt", "lbs", "kg",
                       "hrs", "appt", "wt", "ht", "Hx", "Rx", "dx", "Dx"}
            merged = []
            i = 0
            while i < len(parts):
                part = parts[i]
                last_word = part.rstrip(".").split()[-1] if part.split() else ""
                if last_word in ABBREVS and i + 1 < len(parts):
                    parts[i + 1] = part + " " + parts[i + 1]
                else:
                    merged.append(part)
                i += 1
            sub_sentences = merged
            sentences.extend(sub_sentences)

        return sentences

    def _filter_sentences(self, sentences: List[str]) -> List[str]:
        """Remove very short, blank, or header-only lines."""
        filtered = []
        for s in sentences:
            s = s.strip()
            # Skip if too short
            if len(s) < self.min_sentence_length:
                continue
            # Skip if it's just a header like "SOCIAL HISTORY:" with no content
            if re.match(r'^[A-Z\s/]+:?\s*$', s):
                continue
            filtered.append(s)
        return filtered


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample = """
    Patient Name: John Doe   MRN: 1234567
    Dr. Smith saw the patient on 03/15/2024.

    SOCIAL HISTORY:
    Patient smokes 1.5 packs per day for the past 30 years.
    Drinks 3 beers nightly. BMI 34.2, obese class I.
    Sedentary lifestyle with no regular exercise.
    Sleeps 4-5 hours; loud snoring noted, OSA suspected.
    Diet: high sodium, frequent fast food. Denies drug use.
    """

    preprocessor = ClinicalPreprocessor(deidentify=True)
    result = preprocessor.process(note_id="TEST_001", text=sample)

    print("=" * 60)
    print(f"Note ID     : {result.note_id}")
    print(f"De-ID'd     : {result.deidentified}")
    print(f"Cleaned Text:\n{result.cleaned_text}")
    print(f"\nSentences ({len(result.sentences)}):")
    for i, s in enumerate(result.sentences, 1):
        print(f"  [{i}] {s}")
