"""
modules/fuzzy_matcher.py
========================
Fuzzy matching layer for the NER extractor.

Catches spelling mistakes common in clinical notes:
    "smokss"     → "smokes"    (smoking)
    "alcohool"   → "alcohol"   (alcohol)
    "obses"      → "obese"     (bmi)
    "sedantary"  → "sedentary" (physical_activity)
    "insomina"   → "insomnia"  (sleep)
    "marhijuana" → "marijuana" (drug_use)

Uses rapidfuzz for fast Levenshtein distance matching.
Install: pip install rapidfuzz

Usage:
    from modules.fuzzy_matcher import correct_sentence, fuzzy_available
    corrected, corrections = correct_sentence("pt smokss 2 ppd")
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Keyword Dictionary
# ─────────────────────────────────────────────

FUZZY_KEYWORDS = {
    "smoking": [
        "smokes", "smoking", "smoker", "tobacco", "cigarette",
        "cigarettes", "nicotine", "nonsmoker", "nonsmoking",
        "vaping", "ecig",
    ],
    "alcohol": [
        "alcohol", "alcoholic", "alcoholism", "drinking",
        "drinks", "etoh", "beer", "wine", "liquor", "spirits",
        "sobriety", "sober",
    ],
    "bmi": [
        "obesity", "obese", "overweight", "underweight",
        "morbid", "weight", "bmi",
    ],
    "physical_activity": [
        "sedentary", "exercise", "exercises", "exercising",
        "active", "inactive", "walking", "running", "jogging",
        "cardio", "aerobic", "workout", "gym",
    ],
    "sleep": [
        "insomnia", "apnea", "snoring", "sleeps", "sleeping",
        "cpap", "bipap", "fatigue", "narcolepsy",
    ],
    "diet": [
        "diet", "nutrition", "sodium", "calories",
        "eating", "appetite", "nutritional",
    ],
    "drug_use": [
        "marijuana", "cocaine", "heroin", "cannabis",
        "ivdu", "methamphetamine", "opioid", "fentanyl",
    ],
}

# Flat list + reverse lookup
ALL_KEYWORDS = []
KEYWORD_TO_FACTOR = {}
for _factor, _kws in FUZZY_KEYWORDS.items():
    for _kw in _kws:
        ALL_KEYWORDS.append(_kw)
        KEYWORD_TO_FACTOR[_kw] = _factor

# Similarity threshold (0-100). 85 = catches most 1-2 char typos
DEFAULT_THRESHOLD = 85


# ─────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────

def fuzzy_available() -> bool:
    """Check if rapidfuzz is installed."""
    try:
        import rapidfuzz
        return True
    except ImportError:
        return False


def correct_sentence(
    sentence: str,
    threshold: int = DEFAULT_THRESHOLD
) -> Tuple[str, List[Tuple[str, str, str, int]]]:
    """
    Scan a sentence for misspelled lifestyle keywords.

    Args:
        sentence:  Raw clinical sentence
        threshold: Minimum similarity % to accept a correction (default 85)

    Returns:
        (corrected_sentence, corrections)
        corrections = list of (original, corrected, factor, similarity%)

    Examples:
        "pt smokss 2 ppd"
        → ("pt smokes 2 ppd", [("smokss", "smokes", "smoking", 91)])

        "morbidly obses, alcohool use"
        → ("morbidly obese, alcohol use", [...])

        "patient exercises daily"
        → ("patient exercises daily", [])  # no correction needed
    """
    if not fuzzy_available():
        logger.debug("rapidfuzz not installed — fuzzy matching skipped")
        return sentence, []

    from rapidfuzz import fuzz, process

    words = sentence.split()
    corrected_words = list(words)
    corrections = []

    for i, word in enumerate(words):
        # Strip punctuation for matching, preserve for output
        clean = re.sub(r'[^a-zA-Z]', '', word).lower()
        punct_suffix = word[len(re.match(r'[a-zA-Z]*', word).group()):]

        # Skip words too short to match reliably
        if len(clean) < 4:
            continue

        # Skip if already an exact keyword match
        if clean in KEYWORD_TO_FACTOR:
            continue

        # Fuzzy match against all keywords
        result = process.extractOne(clean, ALL_KEYWORDS, scorer=fuzz.ratio)

        if result and result[1] >= threshold:
            best_match, similarity, _ = result
            factor = KEYWORD_TO_FACTOR[best_match]

            corrected_words[i] = best_match + punct_suffix
            corrections.append((word, best_match, factor, similarity))

            logger.debug(
                f"  Fuzzy: '{word}' → '{best_match}' "
                f"[{factor}] ({similarity}% match)"
            )

    corrected = " ".join(corrected_words)
    return corrected, corrections


def correct_sentences(
    sentences: List[str],
    threshold: int = DEFAULT_THRESHOLD
) -> Tuple[List[str], List[dict]]:
    """
    Run fuzzy correction over a list of sentences.

    Returns:
        (corrected_sentences, all_corrections)
        all_corrections = list of dicts with sentence index + correction details
    """
    corrected_sentences = []
    all_corrections = []

    for i, sentence in enumerate(sentences):
        corrected, corrections = correct_sentence(sentence, threshold)
        corrected_sentences.append(corrected)

        for orig, fixed, factor, sim in corrections:
            all_corrections.append({
                "sentence_index": i,
                "original": orig,
                "corrected": fixed,
                "factor": factor,
                "similarity": sim,
                "sentence_before": sentence,
                "sentence_after": corrected,
            })

    return corrected_sentences, all_corrections


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    test_sentences = [
        # Spelling mistakes
        "Patient smokss 1.5 ppd for 30 years.",
        "Morbidly obses with BMI 42.",
        "Drinks alcohool nightly.",
        "Sedantary lifestyle, no excercise.",
        "Reports insomina and loud snoring.",
        "Denies use of marhijuana or cocain.",
        # Correct spelling — should not be changed
        "Patient smokes 2 packs per day.",
        "BMI 34.2, class I obesity.",
        "Exercises 3 days per week.",
    ]

    if not fuzzy_available():
        print("rapidfuzz not installed. Run: pip install rapidfuzz")
    else:
        print("=" * 60)
        print("FUZZY MATCHING TEST")
        print("=" * 60)
        for sentence in test_sentences:
            corrected, corrections = correct_sentence(sentence)
            if corrections:
                print(f"\n  BEFORE: {sentence}")
                print(f"  AFTER:  {corrected}")
                for orig, fixed, factor, sim in corrections:
                    print(f"    '{orig}' → '{fixed}' [{factor}] ({sim}% match)")
            else:
                print(f"\n  OK (no changes): {sentence}")
