"""
modules/negation_detector.py
=============================
Detects negation and uncertainty contexts in clinical sentences.

Clinical notes are full of negated phrases that flip meaning:
    "no intention to quit"      → quit is NEGATED → patient still smokes
    "refuses to quit"           → quit is NEGATED → patient still smokes
    "denies alcohol use"        → alcohol is NEGATED → never drinks
    "no history of smoking"     → smoking is NEGATED → never smoked
    "not exercising"            → exercise is NEGATED → sedentary

This module tags spans in a sentence as NEGATED or AFFIRMED,
so downstream modules can correctly interpret entity meaning.

Based on NegEx algorithm principles (Chapman et al. 2001)
— simplified rule-based implementation for our 7 risk factors.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Negation Trigger Patterns
# ─────────────────────────────────────────────

# Pre-negation: words BEFORE the target that negate it
PRE_NEGATION = re.compile(
    r'\b('
    r'no\s+intention\s+to'           # "no intention to quit"
    r'|no\s+desire\s+to'             # "no desire to quit"
    r'|refuses?\s+to'                # "refuses to quit"
    r'|unwilling\s+to'               # "unwilling to quit"
    r'|will\s+not'                   # "will not quit"
    r'|does\s+not\s+want\s+to'       # "does not want to quit"
    r'|no\s+plans?\s+to'             # "no plans to quit"
    r'|not\s+(?:willing|ready)\s+to' # "not willing to quit"
    r'|denies?'                      # "denies smoking"
    r'|negative\s+for'               # "negative for tobacco"
    r'|no\s+history\s+of'            # "no history of smoking"
    r'|never'                        # "never smoked"
    r'|without'                      # "without tobacco use"
    r'|not'                          # "not smoking"
    r'|no\b'                         # "no tobacco"
    r')',
    re.I
)

# Post-negation: words AFTER the target that negate it
POST_NEGATION = re.compile(
    r'\b('
    r'denied'                        # "smoking denied"
    r'|absent'                       # "tobacco use absent"
    r'|negative'                     # "tobacco negative"
    r'|none'                         # "tobacco: none"
    r')',
    re.I
)

# Termination: words that STOP negation scope
NEGATION_TERMINATOR = re.compile(
    r'\b('
    r'but|however|except|although|though|yet|still'
    r'|and\s+(?:currently|now|still)'
    r')',
    re.I
)

# Pseudo-negation: looks like negation but isn't
# "no intention to quit" → quit is negated (patient still smokes)
# "prior to quitting"    → quitting is in the past (former)
PSEUDO_NEGATION = re.compile(
    r'\b('
    r'not\s+only'                    # "not only smokes but..."
    r'|not\s+just'
    r')',
    re.I
)


# ─────────────────────────────────────────────
# Semantic Rules for Risk Factors
# ─────────────────────────────────────────────

# Phrases where a "quitting" word is NEGATED → patient is CURRENT
NEGATED_QUIT_PATTERNS = re.compile(
    r'\b('
    r'no\s+intention\s+to\s+(?:quit|stop|cease)'
    r'|no\s+desire\s+to\s+(?:quit|stop|cease)'
    r'|refuses?\s+to\s+(?:quit|stop|cease|give\s+up)'
    r'|unwilling\s+to\s+(?:quit|stop|cease)'
    r'|will\s+not\s+(?:quit|stop|cease)'
    r'|does\s+not\s+(?:want|plan|intend)\s+to\s+(?:quit|stop)'
    r'|not\s+interested\s+in\s+(?:quitting|stopping|cessation)'
    r'|declines?\s+(?:smoking\s+)?cessation'
    r'|counseled\s+(?:on\s+)?(?:smoking\s+)?cessation\s+but\s+declines?'
    r'|prior\s+to\s+(?:quitting|stopping)'  # "prior to quitting" means they haven't yet
    r')',
    re.I
)

# Phrases where quit IS affirmed → patient is FORMER
AFFIRMED_QUIT_PATTERNS = re.compile(
    r'\b('
    r'quit\s+(?:smoking|tobacco|cigarettes?)\s+(?:in|at|about|approximately|\d)'
    r'|quit\s+(?:smoking|tobacco)\s+\d+\s+years?\s+ago'
    r'|quit\s+at\s+(?:the\s+)?age'
    r'|stopped\s+smoking'
    r'|ceased\s+smoking'
    r'|former\s+smoker'
    r'|ex[-\s]?smoker'
    r'|tobacco\s+use[,.]?\s*which\s+(?:he|she|they)\s+quit'
    r'|has\s+(?:not\s+)?smoked\s+(?:since|for\s+the\s+past)'
    r'|smoking\s+cessation\s+(?:achieved|successful|completed)'
    r')',
    re.I
)

# Drinking negation
NEGATED_DRINKING_PATTERNS = re.compile(
    r'\b('
    r'denies?\s+(?:any\s+)?(?:alcohol|drinking|etoh)'
    r'|no\s+alcohol(?:\s+use)?'
    r'|does\s+not\s+drink'
    r'|negative\s+for\s+alcohol'
    r'|alcohol(?:\s+use)?[:\s]+(?:none|no|denied|negative)'
    r'|teetotal'
    r'|abstains?\s+from\s+alcohol'
    r'|never\s+drinks?'
    r')',
    re.I
)

# Exercise negation
NEGATED_EXERCISE_PATTERNS = re.compile(
    r'\b('
    r'no\s+(?:regular\s+)?exercise'
    r'|does\s+not\s+exercise'
    r'|sedentary'
    r'|physically\s+inactive'
    r'|no\s+physical\s+activity'
    r'|difficulty\s+(?:walking|exercising)'
    r')',
    re.I
)


# ─────────────────────────────────────────────
# Data Structure
# ─────────────────────────────────────────────

@dataclass
class NegationResult:
    sentence: str
    is_quit_negated: bool = False      # "no intention to quit" → current smoker
    is_quit_affirmed: bool = False     # "quit smoking in 2018" → former smoker
    is_drinking_negated: bool = False  # "denies alcohol" → never drinker
    is_exercise_negated: bool = False  # "no exercise" → sedentary
    negation_spans: List[Tuple[int, int]] = field(default_factory=list)
    context_flags: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Negation Detector
# ─────────────────────────────────────────────

class NegationDetector:
    """
    Detects negation contexts in clinical sentences.

    Used by RelationExtractor to correctly interpret entity meaning
    when context would otherwise flip the semantic interpretation.

    Key insight:
        "smokes 2 ppd with no intention to quit" 
        → "quit" appears but is NEGATED → status = current

        "quit smoking in 2018"
        → "quit" appears and is AFFIRMED → status = former
    """

    def analyze(self, sentence: str) -> NegationResult:
        """
        Analyze a sentence for negation contexts.

        Args:
            sentence: A single clinical sentence

        Returns:
            NegationResult with boolean flags for each negation type
        """
        result = NegationResult(sentence=sentence)

        # Check quit negation (most important for smoking)
        if NEGATED_QUIT_PATTERNS.search(sentence):
            result.is_quit_negated = True
            result.context_flags.append("quit_negated")
            logger.debug(f"Quit NEGATED in: {sentence[:60]}")

        # Check quit affirmation
        if AFFIRMED_QUIT_PATTERNS.search(sentence):
            result.is_quit_affirmed = True
            result.context_flags.append("quit_affirmed")
            logger.debug(f"Quit AFFIRMED in: {sentence[:60]}")

        # Negated quit takes priority over affirmed quit
        # (e.g., "tried to quit but failed" — still current)
        if result.is_quit_negated and result.is_quit_affirmed:
            result.is_quit_affirmed = False
            result.context_flags.append("negation_wins")

        # Check drinking negation
        if NEGATED_DRINKING_PATTERNS.search(sentence):
            result.is_drinking_negated = True
            result.context_flags.append("drinking_negated")

        # Check exercise negation
        if NEGATED_EXERCISE_PATTERNS.search(sentence):
            result.is_exercise_negated = True
            result.context_flags.append("exercise_negated")

        return result

    def analyze_batch(self, sentences: List[str]) -> List[NegationResult]:
        """Analyze multiple sentences."""
        return [self.analyze(s) for s in sentences]

    def get_smoking_status(self, sentence: str, has_smoking_trigger: bool = True) -> Optional[str]:
        """
        Determine smoking status from a sentence using negation awareness.

        Args:
            sentence: Clinical sentence
            has_smoking_trigger: Whether a smoking trigger word was found (smokes, tobacco, etc.)

        Returns:
            "current" | "former" | "never" | None (unknown)
        """
        result = self.analyze(sentence)

        # Explicit never patterns
        never_pattern = re.compile(
            r'\b(never\s+smoked?|non[-\s]?smoker|nonsmoker|lifelong\s+nonsmoker'
            r'|no\s+tobacco\s+use|tobacco[:\s]+none|smokes?\s+none'
            r'|negative\s+for\s+tobacco|no\s+history\s+of\s+(smoking|tobacco)'
            r'|does\s+not\s+smoke|denies\s+(smoking|tobacco)|no\s+smoking)\b',
            re.I
        )
        if never_pattern.search(sentence):
            return "never"

        # Quit is affirmed (and not negated) → former
        if result.is_quit_affirmed and not result.is_quit_negated:
            return "former"

        # Explicit former patterns without quit
        former_pattern = re.compile(
            r'\b(former\s+smoker|ex[-\s]?smoker|stopped\s+smoking|ceased\s+smoking|used\s+to\s+smoke)\b',
            re.I
        )
        if former_pattern.search(sentence):
            return "former"

        # Has smoking trigger → current
        # (includes "no intention to quit" case — quit is negated, trigger present)
        if has_smoking_trigger:
            return "current"

        return None


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    detector = NegationDetector()

    test_cases = [
        # Should be CURRENT (quit is negated)
        "Patient smokes 2 ppd with no intention to quit.",
        "He smokes 1.5 packs per day and refuses to quit.",
        "Smokes 2 ppd — has been counseled on cessation but declines.",
        "Patient smokes 123.5 packs per day for the past 30 years, with no intention to quit.",

        # Should be FORMER (quit is affirmed)
        "Former smoker, quit smoking in 2018.",
        "History of tobacco use, which he quit at the age of 37.",
        "Stopped smoking 5 years ago.",
        "Quit smoking at age 45.",

        # Should be NEVER
        "Patient denies tobacco use. Never smoked.",
        "Non-smoker. No history of tobacco.",
        "Negative for tobacco.",

        # Edge cases
        "He tried to quit smoking but was unsuccessful — still smokes 1 ppd.",
    ]

    print("=" * 70)
    print("NEGATION DETECTOR TEST")
    print("=" * 70)

    for sentence in test_cases:
        status = detector.get_smoking_status(
            sentence,
            has_smoking_trigger=bool(re.search(r'\bsmokes?\b|\btobacco\b', sentence, re.I))
        )
        result = detector.analyze(sentence)
        flags = ", ".join(result.context_flags) if result.context_flags else "none"
        print(f"\n  Text   : {sentence[:70]}")
        print(f"  Status : {status}")
        print(f"  Flags  : {flags}")
