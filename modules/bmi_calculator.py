"""
modules/bmi_calculator.py
==========================
Extracts height and weight from clinical note sentences and computes BMI.

Handles all real-world clinical note formats found in MTSamples / MIMIC:

  Height formats:
    "5 feet 8 inches"     → 68 inches
    "5'8"  / 5'8"         → 68 inches
    "height 5 feet 8"     → 68 inches
    "Ht: 68 inches"       → 68 inches
    "168 cm"              → 66.1 inches

  Weight formats:
    "weight 159 pounds"   → 159 lbs
    "weighs 232 lbs"      → 232 lbs
    "WT: 223 pounds"      → 223 lbs
    "66.5 kg"             → 146.6 lbs
    "Wt: 85 kg"           → 187.4 lbs

  BMI formula:
    BMI = (weight_lbs / height_inches²) × 703

  Then classified:
    < 18.5  → underweight
    18.5–25 → normal
    25–30   → overweight
    30–35   → obese_I
    35–40   → obese_II
    ≥ 40    → obese_III

Usage:
    from modules.bmi_calculator import BMICalculator
    calc = BMICalculator()
    result = calc.extract_and_compute("Weight 159 pounds, height 5 feet 4 inches.")
    # → {"bmi": 27.3, "class": "overweight", "weight_lbs": 159, "height_inches": 64}
"""

import re
import logging
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class BMIExtraction:
    bmi: Optional[float] = None
    bmi_class: Optional[str] = None
    weight_lbs: Optional[float] = None
    height_inches: Optional[float] = None
    weight_raw: Optional[str] = None
    height_raw: Optional[str] = None
    computed_from: Optional[str] = None  # "height_weight" | "weight_only_estimated"


# ─────────────────────────────────────────────
# BMI Calculator
# ─────────────────────────────────────────────

class BMICalculator:
    """
    Extracts height and weight from clinical text and computes BMI.

    Strategy:
      1. Try to extract BOTH height and weight → compute exact BMI
      2. If only weight found → estimate BMI using average US adult height
         (5'9" for men = 69in, 5'4" for women = 64in → use 66.5in as neutral)
      3. If BMI value already in text → use that directly (handled by NER extractor)
    """

    # Average adult height used when height not mentioned (66.5 inches ≈ 5'6.5")
    DEFAULT_HEIGHT_INCHES = 66.5

    # ── Height Patterns ────────────────────────

    HEIGHT_PATTERNS = [
        # "5 feet 8 inches" / "5 feet 8"
        re.compile(r'\b(\d)\s*feet?\s*(\d{1,2})(?:\s*inch(?:es)?)?\b', re.I),
        # "5'8" / 5'8" / 5′8
        re.compile(r"\b(\d)['\u2019\u2018′`]\s*(\d{1,2})[\"″]?\b"),
        # "height 5 feet 8" / "Ht: 68 inches" / "Ht 68"
        re.compile(r'\b(?:height|ht)[:\s]+(\d{1,3})\s*(inches?|in|feet?|cm)?\b', re.I),
        # "68 inches tall"
        re.compile(r'\b(\d{2,3})\s*inch(?:es)?\s*(?:tall)?\b', re.I),
        # "168 cm" / "170cm"
        re.compile(r'\b(\d{2,3})\s*cm\b', re.I),
    ]

    # ── Weight Patterns ────────────────────────

    WEIGHT_PATTERNS = [
        # "weight 159 pounds" / "weight is 159 lbs" / "weight of 159"
        re.compile(r'\bweight(?:\s+(?:is|was|of))?\s+(\d{2,3}\.?\d*)\s*(lbs?|pounds?|kg|kilograms?)?\b', re.I),
        # "weighs 232 lbs" / "weighs 85 kg"
        re.compile(r'\bweighs?\s+(\d{2,3}\.?\d*)\s*(lbs?|pounds?|kg|kilograms?)\b', re.I),
        # "WT: 223 pounds" / "Wt 85 kg"
        re.compile(r'\bwt[:\s]+(\d{2,3}\.?\d*)\s*(lbs?|pounds?|kg|kilograms?)?\b', re.I),
        # "159 pounds" / "232 lbs" / "85 kg" (standalone)
        re.compile(r'\b(\d{2,3}\.?\d*)\s*(lbs?|pounds?|kg)\b', re.I),
        # "up three pounds" / "lost 10 pounds" — exclude these
    ]

    # Patterns to EXCLUDE (false positives)
    WEIGHT_EXCLUDE = re.compile(
        r'\b(lost?|gained?|lose|gain|up|down|lost\s+about)\s+\d+\s*(lbs?|pounds?|kg)',
        re.I
    )

    def extract_and_compute(self, text: str) -> Optional[BMIExtraction]:
        """
        Main method — extract height/weight from text and compute BMI.

        Args:
            text: A clinical note or sentence

        Returns:
            BMIExtraction or None if no weight found
        """
        weight_lbs = self._extract_weight(text)
        if weight_lbs is None:
            return None

        height_inches = self._extract_height(text)

        result = BMIExtraction(
            weight_lbs=weight_lbs,
            height_inches=height_inches,
        )

        if height_inches:
            bmi = self._compute_bmi(weight_lbs, height_inches)
            result.computed_from = "height_weight"
            result.weight_raw = f"{weight_lbs} lbs"
            result.height_raw = f"{height_inches} inches"
            if bmi:
                result.bmi = bmi
                result.bmi_class = self._classify_bmi(bmi)
        else:
            # Weight only — only classify if extreme enough to be unambiguous
            result.computed_from = "weight_only_estimated"
            result.weight_raw = f"{weight_lbs} lbs"
            if weight_lbs >= 280:
                result.bmi_class = "obese_III" if weight_lbs >= 350 else "obese_II"
                result.bmi = self._compute_bmi(weight_lbs, self.DEFAULT_HEIGHT_INCHES)
            elif weight_lbs >= 220:
                result.bmi_class = "obese_I"
                result.bmi = self._compute_bmi(weight_lbs, self.DEFAULT_HEIGHT_INCHES)
            elif weight_lbs <= 100:
                result.bmi_class = "underweight"
                result.bmi = self._compute_bmi(weight_lbs, self.DEFAULT_HEIGHT_INCHES)
            else:
                # Ambiguous weight range — report weight but no BMI class
                result.bmi = None
                result.bmi_class = None
            logger.debug(f"Weight only: {weight_lbs} lbs → class={result.bmi_class}")

        return result

    def extract_from_sentences(self, sentences: List[str]) -> Optional[BMIExtraction]:
        """
        Run extraction across multiple sentences.
        Tries each sentence and combines height from one with weight from another.
        """
        all_weights = []
        all_heights = []

        for sentence in sentences:
            w = self._extract_weight(sentence)
            h = self._extract_height(sentence)
            if w:
                all_weights.append(w)
            if h:
                all_heights.append(h)

        if not all_weights:
            return None

        # Use median weight (avoid outlier values like "132/73" blood pressure)
        # Filter to plausible adult weights: 70–700 lbs
        valid_weights = [w for w in all_weights if 70 <= w <= 700]
        if not valid_weights:
            return None

        # Use the most common weight if multiple found
        weight_lbs = sorted(valid_weights)[len(valid_weights) // 2]

        # Filter to plausible adult heights: 48–84 inches (4ft–7ft)
        valid_heights = [h for h in all_heights if 48 <= h <= 84]
        height_inches = valid_heights[0] if valid_heights else None

        result = BMIExtraction(weight_lbs=weight_lbs, height_inches=height_inches)

        if height_inches:
            result.bmi = self._compute_bmi(weight_lbs, height_inches)
            result.computed_from = "height_weight"
            if result.bmi:
                result.bmi_class = self._classify_bmi(result.bmi)
        else:
            # Weight only — compute estimated BMI but only classify if clearly obese or underweight
            # A 300lb person is obese regardless of height. A 120lb person is not necessarily underweight.
            # Only commit to a class when weight alone is unambiguous.
            estimated_bmi = self._compute_bmi(weight_lbs, self.DEFAULT_HEIGHT_INCHES)
            result.bmi = estimated_bmi
            result.computed_from = "weight_only_estimated"
            if estimated_bmi:
                # Only assign class if the weight is extreme enough to be unambiguous
                if weight_lbs >= 280:
                    result.bmi_class = "obese_III" if weight_lbs >= 350 else "obese_II"
                elif weight_lbs >= 220:
                    result.bmi_class = "obese_I"
                elif weight_lbs <= 100:
                    result.bmi_class = "underweight"
                else:
                    # Ambiguous — don't guess the class, just report weight
                    result.bmi_class = None
                    result.bmi = None  # don't report unreliable BMI number

        return result

    # ── Private Methods ────────────────────────

    def _extract_weight(self, text: str) -> Optional[float]:
        """Extract weight in lbs from text."""
        # Skip if this sentence is about weight change
        if self.WEIGHT_EXCLUDE.search(text):
            return None

        for pattern in self.WEIGHT_PATTERNS:
            m = pattern.search(text)
            if m:
                try:
                    value = float(m.group(1))
                    unit = m.group(2).lower() if len(m.groups()) >= 2 and m.group(2) else "lbs"

                    # Convert kg → lbs
                    if unit in ('kg', 'kilograms', 'kilogram'):
                        value = round(value * 2.205, 1)

                    # Sanity check — adult weight range
                    if 70 <= value <= 700:
                        return value
                except (IndexError, ValueError, AttributeError):
                    continue
        return None

    def _extract_height(self, text: str) -> Optional[float]:
        """Extract height in inches from text."""
        # Pattern 1: "5 feet 8 inches" / "5 feet 8"
        m = re.search(r'\b(\d)\s*feet?\s*(\d{1,2})(?:\s*inch(?:es)?)?\b', text, re.I)
        if m:
            feet = int(m.group(1))
            inches = int(m.group(2))
            total = feet * 12 + inches
            if 48 <= total <= 84:
                return float(total)

        # Pattern 2: "5'8" / 5'8"
        m = re.search(r"\b(\d)['\u2019\u2018′`]\s*(\d{1,2})[\"″]?\b", text)
        if m:
            feet = int(m.group(1))
            inches = int(m.group(2))
            total = feet * 12 + inches
            if 48 <= total <= 84:
                return float(total)

        # Pattern 3: "height 68 inches" / "Ht: 68"
        m = re.search(r'\b(?:height|ht)[:\s]+(\d{2,3})\s*(inches?|in)?\b', text, re.I)
        if m:
            val = float(m.group(1))
            if 48 <= val <= 84:
                return val

        # Pattern 4: "168 cm" → convert to inches
        m = re.search(r'\b(1[4-9]\d|2[0-2]\d)\s*cm\b', text, re.I)
        if m:
            cm = float(m.group(1))
            inches = round(cm / 2.54, 1)
            if 48 <= inches <= 84:
                return inches

        return None

    def _compute_bmi(self, weight_lbs: float, height_inches: float) -> Optional[float]:
        """BMI = (weight_lbs / height_inches²) × 703"""
        if height_inches <= 0:
            return None
        bmi = (weight_lbs / (height_inches ** 2)) * 703
        return round(bmi, 1)

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
    logging.basicConfig(level=logging.DEBUG)
    calc = BMICalculator()

    test_cases = [
        # height + weight → exact BMI
        ("Weight 232 pounds, height 5 feet 8 inches.", True),
        ("She is 5'7-1/2 tall, 148 pounds.", True),
        ("Weight 66.5 kg. Height 168 cm.", True),
        ("WT: 223 pounds. Ht: 68 inches.", True),
        # weight only → estimated BMI
        ("Weight 159 pounds.", False),
        ("Patient weighs 85 kg.", False),
        ("Wt 250 pounds.", False),
        # should NOT extract (weight change context)
        ("She lost 10 pounds last month.", False),
        ("He gained 3 pounds.", False),
        # no weight
        ("Patient is alert and oriented.", False),
    ]

    print("=" * 65)
    print("BMI CALCULATOR TEST")
    print("=" * 65)
    for text, has_height in test_cases:
        result = calc.extract_and_compute(text)
        if result:
            exact = "✓ exact" if result.computed_from == "height_weight" else "~ estimated"
            print(f"\n  Input  : {text}")
            print(f"  Weight : {result.weight_lbs} lbs")
            print(f"  Height : {result.height_inches} in" if result.height_inches else f"  Height : not found")
            print(f"  BMI    : {result.bmi} → {result.bmi_class} ({exact})")
        else:
            print(f"\n  Input  : {text}")
            print(f"  Result : no weight found (correct)" if "lost" in text or "gained" in text or "alert" in text else f"  Result : nothing extracted")
