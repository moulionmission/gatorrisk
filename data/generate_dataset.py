"""
data/generate_dataset.py
========================
Generates a large, realistic synthetic clinical note dataset
for development and testing — no credentialing required.

Notes are modeled after real MTSamples / MIMIC discharge summary
"Social History" sections, with natural language variation.

Run:
    python data/generate_dataset.py --n 200 --output data/generated_notes.json

On your own machine, swap this with real data:
    → MTSamples (free, no login): https://mtsamples.com  (3,000+ real notes)
    → MIMIC-III (free, 1 week credentialing): physionet.org
    → n2c2 datasets: portal.dbmi.hms.harvard.edu
"""

import json
import random
import argparse
from pathlib import Path
from itertools import product

random.seed(42)

# ─────────────────────────────────────────────
# Building Blocks — Natural Language Variation
# ─────────────────────────────────────────────

# Each list = different ways to say the same thing

SMOKING = {
    "current": [
        "Patient smokes {ppd} packs per day.",
        "He is a current smoker, approximately {ppd} ppd.",
        "Active smoker — {cig} cigarettes daily.",
        "Smokes {ppd} packs a day, has smoked for {years} years.",
        "Continues to smoke {ppd} packs per day despite counseling.",
        "{ppd}-pack-per-day smoker with a {py} pack-year history.",
        "Patient reports smoking {cig} cigarettes per day.",
        "Tobacco use: smokes {ppd} ppd, {py} pack-years total.",
        "She smokes {ppd} packs daily; started at age {start_age}.",
        "Heavy smoker: {ppd} packs per day for the past {years} years.",
    ],
    "former": [
        "Former smoker, quit {years_ago} years ago.",
        "Ex-smoker — stopped smoking in {quit_year}.",
        "Smoked {ppd} ppd for {years} years, quit {years_ago} years ago.",
        "Previously smoked; cessation {years_ago} years ago.",
        "{py} pack-year history of smoking; has not smoked since {quit_year}.",
        "Quit smoking {years_ago} years ago after {years} years of use.",
        "Former 1 ppd smoker, now smoke-free for {years_ago} years.",
        "Used to smoke but stopped in {quit_year}.",
    ],
    "never": [
        "Denies tobacco use.",
        "Non-smoker.",
        "Never smoked.",
        "No tobacco history.",
        "Patient is a lifelong non-smoker.",
        "Denies ever smoking cigarettes or using tobacco products.",
        "Never-smoker.",
        "No history of tobacco or nicotine use.",
    ],
}

ALCOHOL = {
    "current_light": [
        "Social drinker — occasional glass of wine on weekends.",
        "Drinks socially, approximately 1-2 drinks per week.",
        "Occasional alcohol use, 1-2 drinks on weekends.",
        "Light drinker; 1 beer occasionally.",
        "Reports drinking {drinks} drinks per week socially.",
        "Drinks wine socially, rarely more than {drinks} glasses per week.",
    ],
    "current_moderate": [
        "Drinks {drinks} beers per day.",
        "Reports {drinks} glasses of wine nightly.",
        "Alcohol use: approximately {drinks} drinks per day.",
        "Consumes {drinks} drinks daily, primarily beer.",
        "Drinks {drinks} alcoholic beverages per day.",
        "ETOH: {drinks} drinks nightly.",
        "Drinks {drinks_week} drinks per week.",
    ],
    "current_heavy": [
        "Heavy drinker — approximately {drinks} drinks per day.",
        "Reports drinking {drinks} beers nightly.",
        "Alcohol abuse history; currently drinking {drinks} drinks per day.",
        "Heavy ETOH use: {drinks} drinks daily.",
        "Binge drinking pattern; up to {drinks} drinks per episode, several times weekly.",
    ],
    "former": [
        "Former alcohol user, now sober for {years} years.",
        "History of alcohol abuse, in recovery for {years} years.",
        "Quit drinking {years} years ago.",
        "Former heavy drinker, achieved sobriety in {quit_year}.",
        "Abstains from alcohol; has been sober since {quit_year}.",
    ],
    "never": [
        "Denies alcohol use.",
        "Does not drink alcohol.",
        "Teetotal.",
        "No alcohol use.",
        "Abstains from alcohol.",
        "Denies any ETOH use.",
        "Never drinks alcohol.",
    ],
}

BMI = {
    "underweight": [
        "BMI {bmi}, underweight.",
        "Body mass index {bmi} — underweight.",
        "Patient appears underweight; BMI {bmi}.",
        "BMI of {bmi}, below normal range.",
    ],
    "normal": [
        "BMI {bmi}.",
        "BMI {bmi}, within normal limits.",
        "Body mass index {bmi} — normal weight.",
        "BMI is {bmi}, appropriate for height.",
        "Weight appropriate; BMI {bmi}.",
    ],
    "overweight": [
        "BMI {bmi}, overweight.",
        "Body mass index {bmi} — overweight.",
        "Patient is overweight with a BMI of {bmi}.",
        "BMI {bmi}; advised on weight management.",
    ],
    "obese_I": [
        "BMI {bmi}, class I obesity.",
        "BMI of {bmi}, consistent with class I obesity.",
        "Obese — BMI {bmi}.",
        "Class I obese; BMI {bmi}.",
        "BMI {bmi}; patient meets criteria for class I obesity.",
    ],
    "obese_II": [
        "BMI {bmi}, class II obesity.",
        "Morbidly overweight; BMI {bmi}.",
        "BMI of {bmi}, class II obese.",
        "Class II obesity with BMI {bmi}.",
    ],
    "obese_III": [
        "BMI {bmi}, morbidly obese.",
        "Morbid obesity — BMI {bmi}.",
        "BMI of {bmi}, class III obesity.",
        "Patient is morbidly obese; BMI {bmi}.",
        "Extreme obesity; BMI {bmi}.",
    ],
}

ACTIVITY = {
    "sedentary": [
        "Sedentary lifestyle with no regular exercise.",
        "Does not exercise regularly.",
        "Patient is physically inactive.",
        "No regular physical activity.",
        "Sedentary — works a desk job, no exercise routine.",
        "Patient denies regular exercise.",
        "Physically inactive; does not engage in regular exercise.",
    ],
    "low": [
        "Minimal physical activity — occasional walking.",
        "Light activity only; walks short distances.",
        "Rarely exercises; occasional leisure walks.",
        "Low activity level; walks occasionally.",
    ],
    "moderate": [
        "Exercises {days} days per week, {mins} minutes each session.",
        "Walks {days} times per week for {mins} minutes.",
        "Moderate activity — walks daily.",
        "Exercises regularly, {days} days per week.",
        "Engages in light exercise {days} days per week, approximately {mins} minutes.",
        "Active — walks {miles} miles per day.",
    ],
    "high": [
        "Runs {miles} miles {days} days per week.",
        "Very active — exercises {days} days per week, including running and weightlifting.",
        "Athlete; trains {days} times per week.",
        "Highly active lifestyle; {days} workouts per week averaging {mins} minutes.",
        "Active — runs {miles} miles daily and lifts weights {days} times per week.",
        "Regular vigorous exercise: {days} days per week.",
    ],
}

SLEEP = {
    "adequate": [
        "Sleeps {hrs} hours per night without difficulty.",
        "Reports {hrs} hours of sleep nightly.",
        "Sleep: {hrs} hours, no complaints.",
        "Adequate sleep — {hrs} hours per night.",
        "No sleep disturbances; sleeps {hrs} hours nightly.",
    ],
    "insufficient": [
        "Sleeps only {hrs} hours per night.",
        "Reports {hrs} hours of sleep — insufficient.",
        "Sleep: {hrs} hours nightly; reports fatigue.",
        "Poor sleep; approximately {hrs} hours per night.",
        "Difficulty maintaining sleep; gets only {hrs} hours.",
    ],
    "osa_suspected": [
        "Loud snoring reported by partner; OSA suspected.",
        "Reports loud snoring and daytime fatigue; OSA is suspected.",
        "Possible OSA — snores loudly, wakes feeling unrefreshed.",
        "Snoring and witnessed apneas per partner; OSA likely.",
        "Sleeps {hrs} hours but reports non-restorative sleep; OSA suspected.",
    ],
    "osa_confirmed": [
        "Diagnosed with obstructive sleep apnea; on CPAP.",
        "Known OSA — uses CPAP nightly.",
        "OSA confirmed on sleep study; CPAP compliant.",
        "Obstructive sleep apnea, CPAP dependent.",
        "Established OSA diagnosis; BiPAP at night.",
        "Sleeps {hrs} hours; known OSA on CPAP therapy.",
    ],
    "insomnia": [
        "Reports insomnia — difficulty falling and staying asleep.",
        "Chronic insomnia; sleeps {hrs} hours on a good night.",
        "Difficulty sleeping; estimates {hrs} hours per night.",
        "Insomnia — takes {hrs} hours to fall asleep.",
        "Poor sleep quality; insomnia reported.",
    ],
}

DIET = {
    "poor": [
        "Diet is poor — high sodium, frequent fast food.",
        "Poor diet; eats mostly processed and fast food.",
        "High fat, high sodium diet.",
        "Reports unhealthy diet — frequent fast food, skips meals.",
        "Diet consists largely of fast food and campus dining.",
        "Poor nutritional habits; high carbohydrate diet.",
        "Eats mostly processed food; low fruit and vegetable intake.",
        "High-fat, low-fiber diet.",
    ],
    "moderate": [
        "Diet is adequate but room for improvement.",
        "Average diet — some home cooking, occasional fast food.",
        "Moderate diet quality; could increase vegetable intake.",
        "Generally eats home-cooked meals but frequently snacks.",
    ],
    "good": [
        "Follows a balanced, healthy diet.",
        "Healthy diet — low sodium per cardiologist recommendation.",
        "Mediterranean diet per nutritionist guidance.",
        "Well-balanced diet; good fruit and vegetable intake.",
        "Healthy diet; avoids processed foods.",
        "Low sodium, low fat diet.",
        "Plant-based diet, primarily vegetables and legumes.",
    ],
    "therapeutic": [
        "Follows a diabetic diet.",
        "On a low-sodium cardiac diet.",
        "Renal diet due to chronic kidney disease.",
        "Low-carbohydrate diet per endocrinologist.",
        "Gluten-free diet for celiac disease.",
    ],
}

DRUG = {
    "never": [
        "Denies illicit drug use.",
        "No history of illicit drug use.",
        "Denies recreational drug use.",
        "Drug-free.",
        "No drug use reported.",
        "Never used illicit substances.",
    ],
    "marijuana": [
        "Admits to occasional marijuana use.",
        "Uses marijuana {freq}.",
        "Recreational marijuana use — {freq}.",
        "Cannabis use disorder; uses {freq}.",
        "Smokes marijuana {freq}.",
    ],
    "opioid": [
        "Known heroin user; last use {last_use}.",
        "IVDU — heroin use reported.",
        "Active opioid use disorder.",
        "Heroin dependence; currently using.",
        "IV heroin use; last use this morning.",
    ],
    "cocaine": [
        "Reports cocaine use.",
        "Cocaine use disorder.",
        "Uses cocaine occasionally.",
        "Active cocaine use.",
    ],
    "polysubstance": [
        "Polysubstance use — marijuana and cocaine.",
        "Uses multiple substances: heroin, marijuana.",
        "Known IVDU with concurrent marijuana use.",
    ],
    "former": [
        "Former marijuana user, quit {years} years ago.",
        "History of cocaine use; in recovery for {years} years.",
        "Former drug user — sober {years} years.",
        "Recovered from opioid use disorder; {years} years clean.",
    ],
}

# ─────────────────────────────────────────────
# Note Templates (mimic real clinical structure)
# ─────────────────────────────────────────────

NOTE_TEMPLATES = [
    # Full social history section
    "SOCIAL HISTORY:\n{smoking} {alcohol} {bmi} {activity} {sleep} {diet} {drug}",
    # Flowing paragraph
    "{age}-year-old {sex} with a history of {comorbidity}. {smoking} {alcohol} {bmi} {activity} {sleep} {diet} {drug}",
    # Bullet-style
    "Social History:\n- Tobacco: {smoking}\n- Alcohol: {alcohol}\n- BMI: {bmi}\n- Exercise: {activity}\n- Sleep: {sleep}\n- Diet: {diet}\n- Drugs: {drug}",
    # Progress note style
    "Patient seen today for follow-up. {smoking} {alcohol} {bmi} {activity} {sleep} {diet} {drug}",
    # ED note style (sparse)
    "{age}yo {sex} presenting with {chief_complaint}. {smoking} {alcohol} {drug}",
]

AGES = list(range(18, 85))
SEXES = ["male", "female"]
COMORBIDITIES = [
    "hypertension", "type 2 diabetes", "COPD", "heart failure",
    "chronic kidney disease", "depression", "anxiety", "hyperlipidemia",
    "asthma", "atrial fibrillation", "coronary artery disease",
]
CHIEF_COMPLAINTS = [
    "chest pain", "shortness of breath", "abdominal pain", "fall",
    "altered mental status", "syncope", "palpitations", "fever",
]


# ─────────────────────────────────────────────
# Value Generators
# ─────────────────────────────────────────────

def gen_smoking(status=None):
    status = status or random.choice(["current", "former", "never", "never", "never"])
    ppd = round(random.choice([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]), 1)
    years = random.randint(5, 40)
    py = round(ppd * years, 1)
    quit_year = random.randint(1990, 2020)
    years_ago = 2024 - quit_year
    cig = int(ppd * 20)
    start_age = random.randint(14, 25)

    templates = SMOKING[status]
    text = random.choice(templates).format(
        ppd=ppd, years=years, py=py, quit_year=quit_year,
        years_ago=years_ago, cig=cig, start_age=start_age
    )
    return text, {"status": status, "value": ppd if status == "current" else None,
                  "unit": "ppd", "pack_years": py if status != "never" else None}


def gen_alcohol(status=None):
    if status is None:
        status = random.choice(["current_light", "current_moderate", "current_heavy",
                                 "former", "never", "never"])
    drinks = round(random.uniform(1, 8), 1)
    drinks_week = round(drinks * 7 if "moderate" in status or "heavy" in status else random.randint(1, 4), 1)
    quit_year = random.randint(1995, 2020)
    years = 2024 - quit_year

    templates = ALCOHOL[status]
    text = random.choice(templates).format(
        drinks=drinks, drinks_week=drinks_week,
        quit_year=quit_year, years=years
    )

    if "never" in status:
        return text, {"status": "never", "value": None}
    elif "former" in status:
        return text, {"status": "former", "value": None}
    else:
        dpd = drinks if "moderate" in status or "heavy" in status else round(drinks_week / 7, 2)
        return text, {"status": "current", "value": round(dpd, 2), "unit": "drinks/day"}


def gen_bmi(bmi_class=None):
    ranges = {
        "underweight": (14.0, 18.4),
        "normal":      (18.5, 24.9),
        "overweight":  (25.0, 29.9),
        "obese_I":     (30.0, 34.9),
        "obese_II":    (35.0, 39.9),
        "obese_III":   (40.0, 55.0),
    }
    bmi_class = bmi_class or random.choices(
        list(ranges.keys()),
        weights=[0.05, 0.30, 0.30, 0.20, 0.10, 0.05]
    )[0]
    lo, hi = ranges[bmi_class]
    bmi = round(random.uniform(lo, hi), 1)

    text = random.choice(BMI[bmi_class]).format(bmi=bmi)
    return text, {"value": bmi, "class": bmi_class}


def gen_activity(level=None):
    level = level or random.choices(
        ["sedentary", "low", "moderate", "high"],
        weights=[0.30, 0.20, 0.35, 0.15]
    )[0]
    days = random.randint(2, 6)
    mins = random.choice([20, 30, 45, 60])
    miles = round(random.uniform(1.0, 6.0), 1)

    text = random.choice(ACTIVITY[level]).format(days=days, mins=mins, miles=miles)
    return text, {"level": level, "days_per_week": days if level in ("moderate", "high") else None}


def gen_sleep(condition=None):
    condition = condition or random.choices(
        ["adequate", "insufficient", "osa_suspected", "osa_confirmed", "insomnia"],
        weights=[0.40, 0.25, 0.15, 0.10, 0.10]
    )[0]

    if condition == "adequate":
        hrs = round(random.uniform(7.0, 9.0), 1)
    elif condition in ("insufficient", "insomnia"):
        hrs = round(random.uniform(3.5, 6.5), 1)
    else:
        hrs = round(random.uniform(4.5, 7.5), 1)

    text = random.choice(SLEEP[condition]).format(hrs=hrs)
    return text, {"condition": condition, "hours": hrs}


def gen_diet(quality=None):
    quality = quality or random.choices(
        ["poor", "moderate", "good", "therapeutic"],
        weights=[0.35, 0.30, 0.25, 0.10]
    )[0]
    text = random.choice(DIET[quality])
    return text, {"quality": quality if quality != "therapeutic" else "moderate"}


def gen_drug(status=None):
    status = status or random.choices(
        ["never", "marijuana", "opioid", "cocaine", "polysubstance", "former"],
        weights=[0.55, 0.20, 0.08, 0.05, 0.05, 0.07]
    )[0]
    freq = random.choice(["daily", "weekly", "occasionally", "on weekends"])
    last_use = random.choice(["this morning", "yesterday", "2 days ago"])
    years = random.randint(1, 10)
    quit_year = random.randint(2010, 2022)

    text = random.choice(DRUG[status]).format(
        freq=freq, last_use=last_use, years=years, quit_year=quit_year
    )

    if status == "never":
        return text, {"status": "never", "substances": []}
    elif status == "former":
        return text, {"status": "former", "substances": []}
    else:
        subs = {
            "marijuana": ["marijuana"],
            "opioid": ["heroin"],
            "cocaine": ["cocaine"],
            "polysubstance": ["marijuana", "heroin"],
        }.get(status, [])
        return text, {"status": "current", "substances": subs}


# ─────────────────────────────────────────────
# Note Generator
# ─────────────────────────────────────────────

def generate_note(note_id: str) -> dict:
    age = random.choice(AGES)
    sex = random.choice(SEXES)
    comorbidity = random.choice(COMORBIDITIES)
    chief_complaint = random.choice(CHIEF_COMPLAINTS)

    smoking_text, smoking_gt = gen_smoking()
    alcohol_text, alcohol_gt = gen_alcohol()
    bmi_text, bmi_gt = gen_bmi()
    activity_text, activity_gt = gen_activity()
    sleep_text, sleep_gt = gen_sleep()
    diet_text, diet_gt = gen_diet()
    drug_text, drug_gt = gen_drug()

    template = random.choice(NOTE_TEMPLATES)
    text = template.format(
        age=age, sex=sex, comorbidity=comorbidity,
        chief_complaint=chief_complaint,
        smoking=smoking_text, alcohol=alcohol_text,
        bmi=bmi_text, activity=activity_text,
        sleep=sleep_text, diet=diet_text, drug=drug_text,
    )

    return {
        "note_id": note_id,
        "patient_id": f"PT_{note_id.split('_')[1]}",
        "age": age,
        "sex": sex,
        "note_type": random.choice(["Progress Note", "Discharge Summary",
                                     "H&P", "ED Note", "Annual Physical"]),
        "text": text,
        "ground_truth": {
            "smoking": smoking_gt,
            "alcohol": alcohol_gt,
            "bmi": bmi_gt,
            "physical_activity": activity_gt,
            "sleep": sleep_gt,
            "diet": diet_gt,
            "drug_use": drug_gt,
        }
    }


def generate_dataset(n: int, output_path: str):
    notes = [generate_note(f"GEN_{i:04d}") for i in range(n)]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"Generated {n} notes → {output_path}")
    return notes


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of notes to generate")
    parser.add_argument("--output", default="data/generated_notes.json")
    parser.add_argument("--preview", type=int, default=2, help="Print N sample notes")
    args = parser.parse_args()

    notes = generate_dataset(args.n, args.output)

    if args.preview:
        print(f"\n{'='*60}")
        print(f"SAMPLE NOTES (showing {args.preview})")
        print('='*60)
        for note in notes[:args.preview]:
            print(f"\n[{note['note_id']}] {note['age']}yo {note['sex']} | {note['note_type']}")
            print(f"TEXT:\n{note['text']}")
            print(f"GROUND TRUTH: {json.dumps(note['ground_truth'], indent=4)}")
