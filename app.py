"""
app.py — GatorRisk Streamlit App
=================================
Run locally:
    streamlit run app.py

Deploy to Streamlit Cloud (free):
    1. Push gatorrisk to GitHub
    2. Go to share.streamlit.io
    3. Connect repo → set Main file = app.py → Deploy
"""

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from modules.pipeline import Pipeline

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="GatorRisk",
    page_icon="🐊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0021A5;
        margin-bottom: 0;
    }
    .sub-title {
        font-size: 1rem;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .uf-badge {
        background: #FA4616;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .risk-LOW      { color: #2e7d32; font-weight: 700; }
    .risk-MODERATE { color: #f57c00; font-weight: 700; }
    .risk-HIGH     { color: #c62828; font-weight: 700; }
    .risk-CRITICAL { color: #4a0000; font-weight: 700; background: #ffebee; padding: 2px 6px; border-radius: 4px; }
    .disclaimer {
        background: #fff3e0;
        border: 1px solid #ffb74d;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        font-size: 0.8rem;
        color: #e65100;
    }
    div[data-testid="stTabs"] button { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown('<p class="main-title">🐊 GatorRisk</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Clinical NLP Pipeline for Lifestyle Risk Factor Extraction</p>',
                unsafe_allow_html=True)
with col_badge:
    st.markdown('<br><span class="uf-badge">UF CTSI Extension</span>', unsafe_allow_html=True)

st.markdown('<div class="disclaimer">⚠️ <b>Research Use Only</b> — Not a certified clinical decision support system. All outputs require clinician review.</div>',
            unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# Pipeline (cached so it loads once)
# ─────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    return Pipeline(use_transformer=False, deidentify=True)

pipeline = load_pipeline()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    deidentify = st.toggle("De-identify PHI", value=True,
                           help="Remove dates, MRNs, provider names before processing")
    show_entities = st.toggle("Show raw entities", value=False,
                              help="Show every NER entity found in the note")
    show_relations = st.toggle("Show relations", value=False,
                               help="Show entity-value pairs before normalization")

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    GatorRisk extracts **7 lifestyle risk factors** from unstructured clinical notes:

    - 🚬 Smoking
    - 🍺 Alcohol
    - ⚖️ BMI
    - 🏃 Physical Activity
    - 😴 Sleep
    - 🥗 Diet
    - 💊 Drug Use

    Built at **University of Florida** as an extension of the
    [CTSI NLP Core](https://www.ctsi.ufl.edu/research/laboratory-services/nlp-core/)
    GatorTron smoking extractor.
    """)
    st.markdown("---")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-moulionmission%2Fgatorrisk-blue?logo=github)](https://github.com/moulionmission/gatorrisk)")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📝 Single Note", "📦 Batch Upload", "ℹ️ About"])

# ══════════════════════════════════════════════
# TAB 1 — Single Note
# ══════════════════════════════════════════════

with tab1:
    st.markdown("#### Paste a clinical note below")

    samples = {
        "-- Select a sample --": "",
        "High risk patient (smoker, obese, OSA)": """58-year-old male presenting for annual physical.
SOCIAL HISTORY:
Patient smokes 1.5 packs per day for the past 30 years, with no intention to quit.
Drinks approximately 3 beers nightly.
BMI is 34.2, consistent with class I obesity.
Patient leads a sedentary lifestyle with no regular exercise.
Sleeps only 4-5 hours per night and reports loud snoring; OSA is suspected.
Diet is poor — high sodium, frequent fast food.
Denies illicit drug use.""",

        "Low risk patient (active, healthy)": """45-year-old female presenting for wellness visit.
SOCIAL HISTORY:
Former smoker — quit in 2018 after 10 pack-years.
She is a social drinker, approximately 1-2 glasses of wine on weekends.
BMI 22.4, normal weight.
Exercises 4 days per week, 45 minutes moderate intensity walking and cycling.
Sleeps 7-8 hours nightly, no sleep complaints.
Follows a balanced, healthy diet per nutritionist recommendation.
Denies any illicit or recreational drug use.""",

        "ED patient (IVDU)": """32-year-old male brought in by EMS.
Known IVDU — admits to heroin use, last use this morning.
Also uses marijuana daily.
Does not smoke cigarettes.
Alcohol: denies regular use.
BMI not recorded, appears underweight.
No known exercise routine.
Sleep reportedly erratic.
Diet history not obtained.""",

        "College student": """22-year-old college student presenting for routine checkup.
Non-smoker. Drinks socially, approximately 5-6 drinks on weekends.
BMI 22.4, normal. Very active — runs 5 miles 4 days per week and lifts weights twice weekly.
Sleeps 6 hours on weekdays, 9 on weekends.
Diet consists largely of fast food and campus dining.
Admits to occasional marijuana use on weekends, denies other drug use.""",
    }

    selected = st.selectbox("Or load a sample:", list(samples.keys()))
    default_text = samples[selected]

    # Use default_text directly as value — Streamlit text_area with no key
    # means the widget value is always what the user sees/types
    # We DON'T use session_state here to avoid conflicts
    note_text = st.text_area(
        "Clinical Note",
        value=default_text,
        height=220,
        placeholder="Paste any clinical note here — progress note, H&P, discharge summary...",
        label_visibility="collapsed",
    )

    run_btn = st.button("🔍 Extract Risk Factors", type="primary", use_container_width=True)

    if run_btn:
        actual_text = note_text
        if not actual_text.strip():
            st.warning("Please paste a clinical note or select a sample.")
        else:
            with st.spinner("Running GatorRisk pipeline..."):
                result = pipeline.run_note(note_id="STREAMLIT_001", text=actual_text)

            profile = result.normalized_profile
            risk    = result.risk_profile

            st.success(f"✓ Processed in {result.processing_time_ms:.1f}ms — {len(result.processed_note.sentences)} sentences analyzed")

            # ── Composite Score ──────────────────────────
            st.markdown("### Composite Risk Score")

            tier_colors = {"LOW": "#2e7d32", "MODERATE": "#f57c00",
                           "HIGH": "#c62828", "CRITICAL": "#4a0000"}
            tier_color = tier_colors.get(risk.composite_tier, "#999")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(risk.composite_score, 1),
                number={"suffix": "%", "font": {"size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": tier_color},
                    "steps": [
                        {"range": [0,  25], "color": "#e8f5e9"},
                        {"range": [25, 50], "color": "#fff3e0"},
                        {"range": [50, 75], "color": "#ffebee"},
                        {"range": [75,100], "color": "#ffcdd2"},
                    ],
                    "threshold": {
                        "line": {"color": tier_color, "width": 4},
                        "thickness": 0.75,
                        "value": risk.composite_score,
                    },
                },
                title={"text": f"<b>{risk.composite_tier}</b>", "font": {"size": 20, "color": tier_color}},
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=30, b=0, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Per-Factor Bar Chart ─────────────────────
            st.markdown("### Risk by Factor")

            factor_data = {
                f.factor.replace("_", " ").title(): f.score
                for f in sorted(risk.factors, key=lambda x: -x.score)
            }
            colors = []
            for score in factor_data.values():
                if score >= 75.0:   colors.append("#c62828")
                elif score >= 50.0: colors.append("#f57c00")
                elif score >= 25.0: colors.append("#fbc02d")
                else:               colors.append("#2e7d32")

            fig_bar = go.Figure(go.Bar(
                x=list(factor_data.values()),
                y=list(factor_data.keys()),
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1f}" for v in factor_data.values()],
                textposition="outside",
            ))
            fig_bar.update_layout(
                height=300,
                xaxis=dict(range=[0, 105], title="Risk Score (0–100)"),
                margin=dict(t=10, b=10, l=10, r=40),
                plot_bgcolor="white",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Factor Detail Cards ──────────────────────
            st.markdown("### Extracted Values")

            cols = st.columns(2)
            factor_icons = {
                "smoking": "🚬", "alcohol": "🍺", "bmi": "⚖️",
                "physical_activity": "🏃", "sleep": "😴",
                "diet": "🥗", "drug_use": "💊",
            }

            profile_dict = profile.to_dict()
            risk_map = {f.factor: f for f in risk.factors}

            for i, factor in enumerate(["smoking", "alcohol", "bmi", "physical_activity",
                                         "sleep", "diet", "drug_use"]):
                col = cols[i % 2]
                data = profile_dict.get(factor, {})
                rf   = risk_map.get(factor)
                icon = factor_icons.get(factor, "•")

                with col:
                    tier = rf.tier if rf else "LOW"
                    score = rf.score if rf else 0.0
                    tier_cls = f"risk-{tier}"

                    with st.expander(f"{icon} {factor.replace('_',' ').title()}  —  score: {score:.1f}", expanded=True):
                        st.markdown(f'<span class="{tier_cls}">{tier}</span>', unsafe_allow_html=True)
                        if rf:
                            st.caption(rf.rationale)
                            # Render structured attributes as pills
                            record_obj = getattr(result.normalized_profile, factor, None)
                            if record_obj:
                                exp = getattr(record_obj, "experiencer", "patient")
                                pol = getattr(record_obj, "polarity", "affirmed")
                                cert = getattr(record_obj, "certainty", "certain")
                                
                                exp_color = "#2e7d32" if exp == "patient" else "#ef6c00"
                                pol_color = "#2e7d32" if pol == "affirmed" else "#c62828"
                                cert_color = "#2e7d32" if cert == "certain" else "#d84315"
                                
                                st.markdown(
                                    f'<div style="margin-top: 5px; margin-bottom: 10px; display: flex; flex-wrap: wrap; gap: 5px;">'
                                    f'<span style="background-color: {exp_color}18; color: {exp_color}; border: 1px solid {exp_color}40; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">Experiencer: {exp.upper()}</span>'
                                    f'<span style="background-color: {pol_color}18; color: {pol_color}; border: 1px solid {pol_color}40; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">Polarity: {pol.upper()}</span>'
                                    f'<span style="background-color: {cert_color}18; color: {cert_color}; border: 1px solid {cert_color}40; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">Certainty: {cert.upper()}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                        st.markdown("---")
                        for k, v in data.items():
                            if v not in (None, [], "unknown", "") and k != "note_id" and k not in ("experiencer", "polarity", "certainty"):
                                st.markdown(f"**{k}:** `{v}`")

            # ── Optional: Raw Entities ───────────────────
            if show_entities:
                st.markdown("### Raw NER Entities")
                entity_rows = [
                    {"Factor": e.label, "Sub-label": e.sub_label,
                     "Text": e.text, "Source": e.source, "Confidence": e.confidence}
                    for e in result.ner_result.entities
                ]
                if entity_rows:
                    st.dataframe(pd.DataFrame(entity_rows), use_container_width=True)
                else:
                    st.info("No entities found.")

            if show_relations:
                st.markdown("### Relations")
                rel_rows = [
                    {"Factor": r.factor, "Status": r.status,
                     "Value": r.value, "Unit": r.unit, "Flags": str(r.flags)}
                    for r in result.relation_result.relations
                ]
                if rel_rows:
                    st.dataframe(pd.DataFrame(rel_rows), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — Batch Upload
# ══════════════════════════════════════════════

with tab2:
    st.markdown("#### Upload a CSV of clinical notes")
    st.markdown("CSV must have a `text` column. MTSamples format (`transcription` column) is auto-detected.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_up)} rows")

        # Auto-rename transcription → text
        if "transcription" in df_up.columns and "text" not in df_up.columns:
            df_up = df_up.rename(columns={"transcription": "text"})
            st.info("Auto-detected MTSamples format — renamed 'transcription' to 'text'")

        if "text" not in df_up.columns:
            st.error("CSV must have a 'text' or 'transcription' column.")
        else:
            # Optional specialty filter — only shown if column exists
            if "medical_specialty" in df_up.columns:
                specialties = sorted(df_up["medical_specialty"].dropna().unique())
                use_specialty_filter = st.checkbox(
                    "Filter by specialty (optional)",
                    value=False,
                    help="Filter to specific note types. Leave unchecked to run on all notes."
                )
                if use_specialty_filter:
                    selected_specs = st.multiselect(
                        "Select specialties to include:",
                        options=specialties,
                        default=[s for s in specialties if s.strip() in [
                            "Consult - History and Phy.", "General Medicine",
                            "Emergency Room Reports", "Discharge Summary",
                            "SOAP / Chart / Progress Notes", "Psychiatry / Psychology"
                        ]]
                    )
                    if selected_specs:
                        before = len(df_up)
                        df_up = df_up[df_up["medical_specialty"].isin(selected_specs)].reset_index(drop=True)
                        st.info(f"Filtered to {len(df_up)} notes from {len(selected_specs)} specialties ({before - len(df_up)} excluded)")

            st.dataframe(df_up.head(3), use_container_width=True)

            process_all = st.checkbox("Process entire file (may be slow for large files)")
            if process_all:
                limit = len(df_up)
            elif len(df_up) <= 5:
                limit = len(df_up)
            else:
                limit = st.slider("Max notes to process", 5, min(500, len(df_up)), min(200, len(df_up)))

            if st.button("🚀 Run Batch", type="primary"):
                df_sample = df_up.head(limit).copy()
                if "note_id" not in df_sample.columns:
                    df_sample["note_id"] = [f"NOTE_{i}" for i in range(len(df_sample))]

                notes = [
                    {"note_id": str(row["note_id"]), "text": str(row["text"])}
                    for _, row in df_sample.iterrows()
                    if pd.notna(row["text"])
                ]

                progress = st.progress(0, text="Processing notes...")
                results = []
                for i, note in enumerate(notes):
                    r = pipeline.run_note(**note)
                    results.append(r)
                    progress.progress((i + 1) / len(notes),
                                      text=f"Processed {i+1}/{len(notes)} notes")

                progress.empty()
                st.success(f"✓ Done — {len(results)} notes processed")

                rows = []
                for r in results:
                    rows.append({
                        "Note ID": r.note_id,
                        "Risk Score": round(r.risk_profile.composite_score, 1),
                        "Tier": r.risk_profile.composite_tier,
                        "Smoking": r.normalized_profile.smoking.status,
                        "BMI": r.normalized_profile.bmi.value,
                        "BMI Class": r.normalized_profile.bmi.bmi_class,
                        "Alcohol": r.normalized_profile.alcohol.status,
                        "Activity": r.normalized_profile.physical_activity.level,
                        "Sleep Hrs": r.normalized_profile.sleep.hours_per_night,
                        "Drug Use": r.normalized_profile.drug_use.status,
                        "Time (ms)": r.processing_time_ms,
                    })

                result_df = pd.DataFrame(rows)
                st.dataframe(result_df, use_container_width=True)

                fig = px.histogram(result_df, x="Risk Score", color="Tier",
                                   color_discrete_map={
                                       "LOW": "#2e7d32", "MODERATE": "#f57c00",
                                       "HIGH": "#c62828", "CRITICAL": "#4a0000"
                                   },
                                   nbins=20, title="Risk Score Distribution")
                fig.update_layout(height=300, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

                csv_out = result_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv_out,
                    file_name="gatorrisk_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ══════════════════════════════════════════════
# TAB 3 — About
# ══════════════════════════════════════════════

with tab3:
    st.markdown("### 🐊 About GatorRisk")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### What It Does
        GatorRisk reads unstructured clinical notes and automatically
        extracts **7 lifestyle risk factors**, scoring each on a 0–100 risk scale.

        | Factor | Extracts |
        |---|---|
        | 🚬 Smoking | status, ppd, pack-years |
        | 🍺 Alcohol | status, drinks/day, pattern |
        | ⚖️ BMI | value, obesity class |
        | 🏃 Physical Activity | level, frequency, duration |
        | 😴 Sleep | hours, OSA, CPAP |
        | 🥗 Diet | quality, flags |
        | 💊 Drug Use | status, substances |

        #### How to Use
        - **Single Note** — paste any clinical note and get instant results
        - **Batch Upload** — upload a CSV of notes, get a full results table + chart
        - Supports MTSamples CSV format directly — no manual prep needed
        """)

    with col2:
        st.markdown("""
        #### Architecture
        ```
        Raw Clinical Note
              ↓
        [1] Preprocessor
            clean text, de-identify PHI
              ↓
        [2] NER Extractor
            rule-based + fuzzy matching
              ↓
        [3] Relation Extractor
            link entities to values
              ↓
        [4] Normalizer + BMI Calculator
            structured output schema
              ↓
        [5] Risk Scorer
            0–100 score + tier
        ```

        #### Data Sources
        - **MTSamples** — 4,966 real transcribed notes (Kaggle, free)
        - **MIMIC-III** — 2M+ ICU notes (PhysioNet, credentialing required)
        - **UF Health EHR** — via CTSI collaboration

        #### Built At
        **University of Florida** — extending the
        [CTSI NLP Core](https://www.ctsi.ufl.edu/research/laboratory-services/nlp-core/)
        GatorTron smoking extractor to 7 lifestyle risk factors.

        #### Links
        - 📂 [GitHub](https://github.com/moulionmission/gatorrisk)
        - 🏥 [UF CTSI NLP Core](https://www.ctsi.ufl.edu/research/laboratory-services/nlp-core/)
        - 📧 Collaboration: yonghui.wu@ufl.edu
        """)

    st.markdown("---")
    st.markdown("""
    > ⚠️ **Research Use Only** — GatorRisk is not a certified clinical decision
    > support system. All risk assessments require review by a licensed clinician.
    """)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<center style='color:#999; font-size:0.8rem'>"
    "GatorRisk v1.0 · University of Florida · "
    "Extending <a href='https://www.ctsi.ufl.edu/research/laboratory-services/nlp-core/' target='_blank'>UF CTSI NLP Core</a> · "
    "<a href='https://github.com/moulionmission/gatorrisk' target='_blank'>GitHub</a>"
    "</center>",
    unsafe_allow_html=True,
)
