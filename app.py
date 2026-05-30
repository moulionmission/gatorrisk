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
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #0021A5;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
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

tab1, tab2, tab3 = st.tabs(["📝 Single Note", "📦 Batch Upload", "📊 MTSamples Results"])

# ══════════════════════════════════════════════
# TAB 1 — Single Note
# ══════════════════════════════════════════════

with tab1:
    st.markdown("#### Paste a clinical note below")

    # Sample note picker
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

    note_text = st.text_area(
        "Clinical Note",
        value=default_text,
        height=220,
        placeholder="Paste any clinical note here — progress note, H&P, discharge summary...",
        label_visibility="collapsed",
    )

    run_btn = st.button("🔍 Extract Risk Factors", type="primary", use_container_width=True)

    if run_btn:
        if not note_text.strip():
            st.warning("Please paste a clinical note or select a sample.")
        else:
            with st.spinner("Running GatorRisk pipeline..."):
                result = pipeline.run_note(note_id="STREAMLIT_001", text=note_text)

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
                value=round(risk.composite_score * 100, 1),
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
                        "value": risk.composite_score * 100,
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
                if score >= 0.75:   colors.append("#c62828")
                elif score >= 0.50: colors.append("#f57c00")
                elif score >= 0.25: colors.append("#fbc02d")
                else:               colors.append("#2e7d32")

            fig_bar = go.Figure(go.Bar(
                x=list(factor_data.values()),
                y=list(factor_data.keys()),
                orientation="h",
                marker_color=colors,
                text=[f"{v:.2f}" for v in factor_data.values()],
                textposition="outside",
            ))
            fig_bar.update_layout(
                height=300,
                xaxis=dict(range=[0, 1.1], title="Risk Score (0–1)"),
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

                    with st.expander(f"{icon} {factor.replace('_',' ').title()}  —  score: {score:.2f}", expanded=True):
                        st.markdown(f'<span class="{tier_cls}">{tier}</span>', unsafe_allow_html=True)
                        if rf:
                            st.caption(rf.rationale)
                        st.markdown("---")
                        for k, v in data.items():
                            if v not in (None, [], "unknown", "") and k != "note_id":
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
    st.markdown("CSV must have columns: `note_id`, `text` (and optionally `specialty`)")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_up)} rows")

        # Auto-rename transcription → text (handles MTSamples directly)
        if "transcription" in df_up.columns and "text" not in df_up.columns:
            df_up = df_up.rename(columns={"transcription": "text"})
            st.info("Auto-detected MTSamples format — renamed 'transcription' column to 'text'")

        if "text" not in df_up.columns:
            st.error("CSV must have a 'text' or 'transcription' column.")
        else:
            # Auto-filter to notes that contain social history keywords
            social_keywords = (
                "social history|smok|alcohol|bmi|exercise|"
                "sleep|drug use|tobacco|drinks|sedentary|obesity"
            )
            before = len(df_up)
            mask = df_up["text"].str.contains(social_keywords, case=False, na=False)
            df_up = df_up[mask].reset_index(drop=True)
            after = len(df_up)

            if before != after:
                st.info(
                    f"🔍 Auto-filtered: {before} total notes → "
                    f"**{after} notes with lifestyle content** "
                    f"({before - after} notes skipped — no social history found)"
                )
            else:
                st.info(f"All {after} notes contain lifestyle content — no filtering needed")

            st.dataframe(df_up.head(3), use_container_width=True)
        else:
            limit = st.slider("Max notes to process", 5, len(df_up), min(200, len(df_up)))

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

                # Summary table
                rows = []
                for r in results:
                    rows.append({
                        "Note ID": r.note_id,
                        "Risk Score": round(r.risk_profile.composite_score, 3),
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

                # Distribution chart
                fig = px.histogram(result_df, x="Risk Score", color="Tier",
                                   color_discrete_map={
                                       "LOW": "#2e7d32", "MODERATE": "#f57c00",
                                       "HIGH": "#c62828", "CRITICAL": "#4a0000"
                                   },
                                   nbins=20, title="Risk Score Distribution")
                fig.update_layout(height=300, margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)

                # Download results
                csv_out = result_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv_out,
                    file_name="gatorrisk_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# ══════════════════════════════════════════════
# TAB 3 — MTSamples Results
# ══════════════════════════════════════════════

with tab3:
    st.markdown("#### Pre-computed results on 1,564 real MTSamples notes")

    results_path = Path("data/processed/mtsamples_results.json")

    if not results_path.exists():
        st.info("Run `python scripts/run_mtsamples.py` first to generate results.")
    else:
        with open(results_path) as f:
            mt_results = json.load(f)

        scores = [r["risk_profile"]["composite_score"] for r in mt_results]
        tiers  = [r["risk_profile"]["composite_tier"]  for r in mt_results]

        from collections import Counter
        tier_counts = Counter(tiers)

        # Summary metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Notes", f"{len(mt_results):,}")
        c2.metric("Avg Risk Score", f"{sum(scores)/len(scores):.3f}")
        c3.metric("Max Risk Score", f"{max(scores):.3f}")
        c4.metric("MODERATE+ Notes", f"{sum(1 for t in tiers if t != 'LOW')}")

        st.markdown("---")
        col_a, col_b = st.columns(2)

        # Score distribution
        with col_a:
            fig_hist = px.histogram(
                x=scores, nbins=30,
                title="Composite Risk Score Distribution",
                labels={"x": "Risk Score"},
                color_discrete_sequence=["#0021A5"],
            )
            fig_hist.update_layout(height=300, margin=dict(t=40, b=10), showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Tier pie
        with col_b:
            fig_pie = px.pie(
                names=list(tier_counts.keys()),
                values=list(tier_counts.values()),
                title="Risk Tier Breakdown",
                color=list(tier_counts.keys()),
                color_discrete_map={
                    "LOW": "#2e7d32", "MODERATE": "#f57c00",
                    "HIGH": "#c62828", "CRITICAL": "#4a0000",
                },
            )
            fig_pie.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Factor extraction rates
        st.markdown("### Factor Extraction Rates")
        from collections import defaultdict

        factor_extracted = defaultdict(int)
        factors = ["smoking", "alcohol", "bmi", "physical_activity", "sleep", "diet", "drug_use"]
        for r in mt_results:
            profile = r["normalized_profile"]
            for factor in factors:
                status = profile.get(factor, {}).get("status", "unknown")
                if status not in ("unknown", None):
                    factor_extracted[factor] += 1

        rate_data = {
            f.replace("_", " ").title(): round(factor_extracted[f] / len(mt_results) * 100, 1)
            for f in factors
        }
        fig_rate = px.bar(
            x=list(rate_data.values()),
            y=list(rate_data.keys()),
            orientation="h",
            title="% of Notes Where Factor Was Extracted",
            labels={"x": "Extraction Rate (%)", "y": ""},
            color=list(rate_data.values()),
            color_continuous_scale=["#e8f5e9", "#0021A5"],
            text=[f"{v}%" for v in rate_data.values()],
        )
        fig_rate.update_layout(height=320, margin=dict(t=40, b=10),
                               coloraxis_showscale=False)
        fig_rate.update_traces(textposition="outside")
        st.plotly_chart(fig_rate, use_container_width=True)

        # Top high risk notes
        st.markdown("### Highest Risk Notes")
        top_notes = sorted(mt_results, key=lambda r: -r["risk_profile"]["composite_score"])[:10]
        top_rows = []
        for r in top_notes:
            p = r["normalized_profile"]
            top_rows.append({
                "Note ID": r["note_id"],
                "Score": round(r["risk_profile"]["composite_score"], 3),
                "Tier": r["risk_profile"]["composite_tier"],
                "Smoking": p["smoking"].get("status"),
                "BMI": p["bmi"].get("value"),
                "BMI Class": p["bmi"].get("bmi_class"),
                "Drug Use": p["drug_use"].get("status"),
            })
        st.dataframe(pd.DataFrame(top_rows), use_container_width=True)

        # Smoking breakdown
        st.markdown("### Smoking Status Across All Notes")
        smoking_statuses = [r["normalized_profile"]["smoking"]["status"] for r in mt_results]
        sm_counts = Counter(smoking_statuses)
        fig_sm = px.bar(
            x=list(sm_counts.keys()),
            y=list(sm_counts.values()),
            title="Smoking Status Distribution",
            labels={"x": "Status", "y": "Count"},
            color=list(sm_counts.keys()),
            color_discrete_map={
                "current": "#c62828", "former": "#f57c00",
                "never": "#2e7d32", "unknown": "#9e9e9e",
            },
        )
        fig_sm.update_layout(height=300, margin=dict(t=40, b=10), showlegend=False)
        st.plotly_chart(fig_sm, use_container_width=True)

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
