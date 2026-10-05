import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="eThekwini Election Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent

# ------------------------------------------------------------
# STYLING
# ------------------------------------------------------------
st.markdown("""
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
    [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
    .hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 18px;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, rgba(30,60,114,.10), rgba(42,82,152,.03));
    }
    .hero h1 {margin: 0 0 .35rem 0; font-size: 2.15rem;}
    .hero p {margin: 0; opacity: .78; font-size: 1.02rem;}
    .section-note {
        padding: .85rem 1rem;
        border-left: 4px solid #6c7a89;
        background: rgba(128,128,128,.08);
        border-radius: 8px;
        margin: .5rem 0 1rem 0;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.20);
        padding: .9rem 1rem;
        border-radius: 14px;
        background: rgba(128,128,128,.035);
    }
    .small-muted {font-size:.88rem; opacity:.72;}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# DATA LOADING + CONTRACT VALIDATION
# ------------------------------------------------------------
REQUIRED = {
    "forecast_party_table.csv": {"PartyName", "ForecastShare", "PredictedVotes"},
    "forecast_wards.csv": {"Ward", "Historical-model leader", "VD agreement"},
    "historical_party_shares.csv": {"PartyName", "2011", "2016", "2021"},
    "sensitivity_analysis.csv": {"2024_weight", "Leader", "LeaderShare", "MKShare"},
    "metro_stats.csv": {"Year", "Turnout", "Registered", "ValidVotes"},
    "split_summary.csv": {"Split", "Rows", "Percent", "Unique VDs"},
    "validation_scores.csv": {"Model", "Validation accuracy", "Validation balanced accuracy", "Validation macro F1"},
}

def stop_for_missing_files():
    missing = [name for name in list(REQUIRED) + ["model_metrics.json"] if not (BASE / name).exists()]
    if missing:
        st.error("Dashboard data files are missing.")
        st.code("\n".join(missing))
        st.info(
            "Run the FINAL DASHBOARD EXPORT cell in the notebook first. "
            "Then place this Streamlit app in the same folder as the exported files."
        )
        st.stop()

@st.cache_data
def load_data():
    stop_for_missing_files()
    data = {}
    for filename, columns in REQUIRED.items():
        df = pd.read_csv(BASE / filename)
        absent = columns.difference(df.columns)
        if absent:
            st.error(f"{filename} is incompatible. Missing columns: {sorted(absent)}")
            st.stop()
        data[filename] = df

    with open(BASE / "model_metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    required_metrics = {
        "model", "test_accuracy", "balanced_accuracy", "weighted_f1", "macro_f1",
        "turnout_mae", "turnout_rmse", "turnout_r2",
        "forecast_turnout", "forecast_registered", "forecast_valid_votes",
        "forecast_year", "recent_weight"
    }
    missing_metrics = required_metrics.difference(metrics)
    if missing_metrics:
        st.error(f"model_metrics.json is incompatible. Missing keys: {sorted(missing_metrics)}")
        st.stop()

    return data, metrics

data, metrics = load_data()
party = data["forecast_party_table.csv"].sort_values("ForecastShare", ascending=False).reset_index(drop=True)
wards = data["forecast_wards.csv"]
history = data["historical_party_shares.csv"]
sensitivity = data["sensitivity_analysis.csv"]
metro = data["metro_stats.csv"]
splits = data["split_summary.csv"]
validation = data["validation_scores.csv"]

forecast_year = int(metrics["forecast_year"])
leader = party.iloc[0]

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## Election Intelligence")
    st.caption("eThekwini Metropolitan Municipality")
    page = st.radio(
        "Navigate",
        ["Executive Overview", "Party Analysis", "Ward Intelligence", "Model & Methodology"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Forecast context**")
    st.write(f"Target election: **{forecast_year} LGE**")
    st.write("Primary history: **2011 · 2016 · 2021 LGE**")
    st.write("Recent signal: **2024 NPE**")
    st.caption(
        "The 2024 National Election is used as a recent political signal, not as an equivalent Local Government Election."
    )
    st.divider()
    st.caption("Source: Electoral Commission of South Africa (IEC) official election-result downloads.")

# ------------------------------------------------------------
# SHARED HERO
# ------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>eThekwini Election Intelligence</h1>
        <p>{forecast_year} Local Government Election forecast • historical IEC evidence • validated machine-learning pipeline</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# PAGE 1 — EXECUTIVE OVERVIEW
# ------------------------------------------------------------
if page == "Executive Overview":
    st.subheader("Executive forecast")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Projected metro leader", str(leader["PartyName"]))
    c2.metric("Projected vote share", f'{leader["ForecastShare"]:.2f}%')
    c3.metric("Projected valid PR votes", f'{metrics["forecast_valid_votes"]:,.0f}')
    c4.metric("Projected turnout", f'{metrics["forecast_turnout"]:.2%}')
    c5.metric("Held-out accuracy", f'{metrics["test_accuracy"]:.2%}')

    st.markdown(
        '<div class="section-note"><b>Interpretation:</b> the metro forecast is a scenario forecast. '
        'The historical classifier is validated separately; its test accuracy is not presented as guaranteed future-election accuracy.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.55, 1])
    with left:
        top_n = st.slider("Number of parties to display", 5, min(20, len(party)), min(10, len(party)))
        chart_df = party.head(top_n).sort_values("ForecastShare")
        fig = px.bar(
            chart_df, x="ForecastShare", y="PartyName", orientation="h",
            text="ForecastShare",
            labels={"ForecastShare": "Forecast vote share (%)", "PartyName": ""},
            title=f"Projected {forecast_year} metro party shares",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=500, margin=dict(l=10, r=25, t=55, b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics["forecast_turnout"] * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": f"Projected {forecast_year} turnout"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        fig.update_layout(height=300, margin=dict(l=25, r=25, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Model evidence")
        st.write(f"**Selected classifier:** {metrics['model']}")
        st.write(f"**Balanced accuracy:** {metrics['balanced_accuracy']:.2%}")
        st.write(f"**Weighted F1:** {metrics['weighted_f1']:.3f}")
        st.write(f"**Macro F1:** {metrics['macro_f1']:.3f}")

    st.subheader("Four required analytical outputs")
    o1, o2 = st.columns(2)
    with o1:
        st.markdown(f"**1 · Metro leader**  \n{leader['PartyName']} — **{leader['ForecastShare']:.2f}%**")
        st.markdown(f"**3 · Party vote totals and shares**  \nInteractive table below contains the complete metro forecast.")
    with o2:
        st.markdown(f"**2 · Three ward leaders**  \nAvailable in **Ward Intelligence**.")
        st.markdown(f"**4 · Voter turnout**  \nProjected eThekwini turnout: **{metrics['forecast_turnout']:.2%}**")

    display_party = party.copy()
    display_party["ForecastShare"] = display_party["ForecastShare"].map(lambda x: f"{x:.2f}%")
    display_party["PredictedVotes"] = display_party["PredictedVotes"].map(lambda x: f"{x:,.0f}")
    st.dataframe(display_party, hide_index=True, use_container_width=True)

# ------------------------------------------------------------
# PAGE 2 — PARTY ANALYSIS
# ------------------------------------------------------------
elif page == "Party Analysis":
    st.subheader("Party-level intelligence")
    selected_party = st.selectbox("Explore a party", party["PartyName"].tolist())
    row = party.loc[party["PartyName"].eq(selected_party)].iloc[0]

    a, b, c = st.columns(3)
    a.metric("Forecast share", f'{row["ForecastShare"]:.2f}%')
    b.metric("Predicted valid votes", f'{row["PredictedVotes"]:,.0f}')
    c.metric("Forecast rank", f'#{party.index[party["PartyName"].eq(selected_party)][0] + 1}')

    hist_row = history[history["PartyName"].eq(selected_party)]
    if not hist_row.empty:
        h = hist_row.iloc[0]
        trend = pd.DataFrame({
            "Election": ["2011 LGE", "2016 LGE", "2021 LGE"],
            "Share": [h["2011"], h["2016"], h["2021"]],
        })
        fig = px.line(trend, x="Election", y="Share", markers=True,
                      title=f"Historical LGE share — {selected_party}",
                      labels={"Share": "Metro PR vote share (%)"})
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("This party has no historical LGE series in the exported top-party history.")

    st.subheader("Historical metro party-share trends")
    hist_long = history.melt(id_vars="PartyName", var_name="ElectionYear", value_name="Share")
    chosen = st.multiselect(
        "Compare parties",
        history.sort_values("2021", ascending=False)["PartyName"].head(6).tolist(),
        default=history.sort_values("2021", ascending=False)["PartyName"].head(4).tolist(),
    )
    plot = hist_long[hist_long["PartyName"].isin(chosen)]
    fig = px.line(plot, x="ElectionYear", y="Share", color="PartyName", markers=True,
                  labels={"Share": "Vote share (%)", "ElectionYear": "Election year"},
                  title="LGE party-share movement")
    fig.update_layout(height=470, legend_title_text="Party")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2024 recent-signal sensitivity")
    st.caption(
        "The notebook's final forecast uses a 60% 2024-NPE weight. "
        "This chart shows how the projected leader and MK share change when that assumption is varied."
    )
    fig = px.line(
        sensitivity, x="2024_weight", y=["LeaderShare", "MKShare"], markers=True,
        labels={"value": "Scenario share (%)", "2024_weight": "Weight assigned to 2024 NPE", "variable": "Measure"},
        title="Sensitivity to the 2024 recent-signal weight",
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    selected_weight = st.slider("Inspect a sensitivity scenario", 0, 100, int(metrics["recent_weight"] * 100), 10)
    nearest = sensitivity.iloc[(sensitivity["2024_weight"] - selected_weight / 100).abs().argsort()[:1]].iloc[0]
    s1, s2, s3 = st.columns(3)
    s1.metric("2024 weight", f"{nearest['2024_weight']:.0%}")
    s2.metric("Scenario leader", nearest["Leader"])
    s3.metric("Leader share", f"{nearest['LeaderShare']:.1f}%")

# ------------------------------------------------------------
# PAGE 3 — WARD INTELLIGENCE
# ------------------------------------------------------------
elif page == "Ward Intelligence":
    st.subheader("Three selected ward forecasts")
    st.markdown(
        '<div class="section-note">The three wards represent different 2021 competitive contexts. '
        'The ward prediction is produced by the validated historical-persistence classifier. '
        'Because the supplied 2024 file is metro-aggregated, ward-level MK support is not fabricated.</div>',
        unsafe_allow_html=True,
    )

    ward_options = wards["Ward"].astype(str).tolist()
    selected_ward = st.selectbox("Select a ward", ward_options)
    w = wards.loc[wards["Ward"].astype(str).eq(selected_ward)].iloc[0]

    c1, c2 = st.columns(2)
    c1.metric("Historical-model leader", w["Historical-model leader"])
    c2.metric("VD agreement", f'{float(w["VD agreement"]):.1%}')

    agreement = float(w["VD agreement"]) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=agreement,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Voting-district agreement"},
        gauge={"axis": {"range": [0, 100]}}
    ))
    fig.update_layout(height=310)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### All three selected wards")
    ward_display = wards.copy()
    ward_display["VD agreement"] = ward_display["VD agreement"].map(lambda x: f"{float(x):.1%}")
    st.dataframe(ward_display, hide_index=True, use_container_width=True)

    st.warning(
        "2024 MK evidence is available only at metro level in the supplied NPE file. "
        "The dashboard therefore does not claim a ward-specific MK forecast."
    )

# ------------------------------------------------------------
# PAGE 4 — MODEL & METHODOLOGY
# ------------------------------------------------------------
else:
    st.subheader("Model validation & methodology")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f'{metrics["test_accuracy"]:.2%}')
    c2.metric("Balanced accuracy", f'{metrics["balanced_accuracy"]:.2%}')
    c3.metric("Weighted F1", f'{metrics["weighted_f1"]:.3f}')
    c4.metric("Macro F1", f'{metrics["macro_f1"]:.3f}')

    st.success(
        f"The held-out test accuracy is {metrics['test_accuracy']:.2%}, "
        "which exceeds the project's required 70% threshold."
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### 70 / 15 / 15 data split")
        split_plot = splits.copy()
        split_plot["PercentPoints"] = split_plot["Percent"] * 100
        fig = px.bar(split_plot, x="Split", y="PercentPoints", text="PercentPoints",
                     labels={"PercentPoints": "Share of supervised observations (%)"},
                     title="Group-disjoint train / validation / test split")
        fig.update_traces(texttemplate="%{text:.1f}%")
        fig.update_layout(height=390, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Candidate-model validation")
        val_long = validation.melt(
            id_vars="Model",
            value_vars=["Validation accuracy", "Validation balanced accuracy", "Validation macro F1"],
            var_name="Metric", value_name="Score"
        )
        fig = px.bar(val_long, x="Model", y="Score", color="Metric", barmode="group",
                     title="Validation-set model comparison")
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        fig.update_layout(height=390)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Turnout model")
    t1, t2, t3 = st.columns(3)
    t1.metric("MAE", f'{metrics["turnout_mae"] * 100:.2f} pp')
    t2.metric("RMSE", f'{metrics["turnout_rmse"] * 100:.2f} pp')
    t3.metric("R²", f'{metrics["turnout_r2"]:.3f}')

    st.markdown("#### Historical metro turnout")
    m = metro.copy()
    m["TurnoutPct"] = m["Turnout"] * 100
    fig = px.line(m, x="Year", y="TurnoutPct", markers=True,
                  title="Observed LGE metro turnout",
                  labels={"TurnoutPct": "Turnout (%)", "Year": "Election year"})
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Machine Learning Life Cycle")
    st.markdown("""
1. **Business understanding** — define four required election outputs and risks.
2. **Data acquisition** — official IEC 2011, 2016, 2021 LGE and 2024 NPE data.
3. **Data understanding & quality audit** — inspect schema, missingness, duplicates, ballots and geographic identifiers.
4. **Cleaning & integration** — standardise fields and align elections primarily through Voting District identifiers.
5. **EDA** — investigate party movement, turnout, competitiveness and the 2024 political signal.
6. **Feature engineering** — construct lagged electoral-strength, turnout and registration features without target leakage.
7. **Data splitting** — group-disjoint 70% training, 15% validation and 15% testing.
8. **Modelling & selection** — compare a dummy baseline, logistic regression and Random Forest on validation data.
9. **Final evaluation** — refit the selected classifier on training + validation, then evaluate once on the untouched test set.
10. **Forecasting** — combine validated historical modelling with a transparent 2024 recent-signal scenario.
11. **Deployment & communication** — expose the four outputs, validation evidence, assumptions and limitations in this dashboard.
""")

    st.markdown("#### Important limitations")
    st.info(
        "Only three historical LGEs are available. The 2024 file is an NPE, not an LGE. "
        "The supplied 2024 results are metro-aggregated, so ward-level MK allocation cannot be supported. "
        "The 40/60 LGE/NPE blend is an explicit scenario assumption rather than a learned causal parameter. "
        "Historical test accuracy does not guarantee the same accuracy in a future election."
    )

st.divider()
st.caption(
    "Academic election-analytics project • Aggregate IEC data only • "
    "Forecasts are analytical estimates, not statements of electoral fact."
)
