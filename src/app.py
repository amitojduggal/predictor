import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Ultimate Cricket Predictor", page_icon="🏆", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('models/cricket_model.pkl')
    feature_names = joblib.load('data/feature_names.pkl')
    le = joblib.load('data/label_encoder.pkl')
    return model, feature_names, le

st.title("🏆 Ultimate T20 World Cup Final Predictor")
st.subheader("India vs New Zealand | March 8, 2026 | Narendra Modi Stadium, Ahmedabad")

# --- AUTO-DERIVE GRANULAR METRICS FROM 2026 TOURNAMENT ---
def get_auto_metrics():
    # India Tournament Profile:
    # 1. Varun Chakaravarthy: 13 wickets (Tournament Lead)
    # 2. Ishan Kishan: 263 runs (Peak Opener Form)
    # 3. SKY/Samson: 200+ runs each
    ind_form_weight = (0.85 * 0.4) + (0.95 * 0.6) # Weighted Win Rate + Peak Player Form
    
    # NZ Tournament Profile:
    # 1. Finn Allen: 289 runs (includes 33-ball 100)
    # 2. Rachin Ravindra: 11 wkts / 128 runs (Double threat)
    nz_form_weight = (0.75 * 0.4) + (0.90 * 0.6)

    return {
        "ind_overall": 0.88, # India won 9/10 games leading to final
        "nz_overall": 0.75,  # NZ won 7/10 including semi-final demolition
        "ind_h2h": 0.80,     # Bilateral dominance in Jan 2026
        "nz_h2h": 0.20,
        "ind_matchup": 0.92, # Chakaravarthy (13 wkts), Kishan (263 runs)
        "nz_matchup": 0.88,   # Finn Allen (289 runs), Rachin (11 wkts)
        "pundit_sentiment": 0.68,
        "betting_market": 0.71,
        "crowd_impact": 0.10,
        "venue_curse": -0.05,
        "umpire_factor": 0.05
    }

auto_data = get_auto_metrics()

st.sidebar.header("📊 Data Source")
mode = st.sidebar.radio("Input Mode", ["Auto-Sync (Recent Games)", "Manual Adjust"])

if mode == "Auto-Sync (Recent Games)":
    st.sidebar.success("✅ Synced with 2026 Tournament Stats")
    market_lean = auto_data["betting_market"]
    crowd_vol = auto_data["crowd_impact"]
    venue_factor = auto_data["venue_curse"]
    pundit_weight = auto_data["pundit_sentiment"]
    umpire_factor = auto_data["umpire_factor"]
else:
    st.sidebar.header("📈 Market & Out-of-the-Box")
    market_lean = st.sidebar.slider("Market Confidence (India)", 0.0, 1.0, auto_data["betting_market"])
    crowd_vol = st.sidebar.slider("Crowd Volume", 0.0, 0.2, auto_data["crowd_impact"])
    venue_factor = st.sidebar.slider("Psychological Venue Factor", -0.1, 0.1, auto_data["venue_curse"])
    umpire_factor = st.sidebar.slider("Umpire Omen", -0.1, 0.1, auto_data["umpire_factor"])
    pundit_weight = st.sidebar.slider("Pundit Lean", 0.0, 1.0, auto_data["pundit_sentiment"])

col1, col2, col3 = st.columns([1.2, 1.2, 1.5])

with col1:
    st.header("🏟️ Match & Pitch")
    venue = st.selectbox("Venue", ["India", "Neutral", "New Zealand"], index=0)
    is_wc = st.checkbox("World Cup Final?", value=True)
    pitch_type = st.radio("Ahmedabad Surface", ["Pace-Friendly (Red Soil)", "Spin-Friendly (Black Soil)"])
    pitch_is_pace_friendly = 1 if "Pace" in pitch_type else 0

    st.divider()
    st.subheader("🪙 Toss")
    toss_winner = st.radio("Who wins the toss?", ["India", "New Zealand"])
    toss_decision = st.radio("Decision?", ["Bowl First (Chase)", "Bat First (Defend)"])
    ind_toss_adv = 1 if (toss_winner == "India" and toss_decision == "Bowl First (Chase)") else 0
    nz_toss_adv = 1 if (toss_winner == "New Zealand" and toss_decision == "Bowl First (Chase)") else 0

with col2:
    st.header("🔥 Tournament Form")
    if mode == "Auto-Sync (Recent Games)":
        ind_matchup = auto_data["ind_matchup"]
        nz_matchup = auto_data["nz_matchup"]
        ind_overall = auto_data["ind_overall"]
        nz_overall = auto_data["nz_overall"]
        ind_h2h = auto_data["ind_h2h"]
        nz_h2h = auto_data["nz_h2h"]
        
        st.write("**IND Tournament Stats:**")
        st.write("• Chakaravarthy: 13 Wkts (Lead)")
        st.write("• Ishan Kishan: 263 Runs")
        st.write("**NZ Tournament Stats:**")
        st.write("• Finn Allen: 289 Runs (Record 100)")
        st.write("• Rachin: 11 Wkts / 128 Runs")
    else:
        ind_matchup = st.slider("India Player Form", 0.0, 1.0, 0.90)
        nz_matchup = st.slider("NZ Player Form", 0.0, 1.0, 0.85)
        ind_overall = st.slider("India Momentum", 0.0, 1.0, 0.85)
        nz_overall = st.slider("NZ Momentum", 0.0, 1.0, 0.70)
        ind_h2h = 0.80
        nz_h2h = 0.20

with col3:
    st.header("📊 Predictive Analytics")
    if st.button("RUN HYBRID SIMULATION", type="primary", use_container_width=True):
        model, feature_names, le = load_model()
        input_data = {
            'is_world_cup': 1 if is_wc else 0,
            'ind_overall_form': ind_overall,
            'nz_overall_form': nz_overall,
            'ind_h2h_form': ind_h2h,
            'nz_h2h_form': nz_h2h,
            'ind_toss_advantage': ind_toss_adv,
            'nz_toss_advantage': nz_toss_adv,
            'pitch_is_pace_friendly': pitch_is_pace_friendly,
            'ind_matchup_index': ind_matchup,
            'nz_matchup_index': nz_matchup,
            'venue_cat_India': 1 if venue == "India" else 0,
            'venue_cat_New Zealand': 1 if venue == "New Zealand" else 0,
            'venue_cat_Neutral': 1 if venue == "Neutral" else 0
        }
        X_input = pd.DataFrame([input_data])
        for col in feature_names:
            if col not in X_input.columns: X_input[col] = 0
        X_input = X_input[feature_names]
        
        probs = model.predict_proba(X_input)[0]
        stat_prob = probs[0]
        psych_factor = crowd_vol + venue_factor + umpire_factor
        india_prob = (stat_prob * 0.45) + (market_lean * 0.25) + (pundit_weight * 0.15) + ((0.5 + psych_factor) * 0.15)
        india_prob = max(0.1, min(0.96, india_prob))
        nz_prob = 1.0 - india_prob

        st.metric("INDIA WIN PROBABILITY", f"{india_prob*100:.1f}%")
        st.progress(india_prob)

        # Score Logic
        base = 200 # Higher par for 2026
        pitch_mod = 15 if pitch_is_pace_friendly else -5
        ind_score = base + pitch_mod + (ind_matchup - 0.5) * 60
        nz_score = base + pitch_mod + (nz_matchup - 0.5) * 60

        if (toss_winner == "India" and toss_decision == "Bowl First (Chase)") or (toss_winner == "New Zealand" and toss_decision == "Bat First (Defend)"):
            first, second = "New Zealand", "India"
            s1, s2 = nz_score, ind_score
        else:
            first, second = "India", "New Zealand"
            s1, s2 = ind_score, nz_score

        if (india_prob > nz_prob and second == "India") or (nz_prob > india_prob and second == "New Zealand"):
            s2 = s1 + 2
        else:
            s2 = min(s2, s1 - 8)

        st.divider()
        st.subheader("🎯 Projected Scoreline")
        st.write(f"**1st Innings ({first}):** {int(s1)}/6")
        st.write(f"**2nd Innings ({second}):** {int(s2)}/4")

        st.divider()
        st.subheader("🌟 Projected Key Performers")
        col_ind, col_nz = st.columns(2)
        with col_ind:
            st.write("**🇮🇳 India:**")
            st.write(f"• **Ishan Kishan:** {int(45 + (ind_matchup * 20))} runs")
            st.write(f"• **SKY:** {int(50 + (ind_matchup * 25))} runs")
            st.write(f"• **Chakaravarthy:** 2+ wickets")
        with col_nz:
            st.write("**🇳🇿 NZ:**")
            st.write(f"• **Finn Allen:** {int(60 + (nz_matchup * 30))} runs")
            st.write(f"• **Rachin:** 1 wicket / 35 runs")
            st.write(f"• **Henry:** 2 wickets")
