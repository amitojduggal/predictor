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

# --- AUTO-DERIVE METRICS FROM RECENT RESEARCH ---
def get_auto_metrics():
    return {
        "ind_overall": 0.85, 
        "nz_overall": 0.70,  
        "ind_h2h": 0.80,     
        "nz_h2h": 0.20,
        "ind_matchup": 0.90, 
        "nz_matchup": 0.85,   
        "pundit_sentiment": 0.68,
        "betting_market": 0.71,
        "crowd_impact": 0.10,
        "venue_curse": -0.05,
        "umpire_factor": 0.05 # +5% for India because Kettleborough is NOT officiating
    }

auto_data = get_auto_metrics()

st.sidebar.header("📊 Data Source")
mode = st.sidebar.radio("Input Mode", ["Auto-Sync (Recent Games)", "Manual Adjust"])

if mode == "Auto-Sync (Recent Games)":
    st.sidebar.success("✅ All parameters synced with March 2026 Reality.")
    market_lean = auto_data["betting_market"]
    crowd_vol = auto_data["crowd_impact"]
    venue_factor = auto_data["venue_curse"]
    pundit_weight = auto_data["pundit_sentiment"]
    umpire_factor = auto_data["umpire_factor"]
else:
    st.sidebar.header("📈 Betting & Market Sentiment")
    market_lean = st.sidebar.slider("Market Confidence (India)", 0.0, 1.0, auto_data["betting_market"])

    st.sidebar.header("🔊 Out-of-the-Box Factors")
    crowd_vol = st.sidebar.slider("Crowd 'Volume' Impact", 0.0, 0.2, auto_data["crowd_impact"])
    venue_factor = st.sidebar.slider("Psychological Venue Factor", -0.1, 0.1, auto_data["venue_curse"])
    umpire_factor = st.sidebar.slider("Umpire/Omen Factor", -0.1, 0.1, auto_data["umpire_factor"], help="Positive: Good omens for India")

    st.sidebar.header("🎙️ Expert Sentiment")
    pundit_weight = st.sidebar.slider("Pundit Lean (India Favored)", 0.0, 1.0, auto_data["pundit_sentiment"])

col1, col2, col3 = st.columns([1.2, 1.2, 1.5])

with col1:
    st.header("🏟️ Match & Pitch Setup")
    venue = st.selectbox("Venue Location", ["India", "Neutral", "New Zealand"], index=0)
    is_wc = st.checkbox("World Cup Final Pressure?", value=True)
    
    st.divider()
    st.subheader("Pitch Condition")
    pitch_type = st.radio("Ahmedabad Pitch Type", ["Pace-Friendly (Red/Mixed Soil)", "Spin-Friendly (Black Soil)"])
    pitch_is_pace_friendly = 1 if "Pace" in pitch_type else 0

    st.divider()
    st.subheader("🪙 Toss & Dew Factor")
    toss_winner = st.radio("Who wins the toss?", ["India", "New Zealand"])
    toss_decision = st.radio("Decision?", ["Bowl First (Chase)", "Bat First (Defend)"])
    
    ind_toss_adv = 1 if (toss_winner == "India" and toss_decision == "Bowl First (Chase)") else 0
    nz_toss_adv = 1 if (toss_winner == "New Zealand" and toss_decision == "Bowl First (Chase)") else 0

with col2:
    st.header("🔥 Player Matchups & Form")
    
    if mode == "Auto-Sync (Recent Games)":
        ind_matchup = auto_data["ind_matchup"]
        nz_matchup = auto_data["nz_matchup"]
        ind_overall = auto_data["ind_overall"]
        nz_overall = auto_data["nz_overall"]
        ind_h2h = auto_data["ind_h2h"]
        nz_h2h = auto_data["nz_h2h"]
        
        st.metric("India Player Form", f"{ind_matchup*100}%")
        st.metric("NZ Player Form", f"{nz_matchup*100}%")
        st.divider()
        st.info(f"**Market Lean:** {market_lean*100}%")
        st.info(f"**Umpire Omen:** {'Positive' if umpire_factor > 0 else 'Neutral'}")
    else:
        st.subheader("India Matchup Index")
        ind_matchup = st.slider("India Key Players Form", 0.0, 1.0, 0.85)
        st.subheader("New Zealand Matchup Index")
        nz_matchup = st.slider("NZ Key Players Form", 0.0, 1.0, 0.80)

    st.divider()
    st.subheader("Team Momentum")
    if mode == "Manual Adjust":
        ind_overall = st.slider("India Overall Win Rate (%)", 0, 100, 80) / 100.0
        nz_overall = st.slider("NZ Overall Win Rate (%)", 0, 100, 60) / 100.0
        ind_h2h = st.slider("India H2H Win Rate (%)", 0, 100, 75) / 100.0
        nz_h2h = st.slider("NZ H2H Win Rate (%)", 0, 100, 25) / 100.0
    else:
        st.write(f"**India Momentum:** {ind_overall*100}%")
        st.write(f"**NZ Momentum:** {nz_overall*100}%")

with col3:
    st.header("📊 Predictive Analytics")
    
    if st.button("RUN HYBRID INTELLIGENCE SIMULATION", type="primary", use_container_width=True):
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
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[feature_names]
        
        probs = model.predict_proba(X_input)[0]

        # --- HYBRID BLENDING ---
        stat_prob = probs[0]
        # Aggregate out-of-the-box psychological factor
        psych_factor = crowd_vol + venue_factor + umpire_factor
        
        # Weighted blend
        india_prob = (stat_prob * 0.45) + (market_lean * 0.25) + (pundit_weight * 0.15) + ((0.5 + psych_factor) * 0.15)
        india_prob = max(0.1, min(0.96, india_prob))
        nz_prob = 1.0 - india_prob

        # --- SCORELINE PREDICTION ---
        base_score = 195 
        pitch_boost = 15 if pitch_is_pace_friendly else 0
        pressure_penalty = -10 if is_wc else 0
        ind_score_adj = (ind_matchup - 0.5) * 50 + (ind_overall - 0.5) * 30
        nz_score_adj = (nz_matchup - 0.5) * 50 + (nz_overall - 0.5) * 30
        projected_ind = base_score + pitch_boost + pressure_penalty + ind_score_adj
        projected_nz = base_score + pitch_boost + pressure_penalty + nz_score_adj

        if (toss_winner == "India" and toss_decision == "Bat First (Defend)") or \
           (toss_winner == "New Zealand" and toss_decision == "Bowl First (Chase)"):
            first_bat, second_bat = "India", "New Zealand"
            score_1, score_2 = projected_ind, projected_nz
        else:
            first_bat, second_bat = "New Zealand", "India"
            score_1, score_2 = projected_nz, projected_ind

        if (india_prob > nz_prob and second_bat == "India") or (nz_prob > india_prob and second_bat == "New Zealand"):
            score_2 = score_1 + 2
        else:
            score_2 = min(score_2, score_1 - 5)

        st.markdown("### Match Win Probability")
        res1, res2 = st.columns(2)
        res1.metric("🇮🇳 INDIA", f"{india_prob*100:.1f}%")
        res2.metric("🇳🇿 NEW ZEALAND", f"{nz_prob*100:.1f}%")
        st.progress(india_prob, text="Hybrid Confidence Level")

        st.divider()
        st.header("🎯 Projected Scoreline")
        s1, s2 = st.columns(2)
        s1.subheader(f"1st Innings: {first_bat}")
        s1.markdown(f"## {int(score_1)} / 6")
        s2.subheader(f"2nd Innings: {second_bat}")
        s2.markdown(f"## {int(score_2)} / 4")
        
        st.divider()
        st.header("🌟 Projected Key Performers")
        kp1, kp2 = st.columns(2)
        with kp1:
            st.subheader("🇮🇳 India")
            st.write(f"**SKY:** {int(55 + (ind_matchup * 20))} runs")
            st.write(f"**Samson:** {int(45 + (ind_matchup * 15))} runs")
            st.write(f"**Bumrah:** {'3 wkts' if pitch_is_pace_friendly else '2 wkts'}")
        with kp2:
            st.subheader("🇳🇿 NZ")
            st.write(f"**Finn Allen:** {int(65 + (nz_matchup * 25))} runs")
            st.write(f"**Rachin:** 2 wkts")
            st.write(f"**Ferguson:** 2 wkts")

        st.divider()
        if india_prob > nz_prob:
            st.success("#### 🏆 PROJECTION: INDIA FAVORED")
            st.write(f"Model Confidence: {india_prob*100:.1f}%. Factors: Home Crowd (+10%), Market Lean (71%), and 'Richard Kettleborough Absent' omen (+5%).")
        else:
            st.warning("#### 🏆 PROJECTION: NEW ZEALAND FAVORED")
            st.write(f"Model Confidence: {nz_prob*100:.1f}%. Factors: NZ World Cup Hex (3-0), Finn Allen peak form, and disciplined underdog betting flow.")

st.markdown("---")
st.caption("Hybrid Model as of March 7, 2026. Data includes match history, betting sentiment, and psychological omens.")
