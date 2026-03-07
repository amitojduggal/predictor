# Ultimate Cricket Predictor: T20 World Cup 2026 Final 🏏

A sophisticated **Hybrid Intelligence System** designed to predict the outcome of the high-stakes T20 World Cup Final between **India and New Zealand** at the Narendra Modi Stadium, Ahmedabad (March 8, 2026).

This project moves beyond basic statistics by blending **Machine Learning (Random Forest)** with real-time **Sentiment Analysis**, **Betting Market Flow**, and **Psychological "Omen" Factors**.

## 🚀 Key Features

*   **Hybrid Predictive Engine:** Combines historical T20I match data (2007–2026) with human-centric variables.
*   **Auto-Sync 2026 Intelligence:** Automatically pulls the latest form metrics, including India's 2026 dominant run and Finn Allen’s record-breaking 33-ball century.
*   **Situational Analytics:** Factors in the **Ahmedabad Red-Soil Pitch** dynamics and the **Evening Dew Advantage** (Toss impact).
*   **"Out-of-the-Box" Parameters:**
    *   **Betting Market Sentiment:** Blends global odds (2/5 for India) into the win probability.
    *   **Crowd Adrenaline:** Simulates the impact of 132,000 home fans (+10% morale boost).
    *   **Umpire/Omen Logic:** Factors in the "Richard Kettleborough Absence" (+5% India favorability).
*   **Dynamic Scoreline Projection:** Generates predicted 1st and 2nd innings scores and identifies "Key Performers" (e.g., SKY vs. Finn Allen).

## 🛠️ Tech Stack

*   **Python 3.9+**
*   **Streamlit:** Interactive Web Dashboard.
*   **Scikit-Learn:** Random Forest Ensemble Model.
*   **Pandas & NumPy:** Data Engineering and Feature Scaling.
*   **Joblib:** Model persistence.

## 📦 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/cricket-predictor-2026.git
   cd cricket-predictor-2026
   ```

2. **Install dependencies:**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. **Train the model:**
   ```bash
   python3 src/trainer.py
   ```

4. **Launch the Dashboard:**
   ```bash
   # Run from the project root. The imghdr.py shim fixes compatibility on Python 3.13+
   python3 -m streamlit run src/app.py
   ```

## 🧠 Model Logic: The Hybrid Approach

The "Win Confidence" is calculated using a weighted blend:
*   **50% Match History:** Pure statistical head-to-head and venue performance.
*   **20% Betting Market:** Real-world financial risk assessment from global sportsbooks.
*   **15% Pundit Sentiment:** Expert opinions from legends like Dale Steyn and Brad Haddin.
*   **15% Psychological Factors:** Crowd volume, venue trauma (2023), and umpire omens.

---
*Disclaimer: This tool is for educational and analytical purposes. Cricket is a game of glorious uncertainties!*
