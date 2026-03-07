import pandas as pd
import numpy as np
import joblib
import os

def predict_match(match_details):
    # Load model and encoders
    model_path = 'models/cricket_model.pkl'
    feature_names_path = 'data/feature_names.pkl'
    le_path = 'data/label_encoder.pkl'
    
    if not os.path.exists(model_path):
        print("Model not found. Please run trainer.py first.")
        return

    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    le = joblib.load(le_path)
    
    # Create input DataFrame
    df_input = pd.DataFrame([match_details])
    
    # One-hot encoding for the input (needs to match training columns)
    categorical_cols = ['venue', 'pitch_type', 'weather', 'toss_winner', 'toss_decision']
    df_input_encoded = pd.get_dummies(df_input, columns=categorical_cols)
    
    # Ensure all training features are present (even if not in this single input)
    final_input = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in df_input_encoded.columns:
            final_input[col] = df_input_encoded[col]
        else:
            final_input[col] = 0 # Default for missing one-hot columns
            
    # Predict
    probs = model.predict_proba(final_input)[0]
    classes = le.classes_ # ['India', 'New Zealand']
    
    print("\n--- MATCH PREDICTION ---")
    print(f"Match: India vs New Zealand (World Cup Final)")
    print(f"Venue: {match_details['venue']} | Pitch: {match_details['pitch_type']}")
    print("-" * 25)
    for i, team in enumerate(classes):
        print(f"{team} Win Probability: {probs[i]*100:.2f}%")
    
    winner_idx = np.argmax(probs)
    print("-" * 25)
    print(f"MODEL PREDICTION: {classes[winner_idx]} is likely to win!")

if __name__ == "__main__":
    # Example input for tomorrow's game
    # Let's assume the final is in a neutral venue (e.g., Lords)
    tomorrow_match = {
        'venue': 'Neutral',
        'pitch_type': 'Balanced',
        'weather': 'Sunny',
        'toss_winner': 'India',
        'toss_decision': 'Bat',
        'ind_form': 0.85, # India in great form
        'nz_form': 0.75   # NZ also in good form
    }
    
    predict_match(tomorrow_match)
