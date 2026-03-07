import pandas as pd
import numpy as np
import os

def generate_mock_data(n_matches=100):
    np.random.seed(42)
    
    venues = ['India', 'New Zealand', 'Neutral']
    pitch_types = ['Spin', 'Pace', 'Balanced']
    weather_conditions = ['Sunny', 'Cloudy']
    teams = ['India', 'New Zealand']
    
    data = []
    
    for _ in range(n_matches):
        venue = np.random.choice(venues)
        pitch = np.random.choice(pitch_types)
        weather = np.random.choice(weather_conditions)
        
        # Toss
        toss_winner = np.random.choice(teams)
        toss_decision = np.random.choice(['Bat', 'Bowl'])
        
        # Form (normalized win rate in last 5 matches)
        ind_form = np.random.uniform(0.4, 0.9)
        nz_form = np.random.uniform(0.4, 0.9)
        
        # Base probabilities
        # India is stronger at home and on spin tracks
        # NZ is stronger at home and on pace tracks
        prob_india = 0.5 
        
        if venue == 'India': prob_india += 0.15
        if venue == 'New Zealand': prob_india -= 0.15
        
        if pitch == 'Spin': prob_india += 0.1
        if pitch == 'Pace': prob_india -= 0.1
        
        prob_india += (ind_form - nz_form) * 0.5
        
        # Ensure probability is within [0, 1]
        prob_india = max(0.1, min(0.9, prob_india))
        
        winner = 'India' if np.random.random() < prob_india else 'New Zealand'
        
        data.append({
            'venue': venue,
            'pitch_type': pitch,
            'weather': weather,
            'toss_winner': toss_winner,
            'toss_decision': toss_decision,
            'ind_form': round(ind_form, 2),
            'nz_form': round(nz_form, 2),
            'winner': winner
        })
        
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/matches.csv', index=False)
    print(f"Generated {n_matches} mock matches in data/matches.csv")

if __name__ == "__main__":
    generate_mock_data()
