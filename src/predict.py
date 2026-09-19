import numpy as np
import pandas as pd
import joblib
import os
import time

TOP_CLUBS = {
    'manchester united', 'real madrid', 'manchester city', 'fc barcelona', 'barcelona',
    'bayern munich', 'paris saint-germain', 'psg', 'liverpool fc', 'liverpool',
    'chelsea fc', 'chelsea', 'arsenal fc', 'arsenal', 'juventus', 'inter milan',
    'ac milan', 'atletico madrid', 'tottenham hotspur', 'borussia dortmund', 'napoli'
}

class MarketValuePredictor:
    """
    Real-Time Inference Engine for Football Player Market Value Prediction.
    Loads saved model, feature scaler, and feature list artifacts.
    """
    def __init__(self, model_dir='models'):
        model_path = os.path.join(model_dir, 'best_model.pkl')
        scaler_path = os.path.join(model_dir, 'feature_scaler.pkl')
        cols_path = os.path.join(model_dir, 'feature_columns.json')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file missing at: {model_path}. Run pipeline first.")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_cols = joblib.load(cols_path)

    def predict_single_player(self, player_dict):
        """
        Predicts market value for a single player dictionary input.
        Returns predicted market value in Euros, log value, confidence interval, and inference latency.
        """
        t0 = time.time()

        # Extract basic input fields with safe defaults
        age = float(player_dict.get('age', 25))
        height = float(player_dict.get('height', 182))
        appearance = float(player_dict.get('appearance', 30))
        goals = float(player_dict.get('goals', 5))
        assists = float(player_dict.get('assists', 4))
        yellow_cards = float(player_dict.get('yellow cards', 3))
        second_yellow_cards = float(player_dict.get('second yellow cards', 0))
        red_cards = float(player_dict.get('red cards', 0))
        goals_conceded = float(player_dict.get('goals conceded', 0))
        clean_sheets = float(player_dict.get('clean sheets', 0))
        minutes_played = float(player_dict.get('minutes played', 2200))
        days_injured = float(player_dict.get('days_injured', 15))
        games_injured = float(player_dict.get('games_injured', 2))
        award = float(player_dict.get('award', 1))
        position_encoded = float(player_dict.get('position_encoded', 3))
        winger = float(player_dict.get('winger', 0))
        pos_group = str(player_dict.get('position_group', 'Midfielder'))
        team_name = str(player_dict.get('team', ''))

        # Feature Engineering calculations
        minutes_safe = max(minutes_played, 1.0)
        apps_safe = max(appearance, 1.0)

        is_top_c = 1 if any(c in team_name.lower() for c in TOP_CLUBS) else 0

        goals_per_90 = (goals / minutes_safe) * 90.0
        assists_per_90 = (assists / minutes_safe) * 90.0
        goal_contrib_per_90 = goals_per_90 + assists_per_90
        clean_sheets_per_game = clean_sheets / apps_safe
        goals_conceded_per_90 = (goals_conceded / minutes_safe) * 90.0
        minutes_per_game = minutes_played / apps_safe

        discipline_score = yellow_cards + (2.0 * second_yellow_cards) + (3.0 * red_cards)
        injury_days_per_game = days_injured / apps_safe
        games_injured_ratio = games_injured / apps_safe

        age_squared = age ** 2
        age_cubed = age ** 3
        age_prime_dist = abs(age - 26.0)
        is_prime_age = 1 if (24 <= age <= 28) else 0

        pos_Attacker = 1 if pos_group == 'Attacker' else 0
        pos_Defender = 1 if pos_group == 'Defender' else 0
        pos_Goalkeeper = 1 if pos_group == 'Goalkeeper' else 0
        pos_Midfielder = 1 if pos_group == 'Midfielder' else 0

        feature_map = {
            'age': age,
            'height': height,
            'appearance': appearance,
            'goals': goals,
            'assists': assists,
            'yellow cards': yellow_cards,
            'second yellow cards': second_yellow_cards,
            'red cards': red_cards,
            'goals conceded': goals_conceded,
            'clean sheets': clean_sheets,
            'minutes played': minutes_played,
            'days_injured': days_injured,
            'games_injured': games_injured,
            'award': award,
            'position_encoded': position_encoded,
            'winger': winger,
            'is_top_club': is_top_c,
            'goals_per_90': goals_per_90,
            'assists_per_90': assists_per_90,
            'goal_contrib_per_90': goal_contrib_per_90,
            'clean_sheets_per_game': clean_sheets_per_game,
            'goals_conceded_per_90': goals_conceded_per_90,
            'minutes_per_game': minutes_per_game,
            'discipline_score': discipline_score,
            'injury_days_per_game': injury_days_per_game,
            'games_injured_ratio': games_injured_ratio,
            'age_squared': age_squared,
            'age_cubed': age_cubed,
            'age_prime_dist': age_prime_dist,
            'is_prime_age': is_prime_age,
            'pos_Attacker': pos_Attacker,
            'pos_Defender': pos_Defender,
            'pos_Goalkeeper': pos_Goalkeeper,
            'pos_Midfielder': pos_Midfielder
        }

        # Align with model input features list
        row_vector = [feature_map.get(col, 0.0) for col in self.feature_cols]
        X_input = pd.DataFrame([row_vector], columns=self.feature_cols)

        # Scale and predict
        X_scaled = self.scaler.transform(X_input)
        pred_log = float(self.model.predict(X_scaled)[0])
        pred_raw = float(np.expm1(pred_log))
        pred_raw = max(pred_raw, 100_000.0)

        latency_ms = (time.time() - t0) * 1000.0

        # Estimated 90% confidence bounds
        lower_bound = max(pred_raw * 0.78, 100_000.0)
        upper_bound = pred_raw * 1.25

        return {
            'predicted_value_eur': round(pred_raw, 2),
            'predicted_value_formatted': f"EUR {pred_raw / 1e6:.2f}M" if pred_raw >= 1e6 else f"EUR {pred_raw / 1e3:.0f}K",
            'lower_bound_eur': round(lower_bound, 2),
            'upper_bound_eur': round(upper_bound, 2),
            'latency_ms': round(latency_ms, 2),
            'features_used': feature_map
        }
