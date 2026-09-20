import os
import json
import joblib
import numpy as np
import pandas as pd

def main():
    print("--- Exporting Ultra-Slim Model & Scaler Artifacts for Vercel ---")

    model_path = 'models/best_model.pkl'
    scaler_path = 'models/feature_scaler.pkl'
    cols_path = 'models/feature_columns.json'

    if not os.path.exists(model_path):
        raise FileNotFoundError("models/best_model.pkl not found. Run pipeline first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    cols = joblib.load(cols_path)

    # Dump LightGBM trees to JSON dict
    lgbm_dict = model.booster_.dump_model()

    # Save tree structure JSON
    with open('models/lightgbm_model.json', 'w') as f:
        json.dump(lgbm_dict, f)

    # Save scaler parameters JSON
    scaler_dict = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    with open('models/scaler_params.json', 'w') as f:
        json.dump(scaler_dict, f)

    # Save lightweight players dataset sample for fast UI search
    if os.path.exists('data/processed_players.csv'):
        df = pd.read_csv('data/processed_players.csv')
    elif os.path.exists('final_data.csv'):
        df = pd.read_csv('final_data.csv')
    else:
        df = pd.DataFrame()

    if not df.empty:
        slim_df = df[['name', 'team', 'position', 'age', 'goals', 'assists', 'appearance', 'minutes played', 'current_value', 'highest_value']].head(150)
        slim_df.to_json('models/players_slim.json', orient='records')

    print("[SUCCESS] Ultra-slim artifacts exported:")
    print(" -> models/lightgbm_model.json")
    print(" -> models/scaler_params.json")
    print(" -> models/players_slim.json")

if __name__ == '__main__':
    main()
