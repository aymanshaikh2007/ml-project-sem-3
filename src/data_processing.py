import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

def map_position_group(pos_str):
    """Categorize detailed position strings into 4 standard football position groups."""
    if not isinstance(pos_str, str):
        return 'Midfielder'
    pos = pos_str.lower()
    if 'goalkeeper' in pos:
        return 'Goalkeeper'
    elif 'defender' in pos or 'back' in pos:
        return 'Defender'
    elif 'midfield' in pos or 'midfielder' in pos:
        return 'Midfielder'
    elif 'attack' in pos or 'winger' in pos or 'striker' in pos or 'forward' in pos or 'second striker' in pos:
        return 'Attacker'
    return 'Midfielder'

# Top European Elite Clubs
TOP_CLUBS = {
    'manchester united', 'real madrid', 'manchester city', 'fc barcelona', 'barcelona',
    'bayern munich', 'paris saint-germain', 'psg', 'liverpool fc', 'liverpool',
    'chelsea fc', 'chelsea', 'arsenal fc', 'arsenal', 'juventus', 'inter milan',
    'ac milan', 'atletico madrid', 'tottenham hotspur', 'borussia dortmund', 'napoli'
}

def is_top_club(team_name):
    if not isinstance(team_name, str):
        return 0
    t = team_name.lower().strip()
    return 1 if any(club in t for club in TOP_CLUBS) else 0

def load_and_preprocess_data(csv_path):
    """
    Loads raw CSV data, performs data cleaning, handles missing values & duplicates,
    executes feature engineering, verifies data leakage isolation, and performs
    train/val/test split.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    initial_rows = len(df)

    # 1. Cleaning & Duplicates
    df = df.drop_duplicates(subset=['player']).reset_index(drop=True)

    # Clean target variable
    df = df[df['current_value'].notnull() & (df['current_value'] > 0)].copy()
    valid_rows = len(df)

    # Fill numerical missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical missing values
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')

    # 2. Feature Engineering
    # Target transformation (log1p to handle right-skewed market values)
    df['log_current_value'] = np.log1p(df['current_value'])

    # Position macro grouping
    df['position_group'] = df['position'].apply(map_position_group)

    # Team tier indicator
    df['is_top_club'] = df['team'].apply(is_top_club)

    # Per-90 statistics & workload ratios
    minutes_safe = np.maximum(df['minutes played'], 1.0)
    apps_safe = np.maximum(df['appearance'], 1.0)

    df['goals_per_90'] = (df['goals'] / minutes_safe) * 90.0
    df['assists_per_90'] = (df['assists'] / minutes_safe) * 90.0
    df['goal_contrib_per_90'] = df['goals_per_90'] + df['assists_per_90']
    df['clean_sheets_per_game'] = df['clean sheets'] / apps_safe
    df['goals_conceded_per_90'] = (df['goals conceded'] / minutes_safe) * 90.0
    df['minutes_per_game'] = df['minutes played'] / apps_safe

    # Discipline & Injury indicators
    df['discipline_score'] = df['yellow cards'] + (2.0 * df['second yellow cards']) + (3.0 * df['red cards'])
    df['injury_days_per_game'] = df['days_injured'] / apps_safe
    df['games_injured_ratio'] = df['games_injured'] / apps_safe

    # Age non-linear polynomial features & prime age distance
    df['age_squared'] = df['age'] ** 2
    df['age_cubed'] = df['age'] ** 3
    df['age_prime_dist'] = np.abs(df['age'] - 26.0)
    df['is_prime_age'] = df['age'].between(24, 28).astype(int)

    # One-hot encoding for position group
    pos_dummies = pd.get_dummies(df['position_group'], prefix='pos', drop_first=False).astype(int)
    for col in ['pos_Attacker', 'pos_Defender', 'pos_Goalkeeper', 'pos_Midfielder']:
        if col not in pos_dummies.columns:
            pos_dummies[col] = 0
    df = pd.concat([df, pos_dummies], axis=1)

    # 3. Data Leakage Prevention Check
    # We explicitly EXCLUDE 'highest_value' and 'player' / 'name' URL strings from model feature matrix X!
    feature_cols = [
        'age', 'height', 'appearance', 'goals', 'assists', 'yellow cards',
        'second yellow cards', 'red cards', 'goals conceded', 'clean sheets',
        'minutes played', 'days_injured', 'games_injured', 'award',
        'position_encoded', 'winger', 'is_top_club', 'goals_per_90', 'assists_per_90',
        'goal_contrib_per_90', 'clean_sheets_per_game', 'goals_conceded_per_90',
        'minutes_per_game', 'discipline_score', 'injury_days_per_game',
        'games_injured_ratio', 'age_squared', 'age_cubed', 'age_prime_dist',
        'is_prime_age', 'pos_Attacker', 'pos_Defender', 'pos_Goalkeeper', 'pos_Midfielder'
    ]

    X = df[feature_cols].copy()
    y_log = df['log_current_value'].values
    y_raw = df['current_value'].values

    # 4. Train / Validation / Test Split (70% Train, 15% Validation, 15% Test)
    X_train_val, X_test, y_train_val_log, y_test_log, y_train_val_raw, y_test_raw = train_test_split(
        X, y_log, y_raw, test_size=0.15, random_state=42
    )

    X_train, X_val, y_train_log, y_val_log, y_train_raw, y_val_raw = train_test_split(
        X_train_val, y_train_val_log, y_train_val_raw, test_size=0.1765, random_state=42
    )

    # 5. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    summary_stats = {
        'initial_rows': initial_rows,
        'dedup_rows': len(df),
        'valid_rows': valid_rows,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'feature_count': len(feature_cols),
        'feature_names': feature_cols
    }

    return {
        'df': df,
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'X_train_scaled': X_train_scaled, 'X_val_scaled': X_val_scaled, 'X_test_scaled': X_test_scaled,
        'y_train_log': y_train_log, 'y_val_log': y_val_log, 'y_test_log': y_test_log,
        'y_train_raw': y_train_raw, 'y_val_raw': y_val_raw, 'y_test_raw': y_test_raw,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'summary_stats': summary_stats
    }
