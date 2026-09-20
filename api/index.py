import os
import json
import time
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Base directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Top European Elite Clubs
TOP_CLUBS = {
    'manchester united', 'real madrid', 'manchester city', 'fc barcelona', 'barcelona',
    'bayern munich', 'paris saint-germain', 'psg', 'liverpool fc', 'liverpool',
    'chelsea fc', 'chelsea', 'arsenal fc', 'arsenal', 'juventus', 'inter milan',
    'ac milan', 'atletico madrid', 'tottenham hotspur', 'borussia dortmund', 'napoli'
}

# Lazy JSON Artifact Loaders
_lgbm_json = None
_scaler_dict = None
_feature_cols = None
_players_data = None
_kpis_data = None
_comparison_data = None

def get_json_model():
    global _lgbm_json, _scaler_dict, _feature_cols
    if _lgbm_json is None:
        model_path = os.path.join(MODEL_DIR, 'lightgbm_model.json')
        scaler_path = os.path.join(MODEL_DIR, 'scaler_params.json')
        cols_path = os.path.join(MODEL_DIR, 'feature_columns.json')

        if os.path.exists(model_path):
            with open(model_path, 'r') as f:
                _lgbm_json = json.load(f)
        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                _scaler_dict = json.load(f)
        if os.path.exists(cols_path):
            with open(cols_path, 'r') as f:
                _feature_cols = json.load(f)
    return _lgbm_json, _scaler_dict, _feature_cols

def get_players():
    global _players_data
    if _players_data is None:
        p_path = os.path.join(MODEL_DIR, 'players_slim.json')
        if os.path.exists(p_path):
            with open(p_path, 'r') as f:
                _players_data = json.load(f)
    return _players_data or []

def get_kpis():
    global _kpis_data
    if _kpis_data is None:
        k_path = os.path.join(MODEL_DIR, 'kpi_metrics.json')
        if os.path.exists(k_path):
            with open(k_path, 'r') as f:
                _kpis_data = json.load(f)
    return _kpis_data or {}

def get_benchmarks():
    global _comparison_data
    if _comparison_data is None:
        c_path = os.path.join(MODEL_DIR, 'model_comparison.csv')
        if os.path.exists(c_path):
            import csv
            with open(c_path, 'r') as f:
                reader = csv.DictReader(f)
                _comparison_data = list(reader)
    return _comparison_data or []

# LightGBM Pure Python Decision Tree Evaluator
def eval_node(node, features):
    if 'leaf_value' in node:
        return node['leaf_value']
    feat_idx = node['split_feature']
    threshold = node['threshold']
    if features[feat_idx] <= threshold:
        return eval_node(node['left_child'], features)
    else:
        return eval_node(node['right_child'], features)

def predict_lgb_json(lgb_json, feature_vector):
    raw_score = float(lgb_json.get('base_score', 0.0))
    for tree in lgb_json.get('tree_info', []):
        raw_score += float(eval_node(tree['tree_structure'], feature_vector))
    return raw_score

def scale_features(raw_vector, mean_vec, scale_vec):
    scaled = []
    for x, m, s in zip(raw_vector, mean_vec, scale_vec):
        s_val = s if s != 0 else 1.0
        scaled.append((x - m) / s_val)
    return scaled

def format_euro(amount):
    if amount >= 1_000_000:
        return f"EUR {amount / 1_000_000:.2f}M"
    elif amount >= 1_000:
        return f"EUR {amount / 1_000:.0f}K"
    else:
        return f"EUR {amount:.0f}"

# --- API Endpoints ---
@app.route('/api/predict', methods=['POST'])
def predict():
    t0 = time.time()
    data = request.json or {}

    lgb_json, scaler_dict, feature_cols = get_json_model()
    if lgb_json is None or scaler_dict is None or feature_cols is None:
        return jsonify({'error': 'Model JSON artifacts missing.'}), 500

    age = float(data.get('age', 24))
    height = float(data.get('height', 182))
    appearance = float(data.get('appearance', 45))
    goals = float(data.get('goals', 12))
    assists = float(data.get('assists', 8))
    yellow_cards = float(data.get('yellow_cards', 4))
    second_yellows = float(data.get('second_yellows', 0))
    red_cards = float(data.get('red_cards', 0))
    goals_conceded = float(data.get('goals_conceded', 0))
    clean_sheets = float(data.get('clean_sheets', 0))
    minutes_played = float(data.get('minutes_played', 3400))
    days_injured = float(data.get('days_injured', 20))
    games_injured = float(data.get('games_injured', 3))
    award = float(data.get('award', 2))
    winger = float(data.get('winger', 0))
    pos_group = str(data.get('position_group', 'Attacker'))
    team_name = str(data.get('team', ''))

    position_encoded = 4 if pos_group == 'Attacker' else (3 if pos_group == 'Midfielder' else (2 if pos_group == 'Defender' else 1))

    minutes_safe = max(minutes_played, 1.0)
    apps_safe = max(appearance, 1.0)

    is_top_c = 1 if any(c in team_name.lower() for c in TOP_CLUBS) else 0
    goals_per_90 = (goals / minutes_safe) * 90.0
    assists_per_90 = (assists / minutes_safe) * 90.0
    goal_contrib_per_90 = goals_per_90 + assists_per_90
    clean_sheets_per_game = clean_sheets / apps_safe
    goals_conceded_per_90 = (goals_conceded / minutes_safe) * 90.0
    minutes_per_game = minutes_played / apps_safe
    discipline_score = yellow_cards + (2.0 * second_yellows) + (3.0 * red_cards)
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
        'age': age, 'height': height, 'appearance': appearance, 'goals': goals,
        'assists': assists, 'yellow cards': yellow_cards, 'second yellow cards': second_yellows,
        'red cards': red_cards, 'goals conceded': goals_conceded, 'clean sheets': clean_sheets,
        'minutes played': minutes_played, 'days_injured': days_injured, 'games_injured': games_injured,
        'award': award, 'position_encoded': position_encoded, 'winger': winger,
        'is_top_club': is_top_c, 'goals_per_90': goals_per_90, 'assists_per_90': assists_per_90,
        'goal_contrib_per_90': goal_contrib_per_90, 'clean_sheets_per_game': clean_sheets_per_game,
        'goals_conceded_per_90': goals_conceded_per_90, 'minutes_per_game': minutes_per_game,
        'discipline_score': discipline_score, 'injury_days_per_game': injury_days_per_game,
        'games_injured_ratio': games_injured_ratio, 'age_squared': age_squared,
        'age_cubed': age_cubed, 'age_prime_dist': age_prime_dist, 'is_prime_age': is_prime_age,
        'pos_Attacker': pos_Attacker, 'pos_Defender': pos_Defender,
        'pos_Goalkeeper': pos_Goalkeeper, 'pos_Midfielder': pos_Midfielder
    }

    raw_vector = [feature_map.get(col, 0.0) for col in feature_cols]
    scaled_vector = scale_features(raw_vector, scaler_dict['mean'], scaler_dict['scale'])

    pred_log = predict_lgb_json(lgb_json, scaled_vector)
    import math
    pred_raw = max(math.expm1(pred_log), 100_000.0)

    latency_ms = (time.time() - t0) * 1000.0

    lower_bound = max(pred_raw * 0.78, 100_000.0)
    upper_bound = pred_raw * 1.25

    tier = "World Class Elite (€50M+)" if pred_raw >= 50e6 else ("Top European Starter (€20M-€50M)" if pred_raw >= 20e6 else ("Regular Squad Player (€5M-€20M)" if pred_raw >= 5e6 else "Prospect / Lower Tier (<€5M)"))

    return jsonify({
        'predicted_value_eur': round(pred_raw, 2),
        'predicted_value_formatted': format_euro(pred_raw),
        'lower_bound_formatted': format_euro(lower_bound),
        'upper_bound_formatted': format_euro(upper_bound),
        'tier': tier,
        'latency_ms': round(latency_ms, 2)
    })

@app.route('/api/players', methods=['GET'])
def get_players_endpoint():
    players = get_players()
    query = request.args.get('search', '').lower()
    if query:
        players = [p for p in players if query in str(p.get('name', '')).lower()]
    return jsonify(players[:50])

@app.route('/api/kpis', methods=['GET'])
def get_kpis_endpoint():
    return jsonify(get_kpis())

@app.route('/api/benchmarks', methods=['GET'])
def get_benchmarks_endpoint():
    return jsonify(get_benchmarks())

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Football Player Market Value AI Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0B0F17;
            --card-bg: rgba(17, 24, 39, 0.75);
            --emerald: #10B981;
            --cyan: #06B6D4;
            --purple: #8B5CF6;
            --text-light: #F9FAFB;
            --text-dim: #9CA3AF;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-dark);
            color: var(--text-light);
            font-family: 'Inter', sans-serif;
            padding: 2rem 1.5rem;
            line-height: 1.5;
        }

        .container { max-width: 1200px; margin: 0 auto; }

        .hero {
            background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 20px 40px -15px rgba(0,0,0,0.5);
        }

        .badges { display: flex; gap: 10px; margin-bottom: 0.8rem; }
        .badge {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34D399;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 20px;
            text-transform: uppercase;
        }

        .title {
            font-size: 2.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #F9FAFB 0%, #10B981 55%, #06B6D4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle { color: var(--text-dim); margin-top: 0.5rem; font-size: 1rem; }

        .tabs { display: flex; gap: 10px; margin-bottom: 1.5rem; }
        .tab-btn {
            background: rgba(17, 24, 39, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: var(--text-dim);
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
        }
        .tab-btn.active {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%);
            border-bottom: 3px solid var(--emerald);
            color: var(--text-light);
        }

        .glass-card {
            background: var(--card-bg);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
        }

        .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }

        label { display: block; font-size: 0.85rem; color: var(--text-dim); margin-bottom: 0.4rem; font-weight: 600; }
        input, select {
            width: 100%;
            background: #111827;
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-light);
            padding: 0.75rem 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            font-size: 0.95rem;
        }

        .btn-submit {
            width: 100%;
            background: linear-gradient(135deg, var(--emerald) 0%, #059669 100%);
            color: white;
            font-size: 1.1rem;
            font-weight: 800;
            border: none;
            padding: 1rem;
            border-radius: 12px;
            cursor: pointer;
            box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.4);
            transition: all 0.3s;
        }
        .btn-submit:hover { transform: translateY(-2px); box-shadow: 0 15px 30px -5px rgba(16, 185, 129, 0.6); }

        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-top: 1.5rem; }
        .metric-card {
            background: rgba(15, 23, 42, 0.9);
            border: 1px solid rgba(16, 185, 129, 0.25);
            border-radius: 14px;
            padding: 1.2rem;
            text-align: center;
        }
        .metric-val { font-size: 1.8rem; font-weight: 900; color: var(--emerald); margin: 0.3rem 0; }

        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        th, td { padding: 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.08); text-align: left; }
        th { color: var(--text-dim); font-size: 0.85rem; text-transform: uppercase; }

        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="hero">
            <div class="badges">
                <span class="badge">VERCEL ULTRA-SLIM ENGINE</span>
                <span class="badge">AI POWERED V2.0</span>
            </div>
            <h1 class="title">Football Player Market Value AI</h1>
            <p class="subtitle">Data-Driven Machine Learning Appraisal System for Transfer Scouting & Financial Evaluation.</p>
        </div>

        <div class="tabs">
            <button class="tab-btn active" onclick="showTab('tab-calc', this)">🎯 Live Calculator</button>
            <button class="tab-btn" onclick="showTab('tab-explorer', this)">🔍 Squad Explorer</button>
            <button class="tab-btn" onclick="showTab('tab-benchmarks', this)">🤖 ML Benchmarks</button>
            <button class="tab-btn" onclick="showTab('tab-kpis', this)">📈 KPI Scorecard</button>
        </div>

        <div id="tab-calc" class="tab-content">
            <div class="glass-card">
                <h3 style="margin-bottom: 1rem; color: var(--emerald);">🎯 Real-Time Player Market Value Valuation Engine</h3>
                <form id="calc-form" onsubmit="handlePredict(event)">
                    <div class="grid-3">
                        <div>
                            <label>Player Age (Years)</label>
                            <input type="number" id="age" value="24" min="16" max="40">
                            
                            <label>Position Group</label>
                            <select id="position_group">
                                <option value="Attacker">Attacker</option>
                                <option value="Midfielder">Midfielder</option>
                                <option value="Defender">Defender</option>
                                <option value="Goalkeeper">Goalkeeper</option>
                            </select>

                            <label>Winger Specialist?</label>
                            <select id="winger">
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>

                            <label>Height (cm)</label>
                            <input type="number" id="height" value="182">
                        </div>
                        <div>
                            <label>Appearances</label>
                            <input type="number" id="appearance" value="45">

                            <label>Minutes Played</label>
                            <input type="number" id="minutes_played" value="3400">

                            <label>Goals Scored</label>
                            <input type="number" id="goals" value="12">

                            <label>Assists Provided</label>
                            <input type="number" id="assists" value="8">
                        </div>
                        <div>
                            <label>Days Injured</label>
                            <input type="number" id="days_injured" value="20">

                            <label>Games Missed</label>
                            <input type="number" id="games_injured" value="3">

                            <label>Yellow Cards</label>
                            <input type="number" id="yellow_cards" value="4">

                            <label>Career Trophies & Awards</label>
                            <input type="number" id="award" value="2">
                        </div>
                    </div>
                    <button type="submit" class="btn-submit">🚀 CALCULATE ESTIMATED MARKET VALUE</button>
                </form>

                <div id="results-box" class="hidden">
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <small>ESTIMATED MARKET VALUE</small>
                            <div class="metric-val" id="res-val">€0M</div>
                            <span class="badge">LightGBM AI Estimate</span>
                        </div>
                        <div class="metric-card">
                            <small>90% VALUATION RANGE</small>
                            <div class="metric-val" style="color: var(--cyan);" id="res-range">€0 - €0</div>
                            <span class="badge">Confidence Bound</span>
                        </div>
                        <div class="metric-card">
                            <small>VALUATION TIER</small>
                            <div class="metric-val" style="color: var(--purple); font-size: 1.3rem;" id="res-tier">Elite</div>
                            <span class="badge">Category</span>
                        </div>
                        <div class="metric-card">
                            <small>INFERENCE SPEED</small>
                            <div class="metric-val" style="font-size: 1.5rem;" id="res-latency">12ms</div>
                            <span class="badge">Real-Time Compute</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div id="tab-explorer" class="tab-content hidden">
            <div class="glass-card">
                <h3>🔍 Player Database Explorer</h3>
                <input type="text" id="search-input" placeholder="Search player name..." onkeyup="loadPlayers()">
                <table id="players-table">
                    <thead>
                        <tr>
                            <th>Player Name</th><th>Team</th><th>Position</th><th>Age</th><th>Goals</th><th>Assists</th><th>Market Value</th>
                        </tr>
                    </thead>
                    <tbody id="players-tbody"></tbody>
                </table>
            </div>
        </div>

        <div id="tab-benchmarks" class="tab-content hidden">
            <div class="glass-card">
                <h3>🤖 Machine Learning Model Benchmarks</h3>
                <table id="benchmarks-table">
                    <thead>
                        <tr>
                            <th>Model Name</th><th>Val Raw R²</th><th>Val Log R²</th><th>Val MAE (EUR)</th><th>Training Time</th>
                        </tr>
                    </thead>
                    <tbody id="benchmarks-tbody"></tbody>
                </table>
            </div>
        </div>

        <div id="tab-kpis" class="tab-content hidden">
            <div class="glass-card">
                <h3>📈 Executive KPI Scorecard</h3>
                <table id="kpis-table">
                    <thead>
                        <tr>
                            <th>Category</th><th>KPI Name</th><th>Target</th><th>Actual Result</th><th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="kpis-tbody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        function showTab(tabId, btn) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.remove('hidden');
            btn.classList.add('active');

            if(tabId === 'tab-explorer') loadPlayers();
            if(tabId === 'tab-benchmarks') loadBenchmarks();
            if(tabId === 'tab-kpis') loadKPIs();
        }

        async function handlePredict(e) {
            e.preventDefault();
            const payload = {
                age: parseFloat(document.getElementById('age').value),
                position_group: document.getElementById('position_group').value,
                winger: parseFloat(document.getElementById('winger').value),
                height: parseFloat(document.getElementById('height').value),
                appearance: parseFloat(document.getElementById('appearance').value),
                minutes_played: parseFloat(document.getElementById('minutes_played').value),
                goals: parseFloat(document.getElementById('goals').value),
                assists: parseFloat(document.getElementById('assists').value),
                days_injured: parseFloat(document.getElementById('days_injured').value),
                games_injured: parseFloat(document.getElementById('games_injured').value),
                yellow_cards: parseFloat(document.getElementById('yellow_cards').value),
                award: parseFloat(document.getElementById('award').value)
            };

            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();

            document.getElementById('res-val').innerText = data.predicted_value_formatted;
            document.getElementById('res-range').innerText = data.lower_bound_formatted + " - " + data.upper_bound_formatted;
            document.getElementById('res-tier').innerText = data.tier;
            document.getElementById('res-latency').innerText = data.latency_ms + " ms";
            document.getElementById('results-box').classList.remove('hidden');
        }

        async function loadPlayers() {
            const query = document.getElementById('search-input').value;
            const res = await fetch('/api/players?search=' + encodeURIComponent(query));
            const data = await res.json();
            const tbody = document.getElementById('players-tbody');
            tbody.innerHTML = data.map(p => `
                <tr>
                    <td><strong>${p.name}</strong></td>
                    <td>${p.team}</td>
                    <td>${p.position}</td>
                    <td>${p.age}</td>
                    <td>${p.goals}</td>
                    <td>${p.assists}</td>
                    <td><strong style="color:#10B981;">€${(p.current_value/1e6).toFixed(2)}M</strong></td>
                </tr>
            `).join('');
        }

        async function loadBenchmarks() {
            const res = await fetch('/api/benchmarks');
            const data = await res.json();
            const tbody = document.getElementById('benchmarks-tbody');
            tbody.innerHTML = data.map(m => `
                <tr>
                    <td><strong>${m.Model}</strong></td>
                    <td>${m['Val Raw R2'] || '-'}</td>
                    <td>${m['Val Log R2'] || '-'}</td>
                    <td>€${m['Val MAE (EUR)'] ? (m['Val MAE (EUR)']/1e6).toFixed(2) + 'M' : '-'}</td>
                    <td>${m['Training Time (s)'] || '-'}s</td>
                </tr>
            `).join('');
        }

        async function loadKPIs() {
            const res = await fetch('/api/kpis');
            const data = await res.json();
            const tbody = document.getElementById('kpis-tbody');
            tbody.innerHTML = Object.values(data).map(k => `
                <tr>
                    <td><strong>${k.category}</strong></td>
                    <td>${k.name}</td>
                    <td>${k.target}</td>
                    <td><strong style="color:#10B981;">${k.actual}</strong></td>
                    <td><span class="badge">${k.status}</span></td>
                </tr>
            `).join('');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
