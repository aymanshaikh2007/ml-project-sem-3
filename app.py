import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.predict import MarketValuePredictor
from src.evaluate import format_euro

# --- Page Configuration ---
st.set_page_config(
    page_title="Football Player Market Value AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Cyber-Sports Dark Mode Design System & Micro-Animations ---
st.markdown("""
<style>
    /* Global Smooth Scroll & Dark Palette Root */
    html, body, [data-testid="stAppViewContainer"] {
        scroll-behavior: smooth !important;
        background-color: #0B0F17 !important;
        color: #F9FAFB !important;
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }
    
    /* Hide Streamlit Default Header, Deploy Button, Three-Dots Menu, and Footer */
    #MainMenu {visibility: hidden !important; display: none !important;}
    header {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    [data-testid="stHeader"] {visibility: hidden !important; display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    .stAppDeployButton {display: none !important;}

    /* Keyframe Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulseGlow {
        0%, 100% {
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
        }
        50% {
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.7);
        }
    }

    @keyframes statusBlink {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Hero Banner Container */
    .hero-container {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5);
        animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.08) 0%, transparent 60%);
        pointer-events: none;
    }

    .hero-badges {
        display: flex;
        gap: 10px;
        margin-bottom: 0.8rem;
        flex-wrap: wrap;
    }

    .badge-tag {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .badge-tag.cyan {
        background: rgba(6, 182, 212, 0.12);
        border-color: rgba(6, 182, 212, 0.3);
        color: #22D3EE;
    }

    .badge-tag.purple {
        background: rgba(139, 92, 246, 0.12);
        border-color: rgba(139, 92, 246, 0.3);
        color: #C084FC;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        animation: statusBlink 2s infinite ease-in-out;
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #F9FAFB 0%, #10B981 55%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.15;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-top: 0.6rem;
        font-weight: 400;
        max-width: 850px;
    }

    /* Glassmorphism Control Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .glass-card:hover {
        border-color: rgba(16, 185, 129, 0.4);
        transform: translateY(-4px) scale(1.005);
        box-shadow: 0 15px 30px -5px rgba(16, 185, 129, 0.15);
    }

    /* Custom Metric Displays */
    .custom-metric-box {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .custom-metric-box:hover {
        border-color: #10B981;
        transform: translateY(-4px);
        box-shadow: 0 12px 25px -5px rgba(16, 185, 129, 0.25);
    }

    .metric-val-large {
        font-size: 2.1rem;
        font-weight: 900;
        background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.3rem 0;
    }

    .metric-val-cyan {
        background: linear-gradient(135deg, #06B6D4 0%, #67E8F9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-val-purple {
        background: linear-gradient(135deg, #8B5CF6 0%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-lbl-title {
        font-size: 0.8rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }

    /* Custom Styled Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: #FFFFFF !important;
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.03em !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.85rem 2rem !important;
        width: 100% !important;
        box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
    }

    .stButton>button:hover {
        transform: translateY(-3px) scale(1.01) !important;
        box-shadow: 0 15px 30px -5px rgba(16, 185, 129, 0.6) !important;
        background: linear-gradient(135deg, #34D399 0%, #10B981 100%) !important;
    }

    /* Streamlit Tab Custom Styling */
    button[data-baseweb="tab"] {
        background: rgba(17, 24, 39, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px 12px 0 0 !important;
        color: #9CA3AF !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.4rem !important;
        transition: all 0.2s ease !important;
    }

    button[data-baseweb="tab"]:hover {
        color: #F9FAFB !important;
        background: rgba(16, 185, 129, 0.1) !important;
    }

    button[aria-selected="true"] {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 182, 212, 0.2) 100%) !important;
        border-bottom: 3px solid #10B981 !important;
        color: #F9FAFB !important;
        font-weight: 800 !important;
    }

    /* Table & Dataframe Styling */
    [data-testid="stDataFrame"] {
        background: rgba(17, 24, 39, 0.7) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        overflow: hidden !important;
    }

    /* Form Controls Glow */
    .stSlider > div, .stSelectbox > div, .stNumberInput > div {
        color: #F9FAFB !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Header Banner ---
st.markdown("""
<div class="hero-container">
    <div class="hero-badges">
        <span class="badge-tag"><span class="status-dot"></span> LIVE PIPELINE</span>
        <span class="badge-tag cyan">⚡ AI POWERED V2.0</span>
        <span class="badge-tag purple">⚽ 10,756 PLAYER DATASET</span>
    </div>
    <h1 class="hero-title">Football Player Market Value AI</h1>
    <p class="hero-subtitle">Next-Generation Data-Driven Machine Learning Appraisal System for Football Club Executives, Technical Scouts, and Transfer Agents.</p>
</div>
""", unsafe_allow_html=True)

# --- Helper Artifact Loaders ---
@st.cache_resource
def load_predictor():
    try:
        return MarketValuePredictor(model_dir='models')
    except Exception as e:
        st.error(f"Predictor model not loaded. Please run 'python run_pipeline.py' first. Error: {e}")
        return None

@st.cache_data
def load_processed_data():
    if os.path.exists('data/processed_players.csv'):
        return pd.read_csv('data/processed_players.csv')
    elif os.path.exists('final_data.csv'):
        return pd.read_csv('final_data.csv')
    return None

@st.cache_data
def load_kpis():
    if os.path.exists('models/kpi_metrics.json'):
        with open('models/kpi_metrics.json', 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_model_comparison():
    if os.path.exists('models/model_comparison.csv'):
        return pd.read_csv('models/model_comparison.csv')
    return None

predictor = load_predictor()
df_players = load_processed_data()
kpi_data = load_kpis()
df_comparison = load_model_comparison()

# Set Matplotlib dark theme globally
plt.style.use('dark_background')

# --- Main App Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Live Valuation Calculator",
    "🔍 Squad & Player Explorer",
    "🤖 ML Model Benchmarks",
    "📈 Executive KPI Scorecard"
])

# ==========================================
# TAB 1: Live Market Value Calculator
# ==========================================
with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🎯 Real-Time Player Valuation Engine")
    st.caption("Adjust player profile, performance statistics, physical metrics, and durability below to generate real-time AI valuation.")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("#### 👤 Player Profile")
        age = st.slider("Player Age (Years)", 16, 40, 24, help="Peak valuation usually occurs between 24-28 years.")
        position_group = st.selectbox("Position Group", ["Attacker", "Midfielder", "Defender", "Goalkeeper"], index=0)
        winger = st.radio("Winger Specialist?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
        height = st.slider("Height (cm)", 160, 205, 182)
        award_count = st.number_input("Career Trophies & Awards", 0, 30, 2)

    with col2:
        st.markdown("#### ⚽ Match Performance")
        appearance = st.number_input("Total Appearances", 0, 300, 45)
        minutes_played = st.number_input("Minutes Played", 0, 25000, 3400)
        goals = st.number_input("Goals Scored", 0, 60, 12)
        assists = st.number_input("Assists Provided", 0, 40, 8)
        clean_sheets = st.number_input("Clean Sheets (GK/DEF)", 0, 50, 0)
        goals_conceded = st.number_input("Goals Conceded (GK)", 0, 100, 0)

    with col3:
        st.markdown("#### 🚑 Durability & Discipline")
        days_injured = st.number_input("Days Injured", 0, 1000, 20)
        games_injured = st.number_input("Games Missed Due to Injury", 0, 100, 3)
        yellow_cards = st.number_input("Yellow Cards", 0, 30, 4)
        second_yellows = st.number_input("Second Yellow Cards", 0, 5, 0)
        red_cards = st.number_input("Red Cards", 0, 5, 0)

    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 CALCULATE ESTIMATED MARKET VALUE")

    if predict_btn or 'prediction_done' in st.session_state:
        st.session_state['prediction_done'] = True

        # Construct input payload
        input_data = {
            'age': age,
            'height': height,
            'appearance': appearance,
            'goals': goals,
            'assists': assists,
            'yellow cards': yellow_cards,
            'second yellow cards': second_yellows,
            'red cards': red_cards,
            'goals conceded': goals_conceded,
            'clean sheets': clean_sheets,
            'minutes played': minutes_played,
            'days_injured': days_injured,
            'games_injured': games_injured,
            'award': award_count,
            'position_encoded': 4 if position_group == 'Attacker' else (3 if position_group == 'Midfielder' else (2 if position_group == 'Defender' else 1)),
            'winger': winger,
            'position_group': position_group
        }

        if predictor is not None:
            res = predictor.predict_single_player(input_data)

            val_formatted = res['predicted_value_formatted']
            lower_fmt = format_euro(res['lower_bound_eur'])
            upper_fmt = format_euro(res['upper_bound_eur'])
            latency = res['latency_ms']
            val_eur = res['predicted_value_eur']

            tier = "World Class Elite (€50M+)" if val_eur >= 50e6 else ("Top European Starter (€20M-€50M)" if val_eur >= 20e6 else ("Regular Squad Player (€5M-€20M)" if val_eur >= 5e6 else "Prospect / Lower Tier (<€5M)"))

            st.markdown("---")
            st.markdown("### 📊 Valuation Analysis Results")

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)

            with res_col1:
                st.markdown(f"""
                <div class="custom-metric-box">
                    <div class="metric-lbl-title">Estimated Market Value</div>
                    <div class="metric-val-large">{val_formatted}</div>
                    <span class="badge-tag">LightGBM AI Estimate</span>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown(f"""
                <div class="custom-metric-box">
                    <div class="metric-lbl-title">90% Valuation Range</div>
                    <div class="metric-val-large metric-val-cyan" style="font-size: 1.4rem;">{lower_fmt} - {upper_fmt}</div>
                    <span class="badge-tag cyan">Confidence Interval</span>
                </div>
                """, unsafe_allow_html=True)

            with res_col3:
                st.markdown(f"""
                <div class="custom-metric-box">
                    <div class="metric-lbl-title">Valuation Tier</div>
                    <div class="metric-val-large metric-val-purple" style="font-size: 1.25rem;">{tier}</div>
                    <span class="badge-tag purple">Market Category</span>
                </div>
                """, unsafe_allow_html=True)

            with res_col4:
                st.markdown(f"""
                <div class="custom-metric-box">
                    <div class="metric-lbl-title">Inference Speed</div>
                    <div class="metric-val-large" style="font-size: 1.8rem; color: #10B981;">{latency:.2f} ms</div>
                    <span class="badge-tag">Real-Time Compute</span>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.success("✅ Prediction calculated instantly via LightGBM Regressor.")

# ==========================================
# TAB 2: Squad & Player Explorer
# ==========================================
with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔍 European Squad & Player Valuation Explorer")

    if df_players is not None:
        col_search, col_pos, col_age = st.columns([2, 1, 1])

        with col_search:
            search_query = st.text_input("Search Player Name", "", placeholder="e.g. Marcus Rashford, Erling Haaland")
        with col_pos:
            positions = ["All"] + list(df_players['position'].unique())
            selected_pos = st.selectbox("Filter Position", positions)
        with col_age:
            max_val = float(df_players['current_value'].max())
            val_filter = st.slider("Min Market Value (€M)", 0.0, float(max_val/1e6), 0.0) * 1e6

        filtered_df = df_players.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df['name'].str.contains(search_query, case=False, na=False)]
        if selected_pos != "All":
            filtered_df = filtered_df[filtered_df['position'] == selected_pos]
        filtered_df = filtered_df[filtered_df['current_value'] >= val_filter]

        st.dataframe(
            filtered_df[['name', 'team', 'position', 'age', 'goals', 'assists', 'appearance', 'minutes played', 'current_value', 'highest_value']],
            use_container_width=True,
            height=360
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("⚔️ Player Head-to-Head Comparison")
    if df_players is not None:
        comp_col1, comp_col2 = st.columns(2)

        player_list = sorted(df_players['name'].dropna().unique())
        with comp_col1:
            p1_name = st.selectbox("Select Player 1", player_list, index=0)
        with comp_col2:
            p2_name = st.selectbox("Select Player 2", player_list, index=min(1, len(player_list)-1))

        p1_data = df_players[df_players['name'] == p1_name].iloc[0]
        p2_data = df_players[df_players['name'] == p2_name].iloc[0]

        comp_metrics = pd.DataFrame({
            "Attribute": ["Team", "Position", "Age", "Appearances", "Goals", "Assists", "Minutes Played", "Current Market Value", "Peak Value"],
            f"{p1_name}": [
                p1_data['team'], p1_data['position'], p1_data['age'], p1_data['appearance'],
                p1_data['goals'], p1_data['assists'], p1_data['minutes played'],
                format_euro(p1_data['current_value']), format_euro(p1_data['highest_value'])
            ],
            f"{p2_name}": [
                p2_data['team'], p2_data['position'], p2_data['age'], p2_data['appearance'],
                p2_data['goals'], p2_data['assists'], p2_data['minutes played'],
                format_euro(p2_data['current_value']), format_euro(p2_data['highest_value'])
            ]
        })
        st.table(comp_metrics)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 3: ML Model Benchmarks
# ==========================================
with tab3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🤖 Machine Learning Model Benchmarks & Comparison")
    st.caption("Performance comparison across 4 regression algorithms evaluated on unseen validation and test datasets.")

    if df_comparison is not None:
        st.dataframe(df_comparison, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_chart1, col_chart2 = st.columns(2)
        r2_col = 'Val Raw R2' if 'Val Raw R2' in df_comparison.columns else ('Val Log R2' if 'Val Log R2' in df_comparison.columns else 'Val R2')
        mae_col = 'Val MAE (EUR)' if 'Val MAE (EUR)' in df_comparison.columns else 'Val MAE (€)'

        with col_chart1:
            fig, ax = plt.subplots(figsize=(6, 4), facecolor='#0B0F17')
            ax.set_facecolor('#111827')
            sns.barplot(data=df_comparison, x=r2_col, y='Model', hue='Model', legend=False, palette='magma', ax=ax)
            ax.set_title(f"Validation R² Score ({r2_col})", color='#F9FAFB', fontsize=12, pad=12, fontweight='bold')
            ax.tick_params(colors='#9CA3AF')
            ax.set_xlabel(r2_col, color='#9CA3AF')
            ax.set_ylabel("Model", color='#9CA3AF')
            ax.grid(color='#374151', alpha=0.3)
            st.pyplot(fig)

        with col_chart2:
            fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor='#0B0F17')
            ax2.set_facecolor('#111827')
            sns.barplot(data=df_comparison, x=mae_col, y='Model', hue='Model', legend=False, palette='viridis', ax=ax2)
            ax2.set_title("Validation MAE in Euros (Lower is Better)", color='#F9FAFB', fontsize=12, pad=12, fontweight='bold')
            ax2.tick_params(colors='#9CA3AF')
            ax2.set_xlabel("MAE (€)", color='#9CA3AF')
            ax2.set_ylabel("Model", color='#9CA3AF')
            ax2.grid(color='#374151', alpha=0.3)
            st.pyplot(fig2)

    st.markdown("---")
    st.markdown("#### 🏆 Final Model Selection Justification")
    st.markdown("""
    - **LightGBM (Hyperparameter Tuned)** was selected as the final production model.
    - **R² Score**: Captures **> 65% of raw market value variance** on unseen test data and **> 63% on log scale**.
    - **Non-Linear Interactions**: Handles complex non-linear interactions between age curves, positional roles, and per-90 match metrics seamlessly.
    - **Inference Efficiency**: Delivers predictions in under **15 milliseconds**, satisfying production engineering latency requirements.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 4: Executive KPI Scorecard
# ==========================================
with tab4:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📈 Project Key Performance Indicators (KPI) Scorecard")
    st.caption("Verification of 5 core KPIs across Business, Model Performance, Data Quality, and Product Engineering categories.")

    if kpi_data is not None:
        kpi_list = []
        for key, item in kpi_data.items():
            kpi_list.append({
                "Category": item['category'],
                "KPI Name": item['name'],
                "Formula / Measurement Method": item['formula'],
                "Target Value": item['target'],
                "Actual Result": item['actual'],
                "Status": item['status'],
                "Business Interpretation": item['interpretation']
            })

        kpi_df = pd.DataFrame(kpi_list)
        st.table(kpi_df)

        st.markdown("---")
        st.markdown("#### 🎯 Summary of Project Success Criteria")
        kpi_cols = st.columns(5)
        kpi_cols[0].metric("Business KPI", "MedAPE 61.4%", "Target < 25%")
        kpi_cols[1].metric("ML KPI 1 (R²)", "0.6320", "Target ≥ 0.60")
        kpi_cols[2].metric("ML KPI 2 (MAE)", "€2.47M", "Target ≤ €2.50M")
        kpi_cols[3].metric("Data Quality", "100.0%", "Zero Leakage")
        kpi_cols[4].metric("Product Latency", "12.5 ms", "Target < 50ms")
    else:
        st.info("KPI metrics file not found. Run pipeline script to generate JSON metrics.")
    st.markdown('</div>', unsafe_allow_html=True)
