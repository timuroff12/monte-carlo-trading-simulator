import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Professional Monte Carlo Sim", layout="wide")

# --- LOCALIZATION ---
languages = {
    "RU": {
        "title": "Симуляция Монте-Карло для трейдеров",
        "settings": "Настройки",
        "mode": "Режим:",
        "start_bal": "Начальный баланс",
        "win_rate": "Победные сделки %",
        "be_rate": "Безубыток %",
        "risk": "Риск",
        "reward": "Прибыль",
        "num_sims": "Количество симуляций",
        "trades_month": "Сделок в месяц",
        "months": "Срок (месяцев)",
        "variability": "Вариативность RR (%)",
        "ruin_threshold": "Порог разорения ($)",
        "run_sim": "Запуск симуляций...",
        "sensitivity": "Анализ чувствительности",
        "analysis_title": "Детальный анализ сценариев",
        "year_total": "Итого за год:",
        "risk_of_ruin": "Риск разорения",
        "stats": "Статистика",
        "hist_title": "Распределение финальных балансов",
        "months_list": ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
    },
    "EN": {
        "title": "Professional Monte Carlo Simulation",
        "settings": "Settings",
        "mode": "Mode:",
        "start_bal": "Starting Balance",
        "win_rate": "Win Rate %",
        "be_rate": "Break Even %",
        "risk": "Risk",
        "reward": "Reward",
        "num_sims": "Number of Simulations",
        "trades_month": "Trades per Month",
        "months": "Duration (Months)",
        "variability": "RR Variability (%)",
        "ruin_threshold": "Ruin Threshold ($)",
        "run_sim": "Running Simulations...",
        "sensitivity": "Sensitivity Analysis",
        "analysis_title": "Detailed Scenario Analysis",
        "year_total": "Year Total:",
        "risk_of_ruin": "Risk of Ruin",
        "stats": "Statistics",
        "hist_title": "Final Balance Distribution",
        "months_list": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    }
}

lang_choice = st.sidebar.selectbox("Language / Язык", ["RU", "EN"])
T = languages[lang_choice]

# --- SIDEBAR ---
with st.sidebar:
    st.header(T['settings'])
    
    # CHANGE 4: Mode $ moved to the right
    side_col1, side_col2 = st.columns([1, 1])
    with side_col2:
        mode = st.radio(T['mode'], ["%", "$"])
    
    start_balance = st.number_input(T['start_bal'], value=10000, step=1000)
    
    col_win, col_be = st.columns(2)
    win_rate = col_win.number_input(T['win_rate'], value=55)
    be_rate = col_be.number_input(T['be_rate'], value=5)
    
    col_r, col_p = st.columns(2)
    risk_val = col_r.number_input(f"{T['risk']} ({mode})", value=1 if "%" in mode else 100)
    reward_val = col_p.number_input(f"{T['reward']} ({mode})", value=2 if "%" in mode else 200)
    
    num_sims = st.number_input(T['num_sims'], value=50, step=1)
    trades_per_month = st.slider(T['trades_month'], 1, 50, 20)
    num_months = st.number_input(T['months'], value=24, step=1)
    variability = st.slider(T['variability'], 0, 100, 20)
    ruin_threshold = st.number_input(T['ruin_threshold'], value=int(start_balance * 0.1))

    st.subheader("Advanced Settings")
    commission_fixed = st.number_input("Комиссия фикс. ($)", value=0.0)
    commission_pct = st.number_input("Комиссия %", value=0.0)
    slippage = st.number_input("Слиппедж ($ или %)", value=0.0)
    dist_type = st.selectbox("Распределение исходов", ["Uniform", "Normal", "LogNormal", "Bootstrap"])
    uploaded_file = st.file_uploader("Загрузить CSV для Bootstrap", type="csv")
    sizing_type = st.selectbox("Position Sizing", ["Fixed Risk", "Kelly"])

# --- CSS ---
st.markdown(f"""
    <style>
    div[class*="stMain"] h1 {{ border-bottom: none !important; padding-bottom: 0.5rem !important; }}
    [data-baseweb="tab-highlight"] {{ display: none !important; }}
    .stTabs [data-baseweb="tab-list"] {{ display: flex; justify-content: center; gap: 12px; padding-bottom: 20px; border: none !important; }}
    .stTabs [data-baseweb="tab"] {{ height: 60px; width: 280px; border-radius: 8px; font-weight: bold; font-size: 22px; color: white !important; border: none !important; transition: all 0.2s ease; }}
    div[data-baseweb="tab-list"] button:nth-child(1) {{ background-color: #3B82F6 !important; }}
    div[data-baseweb="tab-list"] button:nth-child(2) {{ background-color: #EF4444 !important; }}
    div[data-baseweb="tab-list"] button:nth-child(3) {{ background-color: #10B981 !important; }}
    .stTabs [aria-selected="true"] {{ filter: brightness(1.2); transform: scale(1.02); box-shadow: 0px 5px 15px rgba(0,0,0,0.3); }}
    .year-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 16px; }}
    .year-table th, .year-table td {{ border: 1px solid #444; padding: 10px; text-align: center; }}
    .year-table th {{ background-color: #262730; color: #E0E0E0; font-weight: normal; }}
    .year-table td {{ font-size: 17px; }}
    .pos-val {{ color: #10B981; font-weight: bold; }}
    .neg-val {{ color: #EF4444; font-weight: bold; }}
    </style>
""", unsafe_allow_html=True)

st.title(f"{T['title']} by timuroff")

# --- FUNCTIONS ---
def calculate_single_mdd(history):
    h = np.array(history)
    peaks = np.maximum.accumulate(h)
    drawdowns = (peaks - h) / (peaks + 1e-9)
    return float(np.max(drawdowns) * 100)

# CHANGE 3: Max DD calculation from Initial Balance
def calculate_mdd_initial(history, initial_bal):
    h = np.array(history)
    drawdowns = (initial_bal - h) / (initial_bal + 1e-9)
    return float(np.max(drawdowns) * 100)

def get_consecutive(results):
    max_wins, max_losses = 0, 0
    cur_wins, cur_losses = 0, 0
    for r in results:
        if r == 1: cur_wins += 1; cur_losses = 0
        elif r == -1: cur_losses += 1; cur_wins = 0
        else: cur_wins, cur_losses = 0, 0
        max_wins, max_losses = max(max_wins, cur_wins), max(max_losses, cur_losses)
    return max_wins, max_losses

def run_simulation(n
