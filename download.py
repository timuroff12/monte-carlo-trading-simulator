import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Professional Monte Carlo Sim", layout="wide")

# --- ЛОКАЛИЗАЦИЯ ---
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

# --- ВЫБОР ЯЗЫКА ---
lang_choice = st.sidebar.selectbox("Language / Язык", ["RU", "EN"])
T = languages[lang_choice]

# --- SIDEBAR (Настройки и Тема) ---
with st.sidebar:
    st.header(T['settings'])
    
    theme = st.radio("Theme / Тема", ["Dark", "Light"])
    
    with st.expander("FAQ / Что это такое?"):
        st.markdown("""
        - **Монте-Карло симуляция**: Метод моделирования сценариев на основе вероятностей.
        - **Win Rate**: % выигрышных сделок.
        - **Risk/Reward**: Соотношение риска к прибыли (RR).
        - **Вариативность**: Добавляет случайность к RR для реализма.
        - **Риск разорения**: Вероятность падения ниже порога.
        """)

# --- CSS И СТИЛИЗАЦИЯ ---
plotly_template = "plotly_dark"
if theme == "Light":
    plotly_template = "plotly_white"
    st.markdown("""
        <style>
        .stApp { background-color: white; color: black; }
        section[data-testid="stSidebar"] { background-color: #f0f2f6; }
        </style>
    """, unsafe_allow_html=True)

# Основной CSS
st.markdown("""
    <style>
    /* Убираем декоративную полоску под заголовком H1 */
    div[class*="stMain"] h1 {
        border-bottom: none !important;
        padding-bottom: 0.5rem !important;
    }
    
    /* Стилизация табов */
    [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-list"] { display: flex; justify-content: center; gap: 12px; padding-bottom: 20px; border: none !important; }
    .stTabs [data-baseweb="tab"] { height: 60px; width: 250px; border-radius: 8px; font-weight: bold; font-size: 18px; color: white !important; border: none !important; transition: all 0.2s ease; }
    div[data-baseweb="tab-list"] button:nth-child(1) { background-color: #3B82F6 !important; }
    div[data-baseweb="tab-list"] button:nth-child(2) { background-color: #EF4444 !important; }
    div[data-baseweb="tab-list"] button:nth-child(3) { background-color: #10B981 !important; }
    .stTabs [aria-selected="true"] { filter: brightness(1.2); transform: scale(1.02); box-shadow: 0px 5px 15px rgba(0,0,0,0.3); }
    </style>
""", unsafe_allow_html=True)

# --- ВВОД ДАННЫХ В SIDEBAR ---
with st.sidebar:
    mode = st.radio(T['mode'], ["%", "$"])
    start_balance = st.number_input(T['start_bal'], value=10000, step=1000, help="Ваш начальный капитал.")
    
    col_win, col_be = st.columns(2)
    win_rate = col_win.number_input(T['win_rate'], value=55, help="Процент выигрышных сделок (0-100).")
    be_rate = col_be.number_input(T['be_rate'], value=5, help="Процент сделок в безубыток.")
    
    col_r, col_p = st.columns(2)
    risk_val = col_r.number_input(f"{T['risk']} ({mode})", value=1 if "%" in mode else 100, help="Риск на одну сделку.")
    reward_val = col_p.number_input(f"{T['reward']} ({mode})", value=2 if "%" in mode else 200, help="Целевая прибыль.")
    
    num_sims = st.number_input(T['num_sims'], value=50, step=1, min_value=1, max_value=10000)
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

    # ВАЛИДАЦИЯ
    if win_rate < 0 or win_rate > 100:
        st.error("Win Rate должен быть между 0 и 100%.")
        st.stop()
    if be_rate < 0 or be
