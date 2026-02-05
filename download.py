import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Monte Carlo Sim", layout="wide")

# --- CSS ДЛЯ КОМПАКТНОСТИ И ВКЛАДОК ---
st.markdown("""
    <style>
    [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-list"] {
        display: flex; justify-content: center; gap: 8px;
        padding-bottom: 15px; border: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px; width: 200px; border-radius: 6px;
        font-weight: bold; font-size: 14px; color: white !important;
        border: none !important; transition: all 0.2s ease;
    }
    div[data-baseweb="tab-list"] button:nth-child(1) { background-color: #3B82F6 !important; }
    div[data-baseweb="tab-list"] button:nth-child(2) { background-color: #EF4444 !important; }
    div[data-baseweb="tab-list"] button:nth-child(3) { background-color: #10B981 !important; }
    
    /* Уменьшаем отступы в метриках для компактности */
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; }
    
    /* Убираем лишние отступы у контейнеров */
    .block-container { padding-top: 2rem !important; }
    </style>
""", unsafe_allow_html=True)

# --- ФУНКЦИИ РАСЧЕТА ---
def calculate_single_mdd(history):
    if not history or len(history) < 2: return 0.0
    h = np.array(history)
    peaks = np.maximum.accumulate(h)
    drawdowns = (peaks - h) / (peaks + 1e-9)
    return float(np.max(drawdowns) * 100)

def get_consecutive(results):
    max_wins, max_losses = 0, 0
    cur_wins, cur_losses = 0, 0
    for r in results:
        if r == 1: cur_wins += 1; cur_losses = 0
        elif r == -1: cur_losses += 1; cur_wins = 0
        else: cur_wins, cur_losses = 0, 0
        max_wins = max(max_wins, cur_wins); max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses

# --- SIDEBAR (КОМПАКТНЫЙ) ---
with st.sidebar:
    st.subheader("⚙️ Настройки")
    mode = st.radio("Режим:", ["Проценты (%)", "Доллары ($)"], horizontal=True)
    start_balance = st.number_input("Баланс", value=10000, step=1000)
    
    c1, c2 = st.columns(2)
    win_rate = c1.number_input("Win %", value=55)
    be_rate = c2.number_input("BE %", value=5)
    
    c3, c4 = st.columns(2)
    risk_val = c3.number_input(f"Риск", value=1.0 if "%" in mode else 100.0)
    reward_val = c4.number_input(f"Прибыль", value=2.0 if "%" in mode else 200.0)
    
    num_sims = st.number_input("Симуляций", value=50, step=10)
    
    c5, c6 = st.columns(2)
    trades_per_month = c5.number_input("Сделок/мес", value=20)
    num_months = c6.number_input("Месяцев", value=24)
    
    variability = st.slider("Вариативность RR %", 0, 100, 20)

# --- ЛОГИКА ---
def run_simulation():
    all_runs = []
    total_trades = int(num_months * trades_per_month)
    for _ in range(int(num_sims)):
        balance = float(start_balance)
        history, trade_results, trade_diffs = [balance], [], []
        curr_m_start = balance
        monthly_diffs = []
        
        for t in range(1, total_trades + 1):
            if balance <= 0:
                balance = 0.0; history.append(0.0); trade_results.append(-1); trade_diffs.append(0); continue
            
            rn = np.random.random() * 100
            v_factor = np.random.normal(1, variability / 100)
            
            if rn < win_rate:
                ch = (balance * (reward_val * v_factor / 100)) if "%" in mode else (reward_val * v_factor)
                balance += max(0.0, float(ch))
                trade_results.append(1); trade_diffs.append(ch)
            elif rn < (win_rate + be_rate):
                trade_results.append(0); trade_diffs.append(0)
            else:
                ch = (balance * (risk_val * v_factor / 100)) if "%" in mode else (risk_val * v_factor)
                balance -= max(0.0, float(ch))
                trade_results.append(-1); trade_diffs.append(-ch)
            
            history.append(balance)
            if t % trades_per_month == 0:
                monthly_diffs.append(balance - curr_m_start)
                curr_m_start = balance
        
        mdd = calculate_single_mdd(history)
        wins = [v for v in trade_diffs if v > 0]
        losses = [abs(v) for v in trade_diffs if v < 0]
        max_w, max_l = get_consecutive(trade_results)
        
        all_runs.append({
            "history": history, "final": balance, "mdd": mdd,
            "max_wins": max_w, "max_losses": max_l,
            "p_factor": sum(wins)/sum(losses) if sum(losses) > 0 else 0,
            "rec_factor": (balance - start_balance)/(start_balance * (mdd/100)) if mdd > 0 else 0,
            "expectancy": sum(trade_diffs)/len(trade_results) if trade_results else 0,
            "win_pct": (trade_results.count(1)/len(trade_results))*100, "monthly_diffs": monthly_diffs
        })
    return all_runs

results = run_simulation()
finals = [r["final"] for r in results]
idx_best, idx_worst = int(np.argmax(finals)), int(np.argmin(finals))
idx_median = int((np.abs(np.array(finals) - np.median(finals))).argmin())

# --- ГРАФИК ---
fig = go.Figure()
fig.add_trace(go.Scatter(y=results[idx_median]["history"], name="MOST POSSIBLE", line=dict(color="#3B82F6", width=2)))
fig.add_trace(go.Scatter(y=results[idx_worst]["history"], name="WORST CASE", line=dict(color="#EF4444", width=2)))
fig.add_trace(go.Scatter(y=results[idx_best]["history"], name="BEST CASE", line=dict(color="#10B981", width=2)))
fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)

# --- АНАЛИЗ ---
tab_med, tab_worst, tab_best = st.tabs(["MOST POSSIBLE", "WORST", "BEST"])

def render_scenario(data):
    # Компактная сетка метрик без цветных фонов
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final Balance", f"${data['final']:,.0f}")
        c2.metric("Return", f"{((data['final']-start_balance)/start_balance)*100:+.1f}%")
        c3.metric("Max Drawdown", f"-{data['mdd']:.1f}%")
        c4.metric("Profit Factor", f"{data['p_factor']:.2f}")
        c5.metric("Win Rate", f"{data['win_pct']:.1f}%")
        
        c6, c7, c8, c9, c10 = st.columns(5)
        c6.metric("Recovery Factor", f"{data['rec_factor']:.2f}")
        c7.metric("Expectancy", f"${data['expectancy']:,.1f}" if mode == "Доллары ($)" else f"{data['expectancy']:,.1f}%")
        c8.metric("Cons. Wins", data['max_wins'])
        c9.metric("Cons. Losses", data['max_losses'])
        c10.metric("Initial", f"${start_balance:,.0f}")

    # Таблицы месяцев
    diffs = data['monthly_diffs']
    num_years = int(np.ceil(len(diffs) / 12))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    cols = st.columns(min(num_years, 3))
    for y in range(num_years):
        with cols[y % 3]:
            year_data = diffs[y*12 : (y+1)*12]
            df = pd.DataFrame([{"Month": months[i], "Res %": f"{(v/start_balance)*100:+.1f}%", "Res $": f"${v:+,.0f}"} for i, v in enumerate(year_data)])
            st.caption(f"Year {2026 + y}")
            st.table(df)

with tab_med: render_scenario(results[idx_median])
with tab_worst: render_scenario(results[idx_worst])
with tab_best: render_scenario(results[idx_best])
