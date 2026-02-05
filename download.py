import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Professional Monte Carlo Sim", layout="wide")

# --- CSS ДЛЯ СТИЛЬНОЙ ПЛАШКИ МЕТРИК ---
st.markdown("""
    <style>
    [data-baseweb="tab-highlight"] { display: none !important; }
    
    .stTabs [data-baseweb="tab-list"] {
        display: flex; justify-content: center; gap: 12px;
        padding-bottom: 20px; border: none !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px; width: 250px; border-radius: 8px;
        font-weight: bold; font-size: 18px; color: white !important;
        border: none !important; transition: all 0.2s ease;
    }

    div[data-baseweb="tab-list"] button:nth-child(1) { background-color: #3B82F6 !important; }
    div[data-baseweb="tab-list"] button:nth-child(2) { background-color: #EF4444 !important; }
    div[data-baseweb="tab-list"] button:nth-child(3) { background-color: #10B981 !important; }

    /* ТА САМАЯ ВЕРХНЯЯ ПЛАШКА */
    .summary-bar {
        padding: 15px 30px;
        border-radius: 40px; /* Сильное скругление */
        margin-top: 10px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        justify-content: space-around;
    }
    
    .bar-median { background-color: rgba(59, 130, 246, 0.1); } /* Легкий синий */
    .bar-worst { background-color: rgba(239, 68, 68, 0.1); }  /* Легкий красный */
    .bar-best { background-color: rgba(16, 185, 129, 0.1); }   /* Легкий зеленый */

    /* Убираем стандартные рамки Streamlit внутри плашки */
    [data-testid="column"] { border: none !important; }
    
    .stTable { 
        background-color: rgba(255, 255, 255, 0.05) !important; 
        border-radius: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Симуляция Монте-Карло для трейдеров")

# --- ЛОГИКА (БЕЗ ИЗМЕНЕНИЙ) ---
def calculate_single_mdd(history):
    if not history or len(history) < 2: return 0.0
    h = np.array(history); peaks = np.maximum.accumulate(h)
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

with st.sidebar:
    st.header("Настройки")
    mode = st.radio("Режим:", ["Проценты (%)", "Доллары ($)"])
    start_balance = st.number_input("Начальный баланс", value=10000)
    win_rate = st.number_input("Winning trades %", value=55)
    be_rate = st.number_input("Break even trades %", value=5)
    risk_val = st.number_input(f"Риск ({mode[-2]})", value=100.0 if "$" in mode else 1.0)
    reward_val = st.number_input(f"Прибыль ({mode[-2]})", value=200.0 if "$" in mode else 2.0)
    num_sims = st.number_input("Количество симуляций", value=50)
    trades_per_month = st.slider("Сделок в месяц", 1, 50, 20)
    num_months = st.number_input("Месяцев", value=24)
    variability = st.slider("Вариативность RR (%)", 0, 100, 20)

def run_simulation():
    all_runs = []
    total_trades = int(num_months * trades_per_month)
    for _ in range(int(num_sims)):
        balance = float(start_balance)
        history = [balance]; trade_results = []; monthly_diffs = []
        current_month_start_bal = balance
        for t in range(1, total_trades + 1):
            if balance <= 0: balance = 0.0; history.append(balance); trade_results.append(-1); continue
            rn = np.random.random() * 100
            v_factor = np.random.normal(1, variability / 100)
            if rn < win_rate:
                change = (balance * (reward_val * v_factor / 100)) if "%" in mode else (reward_val * v_factor)
                balance += max(0.0, float(change)); trade_results.append(1)
            elif rn < (win_rate + be_rate): trade_results.append(0)
            else:
                change = (balance * (risk_val * v_factor / 100)) if "%" in mode else (risk_val * v_factor)
                balance -= max(0.0, float(change)); trade_results.append(-1)
            history.append(balance)
            if t % trades_per_month == 0:
                monthly_diffs.append(balance - current_month_start_bal)
                current_month_start_bal = balance
        max_w, max_l = get_consecutive(trade_results)
        all_runs.append({"history": history, "final": balance, "mdd": calculate_single_mdd(history),
                         "max_wins": max_w, "max_losses": max_l, 
                         "win_pct": (trade_results.count(1)/len(trade_results))*100, "monthly_diffs": monthly_diffs})
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
fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=20, b=0))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- АНАЛИЗ С ВЕРХНЕЙ ЦВЕТНОЙ ПОЛОСОЙ ---
st.write("<h2 style='text-align: center;'>Детальный анализ сценариев</h2>", unsafe_allow_html=True)
tab_med, tab_worst, tab_best = st.tabs(["MOST POSSIBLE", "WORST", "BEST"])

def style_table(df):
    def color_vals(val):
        if isinstance(val, str) and '-' in val: return 'color: #EF4444; font-weight: bold;'
        if isinstance(val, str) and '+' in val and val != '+0.0%': return 'color: #10B981; font-weight: bold;'
        return ''
    return df.style.applymap(color_vals)

def render_scenario(data, bar_class):
    # ТА САМАЯ ПОЛОСА С МЕТРИКАМИ
    st.markdown(f'<div class="summary-bar {bar_class}">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Initial balance", f"${start_balance:,.0f}")
    c2.metric("Result balance", f"${data['final']:,.0f}")
    c3.metric("Return %", f"{((data['final']-start_balance)/start_balance)*100:.1f}%")
    c4.metric("Max drawdown", f"-{data['mdd']:.1f}%")
    c5.metric("Max cons. loss", data['max_losses'])
    c6.metric("Max cons. win", data['max_wins'])
    c7.metric("Win trades %", f"{data['win_pct']:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # КАЛЕНДАРЬ НИЖЕ
    st.write("#### Результаты по месяцам (Календарь)")
    diffs = data['monthly_diffs']
    num_years = int(np.ceil(len(diffs) / 12))
    cols_years = st.columns(min(num_years, 3))
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    for y in range(num_years):
        with cols_years[y % 3]:
            year_data = diffs[y*12 : (y+1)*12]
            rows = []
            for i, val in enumerate(year_data):
                pct = (val / start_balance) * 100
                rows.append({"Month": months[i], "Results %": f"{pct:+.1f}%", "Results $": f"${val:+,.0f}".replace("$-", "-$")})
            df_year = pd.DataFrame(rows)
            df_year.index = df_year.index + 1
            st.write(f"**Year {2026 + y}**")
            st.table(style_table(df_year))

with tab_med: render_scenario(results[idx_median], "bar-median")
with tab_worst: render_scenario(results[idx_worst], "bar-worst")
with tab_best: render_scenario(results[idx_best], "bar-best")
