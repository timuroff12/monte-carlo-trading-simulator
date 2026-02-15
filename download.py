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

lang_choice = st.sidebar.selectbox("Language / Язык", ["RU", "EN"])
T = languages[lang_choice]

# --- SIDEBAR ---
with st.sidebar:
    st.header(T['settings'])
    
    # ПУНКТ 4: Mode $ перекинут направо через колонки
    col_l, col_r = st.columns([1, 1])
    with col_r:
        mode = st.radio(T['mode'], ["%", "$"], horizontal=True)
    
    start_balance = st.number_input(T['start_bal'], value=10000, step=1000)
    
    col_win, col_be = st.columns(2)
    win_rate = col_win.number_input(T['win_rate'], value=55)
    be_rate = col_be.number_input(T['be_rate'], value=5)
    
    col_r_val, col_p_val = st.columns(2)
    risk_val = col_r_val.number_input(f"{T['risk']} ({mode})", value=1 if "%" in mode else 100)
    reward_val = col_p_val.number_input(f"{T['reward']} ({mode})", value=2 if "%" in mode else 200)
    
    num_sims = st.number_input(T['num_sims'], value=50, step=1)
    trades_per_month = st.slider(T['trades_month'], 1, 50, 20)
    num_months = st.number_input(T['months'], value=24, step=1)
    variability = st.slider(T['variability'], 0, 100, 20)
    ruin_threshold = st.number_input(T['ruin_threshold'], value=int(start_balance * 0.1))

# --- CSS ---
st.markdown(f"""
    <style>
    div[class*="stMain"] h1 {{ border-bottom: none !important; padding-bottom: 0.5rem !important; }}
    .stTabs [data-baseweb="tab"] {{ height: 60px; width: 280px; border-radius: 8px; font-weight: bold; font-size: 22px; color: white !important; }}
    div[data-baseweb="tab-list"] button:nth-child(1) {{ background-color: #3B82F6 !important; }}
    div[data-baseweb="tab-list"] button:nth-child(2) {{ background-color: #EF4444 !important; }}
    div[data-baseweb="tab-list"] button:nth-child(3) {{ background-color: #10B981 !important; }}
    .year-table {{ width: 100%; border-collapse: collapse; font-size: 16px; }}
    .year-table td {{ font-size: 17px; border: 1px solid #444; padding: 10px; text-align: center; }}
    .pos-val {{ color: #10B981; font-weight: bold; }}
    .neg-val {{ color: #EF4444; font-weight: bold; }}
    </style>
""", unsafe_allow_html=True)

st.title(f"{T['title']} by timuroff")

# --- ФУНКЦИИ ---
def calculate_single_mdd(history):
    h = np.array(history)
    peaks = np.maximum.accumulate(h)
    drawdowns = (peaks - h) / (peaks + 1e-9)
    return float(np.max(drawdowns) * 100)

# ПУНКТ 3: Функция расчета просадки от начального баланса
def calculate_mdd_from_init(history, start_bal):
    h = np.array(history)
    peaks = np.maximum.accumulate(h)
    drawdowns = (peaks - h) / start_bal
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

def run_simulation(n_sims, w_rate):
    all_runs = []
    total_trades = int(num_months * trades_per_month)
    ruined_count = 0
    status_text = st.empty()
    status_text.markdown(f"⏳ **{T['run_sim']}**")

    for i in range(int(n_sims)):
        balance = float(start_balance)
        balances, trade_results, trade_amounts = [balance], [], []
        is_ruined = False

        for t in range(total_trades):
            if balance <= ruin_threshold: is_ruined = True
            if balance <= 0: balances.append(0.0); continue
            
            rn = np.random.random() * 100
            v_f = np.random.normal(1, variability / 100)
            curr_risk = risk_val
            curr_reward = curr_risk * (reward_val / risk_val)

            if rn < w_rate: change = (balance * curr_reward / 100) if "%" in mode else (curr_reward * v_f); res = 1
            elif rn < (w_rate + be_rate): change = 0; res = 0
            else: change = -(balance * curr_risk / 100) if "%" in mode else -(curr_risk * v_f); res = -1

            balance += change
            balances.append(max(0.0, balance))
            trade_results.append(res)
            trade_amounts.append(change)

        if is_ruined: ruined_count += 1
        pos_t = [a for a in trade_amounts if a > 0]
        neg_t = [a for a in trade_amounts if a < 0]
        
        all_runs.append({
            "history": balances, "final": balance, 
            "mdd": calculate_single_mdd(balances),
            "mdd_init": calculate_mdd_from_init(balances, start_balance), # ПУНКТ 3
            "win_pct": (trade_results.count(1)/len(trade_results))*100 if trade_results else 0,
            "monthly_diffs": [balances[m*trades_per_month] - balances[(m-1)*trades_per_month] for m in range(1, num_months+1)],
            "profit_factor": sum(pos_t) / abs(sum(neg_t)) if neg_t else 10.0,
            "expectancy": np.mean(trade_amounts) if trade_amounts else 0,
            "sharpe": (np.mean(trade_amounts) / (np.std(trade_amounts) + 1e-9) * np.sqrt(total_trades/num_months*12)),
            "recovery_factor": (balance - start_balance) / ((calculate_single_mdd(balances)/100 * start_balance) + 1e-9),
            "cagr": ((balance/start_balance)**(1/(num_months/12)) - 1) * 100 if balance > 0 else -100
        })
    status_text.empty()
    return all_runs, (ruined_count / n_sims) * 100

results, r_ruin_pct = run_simulation(num_sims, win_rate)
finals = [r["final"] for r in results]
idx_best, idx_worst = int(np.argmax(finals)), int(np.argmin(finals))
idx_median = int((np.abs(np.array(finals) - np.median(finals))).argmin())

# --- ГРАФИК ---
fig = go.Figure()
for i, r in enumerate(results[:100]):
    if i not in [idx_best, idx_worst, idx_median]:
        fig.add_trace(go.Scatter(y=r["history"], mode='lines', line=dict(width=1, color="gray"), opacity=0.1, showlegend=False))

fig.add_trace(go.Scatter(y=results[idx_median]["history"], name=f"MEDIAN", line=dict(color="#3B82F6", width=3)))
fig.add_trace(go.Scatter(y=results[idx_worst]["history"], name=f"WORST", line=dict(color="#EF4444", width=3)))
fig.add_trace(go.Scatter(y=results[idx_best]["history"], name=f"BEST", line=dict(color="#10B981", width=3)))
fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)

# ПУНКТ 2: Перцентили сразу после графика с простым описанием
st.subheader("📊 Вероятные границы вашего результата")
p_worst = np.percentile(finals, 5)
p_best = np.percentile(finals, 95)
col_p1, col_p2 = st.columns(2)
col_p1.metric("5% Percentile (Худший реалистичный)", f"${p_worst:,.0f}")
col_p2.metric("95% Percentile (Лучший реалистичный)", f"${p_best:,.0f}")

st.info(f"**Что это значит?** Представьте 100 вариантов будущего. В 95 из них ваш баланс будет **ВЫШЕ** чем ${p_worst:,.0f}. И только в 5 самых удачных случаях вы заработаете **БОЛЬШЕ** чем ${p_best:,.0f}. Ваша зона комфорта находится между этими числами.")

# --- ФУНКЦИЯ ОТОБРАЖЕНИЯ СЦЕНАРИЯ ---
def render_scenario(data, label, color):
    st.markdown(f'<div style="background-color:{color}; padding:8px; border-radius:5px; text-align:center;"><h4 style="color:white; margin:0;">{label.upper()} SCENARIO</h4></div>', unsafe_allow_html=True)
    st.write("")
    
    # ПУНКТ 1 и 3: Risk of Ruin и Max DD Init включены в общую сетку
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(T['risk_of_ruin'], f"{r_ruin_pct:.1f}%")
    c2.metric("Return %", f"{((data['final']-start_balance)/start_balance)*100:.1f}%")
    c3.metric("Profit Factor", f"{data['profit_factor']:.2f}")
    c4.metric("Sharpe Ratio", f"{data['sharpe']:.2f}")
    c5.metric("CAGR", f"{data['cagr']:.1f}%")
    
    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Expectancy", f"${data['expectancy']:.1f}")
    c7.metric("Recovery Factor", f"{data['recovery_factor']:.2f}")
    c8.metric("Max DD %", f"-{data['mdd']:.1f}%")
    c9.metric("Max DD (Init)", f"-{data['mdd_init']:.1f}%") # ПУНКТ 3
    c10.metric("Actual WinRate", f"{data['win_pct']:.1f}%")

    # Месячная таблица
    diffs = data['monthly_diffs']
    html_table = f'<table class="year-table"><tr>'
    for m_name in T['months_list']: html_table += f'<th>{m_name}</th>'
    html_table += f'<th>Total</th></tr><tr>'
    for val in diffs[:12]:
        style = "pos-val" if val >= 0 else "neg-val"
        html_table += f'<td><div class="{style}">{(val/start_balance)*100:+.1f}%</div><div style="font-size:10px;">${val:,.0f}</div></td>'
    html_table += '</tr></table>'
    st.markdown(html_table, unsafe_allow_html=True)

# --- ТАБЫ ---
tab_med, tab_worst, tab_best = st.tabs(["MOST POSSIBLE", "WORST CASE", "BEST CASE"])
with tab_med: render_scenario(results[idx_median], "Median", "#3B82F6")
with tab_worst: render_scenario(results[idx_worst], "Worst", "#EF4444")
with tab_best: render_scenario(results[idx_best], "Best", "#10B981")
