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

# --- CSS ---
st.markdown("""
    <style>
    [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-list"] { display: flex; justify-content: center; gap: 12px; padding-bottom: 20px; border: none !important; }
    .stTabs [data-baseweb="tab"] { height: 60px; width: 250px; border-radius: 8px; font-weight: bold; font-size: 18px; color: white !important; border: none !important; transition: all 0.2s ease; }
    div[data-baseweb="tab-list"] button:nth-child(1) { background-color: #3B82F6 !important; }
    div[data-baseweb="tab-list"] button:nth-child(2) { background-color: #EF4444 !important; }
    div[data-baseweb="tab-list"] button:nth-child(3) { background-color: #10B981 !important; }
    .stTabs [aria-selected="true"] { filter: brightness(1.2); transform: scale(1.02); box-shadow: 0px 5px 15px rgba(0,0,0,0.3); }
    .stTabs [data-baseweb="tab-list"] { border-bottom: none !important; }
    </style>
""", unsafe_allow_html=True)

# --- ВЫБОР ЯЗЫКА ---
lang_choice = st.sidebar.selectbox("Language / Язык", ["RU", "EN"])
T = languages[lang_choice]

st.title(f"{T['title']} by timuroff")

# --- ФУНКЦИИ ---
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
        if r == 1:
            cur_wins += 1; cur_losses = 0
        elif r == -1:
            cur_losses += 1; cur_wins = 0
        else:
            cur_wins, cur_losses = 0, 0
        max_wins = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses

# --- SIDEBAR ---
with st.sidebar:
    st.header(T['settings'])
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

# --- ЛОГИКА СИМУЛЯЦИИ ---
def run_simulation(n_sims, w_rate, silent=False):
    all_runs = []
    total_trades = int(num_months * trades_per_month)
    ruined_count = 0
    context = st.spinner(T['run_sim']) if not silent else st.empty()
    
    for _ in range(int(n_sims)):
        balance = float(start_balance)
        history = [balance]
        trade_results = []
        trade_amounts = []
        monthly_diffs = []
        current_month_start_bal = balance
        is_ruined = False
        
        for t in range(1, total_trades + 1):
            if balance <= 0:
                balance = 0.0
                history.append(balance)
                trade_results.append(-1)
                trade_amounts.append(0)
                continue
            if balance < ruin_threshold: is_ruined = True
            rn = np.random.random() * 100
            v_factor = np.random.normal(1, variability / 100)
            if rn < w_rate:
                change = (balance * (reward_val * v_factor / 100)) if "%" in mode else (reward_val * v_factor)
                balance += max(0.0, float(change))
                trade_results.append(1); trade_amounts.append(max(0.0, float(change)))
            elif rn < (w_rate + be_rate):
                trade_results.append(0); trade_amounts.append(0)
            else:
                change = (balance * (risk_val * v_factor / 100)) if "%" in mode else (risk_val * v_factor)
                balance -= max(0.0, float(change))
                trade_results.append(-1); trade_amounts.append(-max(0.0, float(change)))
            history.append(balance)
            if t % trades_per_month == 0:
                monthly_diffs.append(balance - current_month_start_bal)
                current_month_start_bal = balance
        
        if is_ruined: ruined_count += 1
        pos_trades = [a for a in trade_amounts if a > 0]
        neg_trades = [a for a in trade_amounts if a < 0]
        m_returns = np.array(monthly_diffs) / start_balance
        
        all_runs.append({
            "history": history, "final": balance, "mdd": calculate_single_mdd(history),
            "max_wins": get_consecutive(trade_results)[0], "max_losses": get_consecutive(trade_results)[1],
            "win_pct": (trade_results.count(1)/len(trade_results))*100, "monthly_diffs": monthly_diffs,
            "profit_factor": sum(pos_trades) / abs(sum(neg_trades)) if neg_trades else 10.0,
            "expectancy": np.mean(trade_amounts),
            "sharpe": (np.mean(m_returns) / np.std(m_returns) * np.sqrt(12)) if np.std(m_returns) > 0 else 0,
            "sortino": (np.mean(m_returns) / np.std(m_returns[m_returns<0]) * np.sqrt(12)) if len(m_returns[m_returns<0]) > 0 else 10.0,
            "avg_win": np.mean(pos_trades) if pos_trades else 0,
            "avg_loss": np.mean(neg_trades) if neg_trades else 0,
            "recovery_factor": (balance - start_balance) / ((calculate_single_mdd(history)/100 * start_balance) + 1e-9)
        })
    return all_runs, (ruined_count / n_sims) * 100

results, r_ruin_pct = run_simulation(num_sims, win_rate)
finals = [r["final"] for r in results]
idx_best, idx_worst = int(np.argmax(finals)), int(np.argmin(finals))
idx_median = int((np.abs(np.array(finals) - np.median(finals))).argmin())

# --- ГРАФИКИ ---
fig = go.Figure()
for i, r in enumerate(results[:100]):
    if i not in [idx_best, idx_worst, idx_median]:
        fig.add_trace(go.Scatter(y=r["history"], mode='lines', line=dict(width=1, color="gray"), opacity=0.1, showlegend=False))
fig.add_trace(go.Scatter(y=results[idx_median]["history"], name="MEDIAN", line=dict(color="#3B82F6", width=2)))
fig.add_trace(go.Scatter(y=results[idx_worst]["history"], name="WORST", line=dict(color="#EF4444", width=2)))
fig.add_trace(go.Scatter(y=results[idx_best]["history"], name="BEST", line=dict(color="#10B981", width=2)))
fig.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

c_ruin, c_p5, c_p95 = st.columns(3)
c_ruin.metric(T['risk_of_ruin'], f"{r_ruin_pct:.1f}%")
c_p5.metric("5% Percentile (Worst)", f"${np.percentile(finals, 5):,.0f}")
c_p95.metric("95% Percentile (Best)", f"${np.percentile(finals, 95):,.0f}")

fig_hist = go.Figure(go.Histogram(x=finals, nbinsx=30, marker_color='#3B82F6'))
fig_hist.update_layout(title=T['hist_title'], template="plotly_dark", height=300)
st.plotly_chart(fig_hist, use_container_width=True)

sens_x = list(range(40, 71, 5))
sens_y = [np.mean([r['final'] for r in run_simulation(15, wr, silent=True)[0]]) for wr in sens_x]
fig_sens = go.Figure(go.Scatter(x=sens_x, y=sens_y, mode='lines+markers', line=dict(color='#10B981')))
fig_sens.update_layout(title=f"{T['sensitivity']} (Win Rate vs Avg Final Balance)", template="plotly_dark", height=300)
st.plotly_chart(fig_sens, use_container_width=True)

st.divider()

# --- ТАБЛИЦЫ И МЕТРИКИ ---
def style_table(df):
    def apply_styles(row):
        styles = [''] * len(row)
        is_total = row['Month'] == T['year_total']
        try:
            val = float(str(row['Results $']).replace('$', '').replace(',', '').replace(' ', ''))
        except: val = 0
        if is_total:
            bg = 'background-color: rgba(16, 185, 129, 0.15)' if val >= 0 else 'background-color: rgba(239, 68, 68, 0.15)'
            styles = [bg] * len(row)
        color = 'color: #10B981' if val >= 0 else 'color: #EF4444'
        styles[1] += color; styles[2] += color
        return styles
    return df.style.apply(apply_styles, axis=1)

def render_scenario(data):
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Return %", f"{((data['final']-start_balance)/start_balance)*100:.1f}%")
        m2.metric("Profit Factor", f"{data['profit_factor']:.2f}")
        m3.metric("Sharpe Ratio", f"{data['sharpe']:.2f}")
        m4.metric("Sortino Ratio", f"{data['sortino']:.2f}")
        
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Expectancy", f"${data['expectancy']:.1f}")
        m6.metric("Recovery Factor", f"{data['recovery_factor']:.2f}")
        
        # ИСПРАВЛЕННАЯ СТРОКА: Теперь Avg Win/Loss визуально совпадает с остальными
        m7.metric("Avg Win/Loss", f"${data['avg_win']:.0f} / ${abs(data['avg_loss']):.0f}")
        
        m8.metric("Max DD", f"-{data['mdd']:.1f}%")

    diffs = data['monthly_diffs']
    num_years = int(np.ceil(len(diffs) / 12))
    for y in range(num_years):
        year_data = diffs[y*12 : (y+1)*12]
        rows = [{"Month": T['months_list'][i], "Results %": f"{(v/start_balance*100):.1f}%", "Results $": f"${v:,.0f}"} for i, v in enumerate(year_data)]
        df = pd.DataFrame(rows)
        df.index = [str(i+1) for i in range(len(year_data))]
        total_val = sum(year_data)
        total_row = pd.DataFrame([{"Month": T['year_total'], "Results %": f"{(total_val/start_balance*100):.1f}%", "Results $": f"${total_val:,.0f}"}], index=[" "])
        df = pd.concat([df, total_row])
        st.write(f"**Year {2026 + y}**")
        st.table(style_table(df))

tab_med, tab_worst, tab_best = st.tabs(["MOST POSSIBLE", "WORST CASE", "BEST CASE"])
with tab_med: render_scenario(results[idx_median])
with tab_worst: render_scenario(results[idx_worst])
with tab_best: render_scenario(results[idx_best])
