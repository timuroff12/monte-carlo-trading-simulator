import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Настройка страницы
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
        "year_total": "Итого за год:",
        "risk_of_ruin": "Риск разорения",
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
        "year_total": "Year Total:",
        "risk_of_ruin": "Risk of Ruin",
        "hist_title": "Final Balance Distribution",
        "months_list": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    }
}

lang_choice = st.sidebar.selectbox("Language / Язык", ["RU", "EN"])
T = languages[lang_choice]

# --- SIDEBAR ---
with st.sidebar:
    st.header(T['settings'])
    
    # 4. Переключатель режима справа
    s_col1, s_col2 = st.columns([1, 1])
    with s_col2:
        mode = st.radio(T['mode'], ["%", "$"], horizontal=True)
    
    start_balance = st.number_input(T['start_bal'], value=10000, step=1000)
    
    col_win, col_be = st.columns(2)
    win_rate = col_win.number_input(T['win_rate'], value=55)
    be_rate = col_be.number_input(T['be_rate'], value=5)
    
    col_r, col_p = st.columns(2)
    risk_val = col_r.number_input(f"{T['risk']} ({mode})", value=1 if "%" in mode else 100)
    reward_val = col_p.number_input(f"{T['reward']} ({mode})", value=2 if "%" in mode else 200)
    
    num_sims = st.number_input(T['num_sims'], value=100, step=10)
    trades_per_month = st.slider(T['trades_month'], 1, 50, 20)
    num_months = st.number_input(T['months'], value=24, step=1)
    variability = st.slider(T['variability'], 0, 100, 20)
    ruin_threshold = st.number_input(T['ruin_threshold'], value=int(start_balance * 0.1))

    st.subheader("Advanced")
    commission_fixed = st.number_input("Комиссия фикс. ($)", value=0.0)
    commission_pct = st.number_input("Комиссия %", value=0.0)
    slippage = st.number_input("Слиппедж ($/%)", value=0.0)
    sizing_type = st.selectbox("Position Sizing", ["Fixed Risk", "Kelly"])

# --- CSS (Исправленный блок) ---
st.markdown(f"""
    <style>
    div[class*="stMain"] h1 {{ border-bottom: none !important; padding-bottom: 0.5rem !important; }}
    .stTabs [data-baseweb="tab-list"] {{ display: flex; justify-content: center; gap: 12px; padding-bottom: 20px; }}
    .stTabs [data-baseweb="tab"] {{ height: 60px; width: 280px; border-radius: 8px; font-weight: bold; font-size: 20px; color: white !important; }}
    div[data-baseweb="tab-list"] button:nth-child(1) {{ background-color: #3B82F6 !important; }}
    div[data-baseweb="tab-list"] button:nth-child(2) {{ background-color: #EF4444 !important; }}
    div[data-baseweb="tab-list"] button:nth-child(3) {{ background-color: #10B981 !important; }}
    .year-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
    .year-table th, .year-table td {{ border: 1px solid #444; padding: 10px; text-align: center; }}
    .pos-val {{ color: #10B981; font-weight: bold; }}
    .neg-val {{ color: #EF4444; font-weight: bold; }}
    </style>
""", unsafe_allow_html=True)

st.title(f"{T['title']} by timuroff")

# --- ФУНКЦИИ РАСЧЕТА ---
def calculate_mdd(history):
    h = np.array(history)
    peaks = np.maximum.accumulate(h)
    drawdowns = (peaks - h) / (peaks + 1e-9)
    return float(np.max(drawdowns) * 100)

def calculate_mdd_init(history, initial_bal):
    h = np.array(history)
    drawdowns = (initial_bal - h) / (initial_bal + 1e-9)
    return float(np.max(drawdowns) * 100)

def run_simulation(n_sims, w_rate):
    all_runs = []
    total_trades = int(num_months * trades_per_month)
    ruined_count = 0
    status = st.empty()
    status.markdown(f"⏳ **{T['run_sim']}**")

    for i in range(int(n_sims)):
        balance = float(start_balance)
        balances, trade_results, trade_amounts = [balance], [], []
        ruined_this_sim = False
        
        kelly_f = (w_rate/100 * (reward_val/risk_val) - (1 - w_rate/100)) / (reward_val/risk_val) if sizing_type == "Kelly" else 0

        for t in range(total_trades):
            if balance <= ruin_threshold: ruined_this_sim = True
            if balance <= 0: balances.append(0.0); continue
            
            rn = np.random.random() * 100
            v_f = np.random.normal(1, variability / 100)
            curr_risk = (balance * kelly_f) if sizing_type == "Kelly" else risk_val
            curr_reward = curr_risk * (reward_val / risk_val)

            if rn < w_rate: 
                change = (balance * curr_reward / 100) if "%" in mode else (curr_reward * v_f)
                res = 1
            elif rn < (w_rate + be_rate): 
                change = 0
                res = 0
            else: 
                change = -(balance * curr_risk / 100) if "%" in mode else -(curr_risk * v_f)
                res = -1

            if res != 0:
                change -= (commission_fixed + abs(change) * (commission_pct / 100) + slippage)

            balance += change
            balances.append(max(0.0, balance))
            trade_amounts.append(change)
            trade_results.append(res)

        if ruined_this_sim: ruined_count += 1
        pos_t = [a for a in trade_amounts if a > 0]
        neg_t = [a for a in trade_amounts if a < 0]
        m_diffs = [balances[m*trades_per_month] - balances[(m-1)*trades_per_month] for m in range(1, num_months+1)]
        
        all_runs.append({
            "history": balances, "final": balance,
            "mdd": calculate_mdd(balances), "mdd_init": calculate_mdd_init(balances, start_balance),
            "win_pct": (trade_results.count(1)/len(trade_results))*100 if trade_results else 0,
            "monthly_diffs": m_diffs, "profit_factor": sum(pos_t) / (abs(sum(neg_t)) + 1e-9) if neg_t else 10.0,
            "expectancy": np.mean(trade_amounts) if trade_amounts else 0,
            "sharpe": (np.mean(m_diffs)/ (np.std(m_diffs)+1e-9)) * np.sqrt(12)
        })
    status.empty()
    return all_runs, (ruined_count / n_sims) * 100

results, r_ruin_total = run_simulation(num_sims, win_rate)
finals = [r["final"] for r in results]
idx_best, idx_worst = int(np.argmax(finals)), int(np.argmin(finals))
idx_median = int((np.abs(np.array(finals) - np.median(finals))).argmin())

# --- ГРАФИК И ПЕРЦЕНТИЛИ ---
fig = go.Figure()
for i, r in enumerate(results[:100]):
    if i not in [idx_best, idx_worst, idx_median]:
        fig.add_trace(go.Scatter(y=r["history"], mode='lines', line=dict(width=1, color="gray"), opacity=0.1, showlegend=False))
fig.add_trace(go.Scatter(y=results[idx_median]["history"], name=f"MEDIAN: ${results[idx_median]['final']:,.0f}", line=dict(color="#3B82F6", width=3)))
fig.add_trace(go.Scatter(y=results[idx_worst]["history"], name=f"WORST: ${results[idx_worst]['final']:,.0f}", line=dict(color="#EF4444", width=3)))
fig.add_trace(go.Scatter(y=results[idx_best]["history"], name=f"BEST: ${results[idx_best]['final']:,.0f}", line=dict(color="#10B981", width=3)))
fig.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=20, b=20))
st.plotly_chart(fig, use_container_width=True)

# 2. Перцентили сразу под графиком
p_col1, p_col2 = st.columns(2)
p_col1.metric("5% Percentile (Worst)", f"${np.percentile(finals, 5):,.0f}")
p_col2.metric("95% Percentile (Best)", f"${np.percentile(finals, 95):,.0f}")

# --- ОТОБРАЖЕНИЕ СЦЕНАРИЕВ ---
def render_scenario(data, label, color):
    st.markdown(f'<div style="background-color:{color}; padding:8px; border-radius:5px; margin-bottom:20px; text-align:center;"><h4 style="color:white; margin:0; font-size: 20px;">{label.upper()} SCENARIO</h4></div>', unsafe_allow_html=True)
    
    # 1. Risk of Ruin внутри блока
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(T['risk_of_ruin'], f"{r_ruin_total:.1f}%")
    m2.metric("Return %", f"{((data['final']-start_balance)/start_balance)*100:.1f}%")
    m3.metric("Profit Factor", f"{data['profit_factor']:.2f}")
    m4.metric("Sharpe Ratio", f"{data['sharpe']:.2f}")
    
    st.write("---")
    
    # 3. Дополнительные метрики (включая Max DD Init)
    r2_1, r2_2, r2_3, r2_4 = st.columns(4)
    r2_1.metric("Expectancy", f"${data['expectancy']:.1f}")
    r2_2.metric("Max DD %", f"-{data['mdd']:.1f}%")
    r2_3.metric("Max DD (Init Balance)", f"-{data['mdd_init']:.1f}%") # Ваш запрос №3
    r2_4.metric("Actual WinRate", f"{data['win_pct']:.1f}%")

    st.write("---")
    
    diffs = data['monthly_diffs']
    num_years = int(np.ceil(len(diffs) / 12))
    for y in range(num_years):
        year_val = 2026 + y
        st.write(f"#### Year {year_val}")
        year_data = diffs[y*12 : (y+1)*12]
        html_table = f'<table class="year-table"><tr>'
        for m_name in T['months_list']: html_table += f'<th>{m_name}</th>'
        html_table += f'<th>Total</th></tr><tr>'
        y_sum = 0
        for i in range(12):
            if i < len(year_data):
                val = year_data[i]; y_sum += val
                style = "pos-val" if val >= 0 else "neg-val"
                html_table += f'<td><div class="{style}">{(val/start_balance)*100:+.1f}%</div><div style="font-size:11px; opacity:0.8;">${val:,.0f}</div></td>'
            else: html_table += '<td>-</td>'
        t_style = "pos-val" if y_sum >= 0 else "neg-val"
        html_table += f'<td style="background-color:#333"><div class="{t_style}">{(y_sum/start_balance)*100:+.1f}%</div><div style="font-size:11px; opacity:0.8;">${y_sum:,.0f}</div></td>'
        html_table += '</tr></table>'
        st.markdown(html_table, unsafe_allow_html=True)

tab_med, tab_worst, tab_best = st.tabs(["MOST POSSIBLE", "WORST CASE", "BEST CASE"])
with tab_med: render_scenario(results[idx_median], "Median", "#3B82F6")
with tab_worst: render_scenario(results[idx_worst], "Worst", "#EF4444")
with tab_best: render_scenario(results[idx_best], "Best", "#10B981")

# --- СТАТИСТИКА И FAQ ---
st.divider()
c_h1, c_h2 = st.columns(2)
with c_h1:
    fig_h = go.Figure(go.Histogram(x=finals, marker_color='#3B82F6'))
    fig_h.update_layout(title=T['hist_title'], template="plotly_dark", height=300)
    st.plotly_chart(fig_h, use_container_width=True)
with c_h2:
    fig_m = go.Figure(go.Histogram(x=[r['mdd'] for r in results], marker_color='#EF4444'))
    fig_m.update_layout(title="Max Drawdown Distribution", template="plotly_dark", height=300)
    st.plotly_chart(fig_m, use_container_width=True)

with st.expander("FAQ / Что это такое?"):
    st.markdown("""
    **Max DD (Init Balance)** — показывает максимальную просадку счета относительно вашего первоначального капитала. 
    Если это значение -50%, значит в какой-то момент ваш баланс падал до половины от стартовой суммы.
    """)
