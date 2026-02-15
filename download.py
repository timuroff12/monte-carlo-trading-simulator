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
    /* ПУНКТ 2: Увеличен шрифт табов */
    .stTabs [data-baseweb="tab"] {{ height: 60px; width: 280px; border-radius: 8px; font-weight: bold; font-size: 22px; color: white !important; border: none !important; transition: all 0.2s ease; }}
    div[data-baseweb="tab-list"] button:nth-child(1) {{ background-color: #3B82F6 !important; }}
    div[data-baseweb="tab-list"] button:nth-child(2) {{ background-color: #EF4444 !important; }}
    div[data-baseweb="tab-list"] button:nth-child(3) {{ background-color: #10B981 !important; }}
    .stTabs [aria-selected="true"] {{ filter: brightness(1.2); transform: scale(1.02); box-shadow: 0px 5px 15px rgba(0,0,0,0.3); }}
    
    /* ПУНКТ 1: Настройка таблицы (цвет месяцев и размер шрифта) */
    .year-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 16px; }}
    .year-table th, .year-table td {{ border: 1px solid #444; padding: 10px; text-align: center; }}
    .year-table th {{ background-color: #262730; color: #E0E0E0; font-weight: normal; }}
    .year-table td {{ font-size: 17px; }}
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

def get_consecutive(results):
    max_wins, max_losses = 0, 0
    cur_wins, cur_losses = 0, 0
    for r in results:
        if r == 1: cur_wins += 1; cur_losses = 0
        elif r == -1: cur_losses += 1; cur_wins = 0
        else: cur_wins, cur_losses = 0, 0
        max_wins, max_losses = max(max_wins, cur_wins), max(max_losses, cur_losses)
    return max_wins, max_losses

def run_simulation(n_sims, w_rate, silent=False):
    all_runs = []
    total_trades = int(num_months * trades_per_month)
    ruined_count = 0
    
    # ПУНКТ 4: Замена синей линии на спиннер
    if not silent:
        status_text = st.empty()
        status_text.markdown(f"⏳ **{T['run_sim']}**")

    for i in range(int(n_sims)):
        balance = float(start_balance)
        balances, trade_results, trade_amounts = [balance], [], []
        is_ruined = False
        kelly_f = (w_rate/100 * (reward_val/risk_val) - (1 - w_rate/100)) / (reward_val/risk_val) if sizing_type == "Kelly" else 0

        for t in range(total_trades):
            if balance <= ruin_threshold: is_ruined = True
            if balance <= 0: balances.append(0.0); continue
            
            rn = np.random.random() * 100
            v_f = np.random.normal(1, variability / 100)
            curr_risk = (balance * kelly_f) if sizing_type == "Kelly" else risk_val
            curr_reward = curr_risk * (reward_val / risk_val)

            if rn < w_rate: res, change = 1, (balance * curr_reward / 100) if "%" in mode else (curr_reward * v_f)
            elif rn < (w_rate + be_rate): res, change = 0, 0
            else: res, change = -1, -(balance * curr_risk / 100) if "%" in mode else -(curr_risk * v_f)

            if res != 0:
                change -= commission_fixed + abs(change) * (commission_pct / 100)
                change -= np.random.normal(slippage, abs(slippage)/2 + 0.001)

            balance += change
            balances.append(max(0.0, balance))
            trade_results.append(res)
            trade_amounts.append(change)

        if is_ruined: ruined_count += 1
        pos_t = [a for a in trade_amounts if a > 0]
        neg_t = [a for a in trade_amounts if a < 0]
        m_diffs = [balances[m*trades_per_month] - balances[(m-1)*trades_per_month] for m in range(1, num_months+1)]
        m_returns = np.array(m_diffs) / start_balance
        max_c_w, max_c_l = get_consecutive(trade_results)

        all_runs.append({
            "history": balances, "final": balance, "mdd": calculate_single_mdd(balances),
            "max_wins": max_c_w, "max_losses": max_c_l,
            "win_pct": (trade_results.count(1)/len(trade_results))*100 if trade_results else 0,
            "monthly_diffs": m_diffs, "profit_factor": sum(pos_t) / abs(sum(neg_t)) if neg_t else 10.0,
            "expectancy": np.mean(trade_amounts) if trade_amounts else 0,
            "sharpe": (np.mean(m_returns) / (np.std(m_returns) + 1e-9) * np.sqrt(12)),
            "recovery_factor": (balance - start_balance) / ((calculate_single_mdd(balances)/100 * start_balance) + 1e-9),
            "cagr": ((balance/start_balance)**(1/(num_months/12)) - 1) * 100 if balance > 0 else -100
        })
    
    if not silent:
        status_text.empty() # Убираем текст загрузки
        
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

fig.add_trace(go.Scatter(y=results[idx_median]["history"], 
                         name=f"MEDIAN: ${results[idx_median]['final']:,.0f}", 
                         line=dict(color="#3B82F6", width=3)))
fig.add_trace(go.Scatter(y=results[idx_worst]["history"], 
                         name=f"WORST: ${results[idx_worst]['final']:,.0f}", 
                         line=dict(color="#EF4444", width=3)))
fig.add_trace(go.Scatter(y=results[idx_best]["history"], 
                         name=f"BEST: ${results[idx_best]['final']:,.0f}", 
                         line=dict(color="#10B981", width=3)))

fig.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=20, b=20), 
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
st.plotly_chart(fig, use_container_width=True)

# --- ФУНКЦИЯ ОТОБРАЖЕНИЯ СЦЕНАРИЯ ---
def render_scenario(data, label, color):
    # ПУНКТ 3: Уменьшен размер шрифта в баннере (был h3, стал h4 с фикс. размером)
    st.markdown(f"""
        <div style="background-color:{color}; padding:8px; border-radius:5px; margin-bottom:20px; text-align:center;">
            <h4 style="color:white; margin:0; font-size: 20px; letter-spacing: 1px;">{label.upper()} SCENARIO</h4>
        </div>
    """, unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric(T['risk_of_ruin'], f"{r_ruin_pct:.1f}%")
    m2.metric("5% Percentile (Worst)", f"${np.percentile(finals, 5):,.0f}")
    m3.metric("95% Percentile (Best)", f"${np.percentile(finals, 95):,.0f}")
    st.write("---")

    r1_1, r1_2, r1_3, r1_4 = st.columns(4)
    r1_1.metric("Return %", f"{((data['final']-start_balance)/start_balance)*100:.1f}%")
    r1_2.metric("Profit Factor", f"{data['profit_factor']:.2f}")
    r1_3.metric("Sharpe Ratio", f"{data['sharpe']:.2f}")
    r1_4.metric("CAGR", f"{data['cagr']:.1f}%")
    
    r2_1, r2_2, r2_3, r2_4 = st.columns(4)
    r2_1.metric("Expectancy", f"${data['expectancy']:.1f}")
    r2_2.metric("Recovery Factor", f"{data['recovery_factor']:.2f}")
    r2_3.metric("Max DD %", f"-{data['mdd']:.1f}%")
    r2_4.metric("Actual WinRate", f"{data['win_pct']:.1f}%")

    st.write("---")
    
    diffs = data['monthly_diffs']
    num_years = int(np.ceil(len(diffs) / 12))
    
    for y in range(num_years):
        year_val = 2026 + y
        st.write(f"#### Year {year_val}")
        year_data = diffs[y*12 : (y+1)*12]
        
        html_table = f'<table class="year-table"><tr>'
        for m_name in T['months_list']:
            html_table += f'<th>{m_name}</th>'
        html_table += f'<th>{T["year_total"]}</th></tr><tr>'
        
        year_sum = 0
        for i in range(12):
            if i < len(year_data):
                val = year_data[i]
                year_sum += val
                pct = (val / start_balance) * 100
                style = "pos-val" if val >= 0 else "neg-val"
                html_table += f'<td><div class="{style}">{pct:+.1f}%</div><div style="font-size:11px; opacity:0.8;">${val:,.0f}</div></td>'
            else:
                html_table += '<td>-</td>'
        
        total_pct = (year_sum / start_balance) * 100
        t_style = "pos-val" if year_sum >= 0 else "neg-val"
        html_table += f'<td style="background-color:#333"><div class="{t_style}">{total_pct:+.1f}%</div><div style="font-size:11px; opacity:0.8;">${year_sum:,.0f}</div></td>'
        html_table += '</tr></table>'
        
        st.markdown(html_table, unsafe_allow_html=True)

# --- ТАБЫ ---
tab_med, tab_worst, tab_best = st.tabs(["MOST POSSIBLE", "WORST CASE", "BEST CASE"])
with tab_med: render_scenario(results[idx_median], "Median", "#3B82F6")
with tab_worst: render_scenario(results[idx_worst], "Worst", "#EF4444")
with tab_best: render_scenario(results[idx_best], "Best", "#10B981")

# Распределение
st.divider()
col_h, col_b = st.columns(2)
with col_h:
    fig_h = go.Figure(go.Histogram(x=finals, nbinsx=30, marker_color='#3B82F6'))
    fig_h.update_layout(title=T['hist_title'], template="plotly_dark", height=300)
    st.plotly_chart(fig_h, use_container_width=True)
with col_b:
    fig_m = go.Figure(go.Histogram(x=[r['mdd'] for r in results], nbinsx=30, marker_color='#EF4444'))
    fig_m.update_layout(title="MDD Distribution", template="plotly_dark", height=300)
    st.plotly_chart(fig_m, use_container_width=True)

with st.expander("FAQ / Что это такое?"):
    st.markdown("""
    ### Справка по инструменту:
    - **Монте-Карло симуляция**: Метод моделирования, использующий случайные числа для создания тысяч вариантов будущего счета.
    - **Risk of Ruin**: Вероятность падения баланса ниже заданного порога.
    """)
