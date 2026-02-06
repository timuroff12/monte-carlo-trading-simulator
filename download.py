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

# --- CSS И ТЕМА ---
theme = st.sidebar.radio("Theme / Тема", ["Dark", "Light"])
theme_css = ""
plotly_template = "plotly_dark"

if theme == "Light":
    plotly_template = "plotly_white"
    theme_css = """
    <style>
    .stApp { background-color: white; color: black; }
    section[data-testid="stSidebar"] { background-color: #f0f2f6; }
    </style>
    """

st.markdown(f"""
    <style>
    [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-list"] {{ display: flex; justify-content: center; gap: 12px; padding-bottom: 20px; border: none !important; }}
    .stTabs [data-baseweb="tab"] {{ height: 60px; width: 250px; border-radius: 8px; font-weight: bold; font-size: 18px; color: white !important; border: none !important; transition: all 0.2s ease; }}
    div[data-baseweb="tab-list"] button:nth-child(1) {{ background-color: #3B82F6 !important; }}
    div[data-baseweb="tab-list"] button:nth-child(2) {{ background-color: #EF4444 !important; }}
    div[data-baseweb="tab-list"] button:nth-child(3) {{ background-color: #10B981 !important; }}
    .stTabs [aria-selected="true"] {{ filter: brightness(1.2); transform: scale(1.02); box-shadow: 0px 5px 15px rgba(0,0,0,0.3); }}
    </style>
    {theme_css}
""", unsafe_allow_html=True)

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
    
    with st.expander("FAQ / Что это такое?"):
        st.markdown("""
        - **Монте-Карло симуляция**: Метод моделирования сценариев на основе вероятностей.
        - **Win Rate**: % выигрышных сделок.
        - **Risk/Reward**: Соотношение риска к прибыли (RR).
        - **Вариативность**: Добавляет случайность к RR для реализма.
        - **Риск разорения**: Вероятность падения ниже порога.
        """)

    mode = st.radio(T['mode'], ["%", "$"])
    start_balance = st.number_input(T['start_bal'], value=10000, step=1000, help="Ваш начальный капитал.")
    
    col_win, col_be = st.columns(2)
    win_rate = col_win.number_input(T['win_rate'], value=55, help="Процент выигрышных сделок (0-100).")
    be_rate = col_be.number_input(T['be_rate'], value=5, help="Процент сделок, закрытых в 0.")
    
    col_r, col_p = st.columns(2)
    risk_val = col_r.number_input(f"{T['risk']} ({mode})", value=1 if "%" in mode else 100, help="Риск на одну сделку.")
    reward_val = col_p.number_input(f"{T['reward']} ({mode})", value=2 if "%" in mode else 200, help="Целевая прибыль на одну сделку.")
    
    num_sims = st.number_input(T['num_sims'], value=50, step=1, min_value=1, max_value=10000, help="Количество прогонов симуляции.")
    trades_per_month = st.slider(T['trades_month'], 1, 50, 20, help="Сколько сделок вы делаете в месяц.")
    num_months = st.number_input(T['months'], value=24, step=1, help="Длительность периода в месяцах.")
    variability = st.slider(T['variability'], 0, 100, 20, help="Насколько сильно RR может отклоняться от цели.")
    ruin_threshold = st.number_input(T['ruin_threshold'], value=int(start_balance * 0.1), help="Баланс, при котором трейдинг считается оконченным.")

    st.subheader("Advanced Settings")
    commission_fixed = st.number_input("Комиссия фикс. ($)", value=0.0, help="Фиксированная плата за каждую сделку.")
    commission_pct = st.number_input("Комиссия %", value=0.0, help="Процент комиссии от объема сделки.")
    slippage = st.number_input("Слиппедж ($ или %)", value=0.0, help="Среднее проскальзывание.")
    
    dist_type = st.selectbox("Распределение исходов", ["Uniform", "Normal", "LogNormal", "Bootstrap"])
    uploaded_file = st.file_uploader("Загрузить историю сделок (для Bootstrap)", type="csv")
    sizing_type = st.selectbox("Position Sizing", ["Fixed Risk", "Kelly", "Volatility"])

    # ВАЛИДАЦИЯ
    if win_rate < 0 or win_rate > 100:
        st.error("Win Rate должен быть между 0 и 100%.")
        st.stop()
    if be_rate < 0 or be_rate > 100 - win_rate:
        st.error("Break Even Rate должен быть между 0 и (100% - Win Rate).")
        st.stop()
    if risk_val <= 0 or reward_val <= 0:
        st.error("Risk и Reward должны быть положительными.")
        st.stop()
    if num_sims < 1 or trades_per_month < 1 or num_months < 1:
        st.error("Количественные параметры должны быть минимум 1.")
        st.stop()
    if ruin_threshold < 0 or ruin_threshold > start_balance:
        st.error("Порог разорения должен быть между 0 и начальным балансом.")
        st.stop()

st.title(f"{T['title']} by timuroff")

# --- ЛОГИКА СИМУЛЯЦИИ ---
def run_simulation(n_sims, w_rate, silent=False):
    all_runs = []
    total_trades = int(num_months * trades_per_month)
    ruined_count = 0
    progress_bar = st.progress(0) if not silent else None

    # Подготовка данных для Bootstrap
    outcomes, amounts = None, None
    if dist_type == "Bootstrap" and uploaded_file:
        df_boot = pd.read_csv(uploaded_file)
        if 'outcome' in df_boot.columns and 'amount' in df_boot.columns:
            outcomes = df_boot['outcome'].values
            amounts = df_boot['amount'].values

    for i in range(int(n_sims)):
        if not silent: progress_bar.progress((i + 1) / n_sims)
        
        balances = np.full(total_trades + 1, float(start_balance))
        trade_results = []
        trade_amounts = np.zeros(total_trades)
        
        # Распределение
        if dist_type == "Normal":
            rns = np.random.normal(w_rate, 10, total_trades)
        elif dist_type == "LogNormal":
            rns = np.random.lognormal(np.log(max(1, w_rate)/100), 0.2, total_trades) * 100
        else:
            rns = np.random.random(total_trades) * 100
            
        v_factors = np.random.normal(1, variability / 100, total_trades)
        is_ruined = False

        # Позиционный сайзинг (Kelly)
        kelly_f = (w_rate/100 * (reward_val/risk_val) - (1 - w_rate/100)) / (reward_val/risk_val) if sizing_type == "Kelly" else 0

        # Цикл симуляции (Оптимизированный)
        for t in range(total_trades):
            curr_bal = balances[t]
            if curr_bal <= ruin_threshold: 
                is_ruined = True
                balances[t+1:] = curr_bal
                break
            
            # Определение исхода
            if dist_type == "Bootstrap" and outcomes is not None:
                idx = np.random.choice(len(outcomes))
                change = amounts[idx]
                res = outcomes[idx]
            else:
                rn = rns[t]
                if rn < w_rate:
                    res = 1
                    # Динамический риск
                    current_risk = (curr_bal * kelly_f) if sizing_type == "Kelly" else risk_val
                    current_reward = (current_risk * (reward_val/risk_val))
                    change = (curr_bal * (current_reward * v_factors[t] / 100)) if "%" in mode else (current_reward * v_factors[t])
                elif rn < (w_rate + be_rate):
                    res = 0
                    change = 0
                else:
                    res = -1
                    current_risk = (curr_bal * kelly_f) if sizing_type == "Kelly" else risk_val
                    change = -(curr_bal * (current_risk * v_factors[t] / 100)) if "%" in mode else -(current_risk * v_factors[t])

            # Комиссии и слиппедж
            if res != 0:
                change -= commission_fixed
                change -= abs(change) * (commission_pct / 100)
                slip = np.random.normal(slippage, slippage/2 + 1e-9)
                change -= slip if change > 0 else +slip

            trade_amounts[t] = change
            trade_results.append(res)
            balances[t+1] = max(0.0, curr_bal + change)

        if is_ruined: ruined_count += 1
        
        history = balances.tolist()
        monthly_diffs = [balances[m*trades_per_month] - balances[(m-1)*trades_per_month] if m>0 else balances[m*trades_per_month]-start_balance for m in range(1, num_months+1)]
        
        pos_trades = trade_amounts[trade_amounts > 0]
        neg_trades = trade_amounts[trade_amounts < 0]
        m_returns = np.array(monthly_diffs) / start_balance
        
        # Расширенные метрики
        data = {
            "history": history, "final": balances[-1], "mdd": calculate_single_mdd(history),
            "max_wins": get_consecutive(trade_results)[0], "max_losses": get_consecutive(trade_results)[1],
            "win_pct": (trade_results.count(1)/len(trade_results))*100 if trade_results else 0, 
            "monthly_diffs": monthly_diffs,
            "profit_factor": np.sum(pos_trades) / abs(np.sum(neg_trades)) if np.sum(neg_trades) != 0 else 10.0,
            "expectancy": np.mean(trade_amounts),
            "sharpe": (np.mean(m_returns) / (np.std(m_returns) + 1e-9) * np.sqrt(12)),
            "sortino": (np.mean(m_returns) / (np.std(m_returns[m_returns<0]) + 1e-9) * np.sqrt(12)) if len(m_returns[m_returns<0]) > 0 else 10.0,
            "avg_win": np.mean(pos_trades) if len(pos_trades) > 0 else 0,
            "avg_loss": np.mean(neg_trades) if len(neg_trades) > 0 else 0,
            "recovery_factor": (balances[-1] - start_balance) / ((calculate_single_mdd(history)/100 * start_balance) + 1e-9),
            "calmar": ((balances[-1]/start_balance - 1) * 100) / (calculate_single_mdd(history) + 1e-9),
            "cagr": ((balances[-1]/start_balance)**(1/(num_months/12)) - 1) * 100 if balances[-1] > 0 else -100,
            "var_95": np.percentile(m_returns, 5) * start_balance
        }
        all_runs.append(data)
            
    return all_runs, (ruined_count / n_sims) * 100

results, r_ruin_pct = run_simulation(num_sims, win_rate)
finals = [r["final"] for r in results]
idx_best, idx_worst = int(np.argmax(finals)), int(np.argmin(finals))
idx_median = int((np.abs(np.array(finals) - np.median(finals))).argmin())

# --- ГРАФИК EQUITY ---
fig = go.Figure()
for i, r in enumerate(results[:100]): # Лимит 100 линий для скорости
    if i not in [idx_best, idx_worst, idx_median]:
        fig.add_trace(go.Scatter(y=r["history"], mode='lines', line=dict(width=1, color="gray"), opacity=0.1, showlegend=False))

fig.add_trace(go.Scatter(y=results[idx_median]["history"], name="MEDIAN", line=dict(color="#3B82F6", width=2)))
fig.add_trace(go.Scatter(y=results[idx_worst]["history"], name="WORST", line=dict(color="#EF4444", width=2)))
fig.add_trace(go.Scatter(y=results[idx_best]["history"], name="BEST", line=dict(color="#10B981", width=2)))
fig.update_layout(template=plotly_template, height=450, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# --- МЕТРИКИ ВЕРХНЕГО УРОВНЯ ---
c_ruin, c_p5, c_p95 = st.columns(3)
c_ruin.metric(T['risk_of_ruin'], f"{r_ruin_pct:.1f}%")
c_p5.metric("5% Percentile (Worst)", f"${np.percentile(finals, 5):,.0f}")
c_p95.metric("95% Percentile (Best)", f"${np.percentile(finals, 95):,.0f}")

# --- ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ ---
col_graph1, col_graph2 = st.columns(2)
with col_graph1:
    fig_hist = go.Figure(go.Histogram(x=finals, nbinsx=30, marker_color='#3B82F6'))
    fig_hist.update_layout(title=T['hist_title'], template=plotly_template, height=300)
    st.plotly_chart(fig_hist, use_container_width=True)

with col_graph2:
    mdds = [r['mdd'] for r in results]
    fig_box = go.Figure(go.Box(y=mdds, name="Max Drawdown %", marker_color='#EF4444'))
    fig_box.update_layout(title="MDD Distribution", template=plotly_template, height=300)
    st.plotly_chart(fig_box, use_container_width=True)

# CDF Chart
sorted_finals = np.sort(finals)
cdf = np.arange(1, len(sorted_finals)+1) / len(sorted_finals)
fig_cdf = go.Figure(go.Scatter(x=sorted_finals, y=cdf, fill='tozeroy', line=dict(color='#10B981')))
fig_cdf.update_layout(title="CDF Final Balances (Probability of achieving X balance)", template=plotly_template, height=300)
st.plotly_chart(fig_cdf, use_container_width=True)

# --- АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ ---
st.subheader(T['sensitivity'])
sens_wr = range(40, 71, 5)
sens_rr = [1.5, 2.0, 2.5, 3.0]
sens_z = []
for wr in sens_wr:
    row = []
    for rr in sens_rr:
        # Упрощенная быстрая симуляция для хитмапа
        s_res, _ = run_simulation(10, wr, silent=True)
        row.append(np.mean([r['final'] for r in s_res]))
    sens_z.append(row)

fig_heat = go.Figure(go.Heatmap(z=sens_z, x=[f"RR {r}" for r in sens_rr], y=[f"WR {w}%" for w in sens_wr], colorscale='Viridis'))
fig_heat.update_layout(title="Sensitivity: Win Rate vs RR vs Avg Final Balance", template=plotly_template, height=400)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# --- ТАБЛИЦЫ И МЕТРИКИ ---
def style_table(df):
    def apply_styles(row):
        styles = [''] * len(row)
        is_total = row['Month'] == T['year_total']
        try:
            val_str = str(row['Results $']).replace('$', '').replace(',', '').replace(' ', '')
            val = float(val_str)
        except: val = 0
        if is_total:
            bg_color = 'background-color: rgba(16, 185, 129, 0.15)' if val >= 0 else 'background-color: rgba(239, 68, 68, 0.15)'
            styles = [bg_color] * len(row)
        if val >= 0:
            styles[1] += '; color: #10B981'
            styles[2] += '; color: #10B981'
        else:
            styles[1] += '; color: #EF4444'
            styles[2] += '; color: #EF4444'
        return styles
    return df.style.apply(apply_styles, axis=1)

def render_scenario(data):
    with st.container(border=True):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Return %", f"{((data['final']-start_balance)/start_balance)*100:.1f}%")
        m2.metric("Profit Factor", f"{data['profit_factor']:.2f}")
        m3.metric("Sharpe", f"{data['sharpe']:.2f}")
        m4.metric("Calmar", f"{data['calmar']:.2f}")
        m5.metric("CAGR", f"{data['cagr']:.1f}%")
        
        m6, m7, m8, m9, m10 = st.columns(5)
        m6.metric("Expectancy", f"${data['expectancy']:.1f}")
        m7.metric("Recovery Factor", f"{data['recovery_factor']:.2f}")
        m8.metric("VaR 95% (Monthly)", f"${abs(data['var_95']):,.0f}")
        m9.markdown(f"<div style='line-height:1;'><p style='color:gray;font-size:14px;margin-bottom:5px;'>Avg Win/Loss</p><p style='font-size:24px;font-weight:bold;margin:0;'>${data['avg_win']:.0f} / ${abs(data['avg_loss']):.0f}</p></div>", unsafe_allow_html=True)
        m10.metric("Max DD", f"-{data['mdd']:.1f}%")

    diffs = data['monthly_diffs']
    num_years = int(np.ceil(len(diffs) / 12))
    for y in range(num_years):
        year_data = diffs[y*12 : (y+1)*12]
        rows = []
        year_sum_pct = 0
        for i, val in enumerate(year_data):
            pct = (val / start_balance) * 100
            year_sum_pct += pct
            rows.append({"Month": T['months_list'][i] if i < len(T['months_list']) else f"M{i+1}", "Results %": f"{pct:.1f}%", "Results $": f"${val:,.0f}"})
        df = pd.DataFrame(rows)
        total_row = pd.DataFrame([{"Month": T['year_total'], "Results %": f"{year_sum_pct:.1f}%", "Results $": f"${sum(year_data):,.0f}"}], index=[" "])
        df = pd.concat([df, total_row])
        st.write(f"**Year {2026 + y}**")
        st.table(style_table(df))

tab_med, tab_worst, tab_best = st.tabs(["MOST POSSIBLE", "WORST CASE", "BEST CASE"])
with tab_med: render_scenario(results[idx_median])
with tab_worst: render_scenario(results[idx_worst])
with tab_best: render_scenario(results[idx_best])
