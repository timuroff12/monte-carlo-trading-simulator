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

# --- CSS И СТИЛИЗАЦИЯ (Исправлено) ---
plotly_template = "plotly_dark"
if theme == "Light":
    plotly_template = "plotly_white"
    st.markdown("""
        <style>
        .stApp { background-color: white; color: black; }
        section[data-testid="stSidebar"] { background-color: #f0f2f6; }
        </style>
    """, unsafe_allow_html=True)

# Основной CSS без f-строки, чтобы скобки {} не вызывали NameError
st.markdown("""
    <style>
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
    if be_rate < 0 or be_rate > 100 - win_rate:
        st.error("Break Even Rate должен быть между 0 и (100% - Win Rate).")
        st.stop()
    if risk_val <= 0 or reward_val <= 0:
        st.error("Риск и прибыль должны быть положительными.")
        st.stop()
    if ruin_threshold < 0 or ruin_threshold > start_balance:
        st.error("Порог разорения некорректен.")
        st.stop()

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

# --- ЛОГИКА СИМУЛЯЦИИ ---
def run_simulation(n_sims, w_rate, silent=False):
    all_runs = []
    total_trades = int(num_months * trades_per_month)
    ruined_count = 0
    progress_bar = st.progress(0) if not silent else None

    # Предзагрузка данных для Bootstrap
    boot_outcomes, boot_amounts = None, None
    if dist_type == "Bootstrap" and uploaded_file:
        df_boot = pd.read_csv(uploaded_file)
        if 'outcome' in df_boot.columns: boot_outcomes = df_boot['outcome'].values
        if 'amount' in df_boot.columns: boot_amounts = df_boot['amount'].values

    for i in range(int(n_sims)):
        if not silent: progress_bar.progress((i + 1) / n_sims)
        
        balance = float(start_balance)
        balances = [balance]
        trade_results = []
        trade_amounts = []
        is_ruined = False
        
        # Формула Келли
        kelly_f = (w_rate/100 * (reward_val/risk_val) - (1 - w_rate/100)) / (reward_val/risk_val) if sizing_type == "Kelly" else 0

        for t in range(total_trades):
            if balance <= ruin_threshold:
                is_ruined = True
                balance = max(0.0, balance)
            
            if balance <= 0:
                balances.append(0.0)
                continue

            # Генерация исхода
            if dist_type == "Bootstrap" and boot_outcomes is not None:
                idx = np.random.choice(len(boot_outcomes))
                res = boot_outcomes[idx]
                change = boot_amounts[idx] if boot_amounts is not None else 0
            else:
                rn = np.random.random() * 100
                v_f = np.random.normal(1, variability / 100)
                
                curr_risk = (balance * kelly_f) if sizing_type == "Kelly" else risk_val
                curr_reward = curr_risk * (reward_val / risk_val)

                if rn < w_rate:
                    res = 1
                    change = (balance * curr_reward / 100) if "%" in mode else (curr_reward * v_f)
                elif rn < (w_rate + be_rate):
                    res = 0
                    change = 0
                else:
                    res = -1
                    change = -(balance * curr_risk / 100) if "%" in mode else -(curr_risk * v_f)

            # Учет издержек
            if res != 0:
                change -= commission_fixed
                change -= abs(change) * (commission_pct / 100)
                slip = np.random.normal(slippage, abs(slippage)/2 + 0.001)
                change -= slip if change > 0 else -slip

            balance += change
            balances.append(max(0.0, balance))
            trade_results.append(res)
            trade_amounts.append(change)

        if is_ruined: ruined_count += 1
        
        pos_trades = [a for a in trade_amounts if a > 0]
        neg_trades = [a for a in trade_amounts if a < 0]
        m_diffs = [balances[m*trades_per_month] - balances[(m-1)*trades_per_month] for m in range(1, num_months+1)]
        m_returns = np.array(m_diffs) / start_balance

        all_runs.append({
            "history": balances, "final": balance, "mdd": calculate_single_mdd(balances),
            "max_wins": get_consecutive(trade_results)[0], "max_losses": get_consecutive(trade_results)[1],
            "win_pct": (trade_results.count(1)/len(trade_results))*100 if trade_results else 0,
            "monthly_diffs": m_diffs,
            "profit_factor": sum(pos_trades) / abs(sum(neg_trades)) if neg_trades else 10.0,
            "expectancy": np.mean(trade_amounts) if trade_amounts else 0,
            "sharpe": (np.mean(m_returns) / (np.std(m_returns) + 1e-9) * np.sqrt(12)),
            "avg_win": np.mean(pos_trades) if pos_trades else 0,
            "avg_loss": np.mean(neg_trades) if neg_trades else 0,
            "recovery_factor": (balance - start_balance) / ((calculate_single_mdd(balances)/100 * start_balance) + 1e-9),
            "cagr": ((balance/start_balance)**(1/(num_months/12)) - 1) * 100 if balance > 0 else -100
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
fig.update_layout(template=plotly_template, height=450, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig, use_container_width=True)

# Метрики
c_ruin, c_p5, c_p95 = st.columns(3)
c_ruin.metric(T['risk_of_ruin'], f"{r_ruin_pct:.1f}%")
c_p5.metric("5% Percentile (Worst)", f"${np.percentile(finals, 5):,.0f}")
c_p95.metric("95% Percentile (Best)", f"${np.percentile(finals, 95):,.0f}")

# Распределение и MDD
col_h, col_b = st.columns(2)
with col_h:
    fig_hist = go.Figure(go.Histogram(x=finals, nbinsx=30, marker_color='#3B82F6'))
    fig_hist.update_layout(title=T['hist_title'], template=plotly_template, height=300)
    st.plotly_chart(fig_hist, use_container_width=True)
with col_b:
    mdds = [r['mdd'] for r in results]
    fig_box = go.Figure(go.Box(y=mdds, name="Max Drawdown %", marker_color='#EF4444'))
    fig_box.update_layout(title="MDD Distribution", template=plotly_template, height=300)
    st.plotly_chart(fig_box, use_container_width=True)

# Анализ чувствительности (Быстрый прогон)
sens_x = list(range(40, 71, 5))
sens_y = []
for wr in sens_x:
    s_res, _ = run_simulation(10, wr, silent=True)
    sens_y.append(np.mean([r['final'] for r in s_res]))
fig_sens = go.Figure(go.Scatter(x=sens_x, y=sens_y, mode='lines+markers', line=dict(color='#10B981')))
fig_sens.update_layout(title=f"{T['sensitivity']} (Win Rate vs Avg Final)", template=plotly_template, height=300)
st.plotly_chart(fig_sens, use_container_width=True)

st.divider()

# --- СЦЕНАРИИ ---
def style_table(df):
    def apply_styles(row):
        styles = [''] * len(row)
        is_total = row['Month'] == T['year_total']
        try:
            val = float(str(row['Results $']).replace('$', '').replace(',', ''))
        except: val = 0
        if is_total:
            styles = [f'background-color: rgba({"16, 185, 129" if val >= 0 else "239, 68, 68"}, 0.15)'] * len(row)
        color = '#10B981' if val >= 0 else '#EF4444'
        styles[1] += f'; color: {color}'; styles[2] += f'; color: {color}'
        return styles
    return df.style.apply(apply_styles, axis=1)

def render_scenario(data):
    with st.container(border=True):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Return %", f"{((data['final']-start_balance)/start_balance)*100:.1f}%")
        m2.metric("Profit Factor", f"{data['profit_factor']:.2f}")
        m3.metric("Sharpe", f"{data['sharpe']:.2f}")
        m4.metric("CAGR", f"{data['cagr']:.1f}%")
        
        m5, m6, m7 = st.columns(3)
        m5.metric("Expectancy", f"${data['expectancy']:.1f}")
        m6.metric("Recovery Factor", f"{data['recovery_factor']:.2f}")
        m7.metric("Max DD", f"-{data['mdd']:.1f}%")

    diffs = data['monthly_diffs']
    num_years = int(np.ceil(len(diffs) / 12))
    for y in range(num_years):
        year_data = diffs[y*12 : (y+1)*12]
        rows = []
        for i, val in enumerate(year_data):
            pct = (val / start_balance) * 100
            rows.append({"Month": T['months_list'][i], "Results %": f"{pct:.1f}%", "Results $": f"${val:,.0f}"})
        df = pd.DataFrame(rows)
        total_row = pd.DataFrame([{"Month": T['year_total'], "Results %": f"{(sum(year_data)/start_balance)*100:.1f}%", "Results $": f"${sum(year_data):,.0f}"}], index=[" "])
        st.write(f"**Year {2026 + y}**")
        st.table(style_table(pd.concat([df, total_row])))

tab_med, tab_worst, tab_best = st.tabs(["MOST POSSIBLE", "WORST CASE", "BEST CASE"])
with tab_med: render_scenario(results[idx_median])
with tab_worst: render_scenario(results[idx_worst])
with tab_best: render_scenario(results[idx_best])
