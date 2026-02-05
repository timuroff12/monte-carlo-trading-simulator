import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Professional Monte Carlo Sim", layout="wide")

# Убираем смайл, оставляем чистый заголовок
st.title("Симуляция Монте-Карло для трейдеров")

# Вспомогательные функции
def calculate_single_mdd(history):
    if not history or len(history) < 2: return 0.0
    h = np.array(history)
    peaks = np.maximum.accumulate(h)
    drawdowns = (peaks - h) / (peaks + 1e-9)
    return float(np.max(drawdowns) * 100)

def get_consecutive(results):
    # results: список 1 (win), 0 (be), -1 (loss)
    max_wins = 0
    max_losses = 0
    cur_wins = 0
    cur_losses = 0
    for r in results:
        if r == 1:
            cur_wins += 1
            cur_losses = 0
        elif r == -1:
            cur_losses += 1
            cur_wins = 0
        else:
            cur_wins = 0
            cur_losses = 0
        max_wins = max(max_wins, cur_wins)
        max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses

# --- SIDEBAR ---
with st.sidebar:
    st.header("Настройки стратегии")
    mode = st.radio("Режим расчета:", ["Проценты (%)", "Доллары ($)"])
    start_balance = st.number_input("Начальный баланс", value=10000)
    
    # Ввод как на скрине image_cf0549.png
    col_win, col_be = st.columns(2)
    win_rate = col_win.number_input("Winning trades %", value=55, min_value=0, max_value=100)
    be_rate = col_be.number_input("Break even trades %", value=5, min_value=0, max_value=100)
    
    loss_rate = 100 - win_rate - be_rate
    if loss_rate < 0:
        st.error("Сумма % побед и безубытка больше 100!")
        loss_rate = 0
    st.caption(f"Losing trades: {loss_rate}%")

    col_r, col_p = st.columns(2)
    risk_val = col_r.number_input(f"Риск ({mode[-2]})", value=1.0 if "%" in mode else 100.0)
    reward_val = col_p.number_input(f"Прибыль ({mode[-2]})", value=2.0 if "%" in mode else 200.0)
    
    st.divider()
    num_sims = st.number_input("Количество симуляций", value=50)
    trades_per_month = st.slider("Сделок в месяц", 1, 50, 20)
    num_months = st.number_input("Срок (месяцев)", value=24)
    variability = st.slider("Вариативность RR (%)", 0, 100, 20)

# --- ЛОГИКА ---
def run_simulation():
    all_runs = []
    all_stats = []
    total_trades = int(num_months * trades_per_month)

    for _ in range(int(num_sims)):
        balance = float(start_balance)
        history = [balance]
        trade_results = [] # 1, 0, -1
        monthly_diffs = []
        
        current_month_start_bal = balance
        
        for t in range(1, total_trades + 1):
            if balance <= 0:
                balance = 0.0
                history.append(balance)
                trade_results.append(-1)
                continue
            
            # Рандом исхода
            rn = np.random.random() * 100
            v_factor = np.random.normal(1, variability / 100)
            
            if rn < win_rate:
                change = (balance * (reward_val * v_factor / 100)) if "%" in mode else (reward_val * v_factor)
                balance += max(0.0, float(change))
                trade_results.append(1)
            elif rn < (win_rate + be_rate):
                # Безубыток (маленький шум около 0)
                balance += np.random.normal(0, 5) if "$" in mode else balance * np.random.normal(0, 0.001)
                trade_results.append(0)
            else:
                change = (balance * (risk_val * v_factor / 100)) if "%" in mode else (risk_val * v_factor)
                balance -= max(0.0, float(change))
                trade_results.append(-1)
            
            history.append(balance)
            
            # Сбор данных для таблицы в конце "месяца"
            if t % trades_per_month == 0:
                monthly_diffs.append(balance - current_month_start_bal)
                current_month_start_bal = balance

        max_w, max_l = get_consecutive(trade_results)
        all_runs.append({
            "history": history,
            "final": balance,
            "mdd": calculate_single_mdd(history),
            "max_wins": max_w,
            "max_losses": max_l,
            "win_pct": (trade_results.count(1) / len(trade_results)) * 100,
            "monthly_diffs": monthly_diffs
        })
    return all_runs

results = run_simulation()
finals = [r["final"] for r in results]

# Сортировка для выбора сценариев
idx_best = int(np.argmax(finals))
idx_worst = int(np.argmin(finals))
median_val = np.median(finals)
idx_median = int((np.abs(np.array(finals) - median_val)).argmin())

# Интерфейс выбора сценария (кнопки как на скрине image_c25da4.png)
st.write("### Выберите сценарий для детального анализа")
tab_med, tab_worst, tab_best = st.tabs(["MOST POSSIBLE", "WORST", "BEST"])

def render_scenario(data):
    # Верхняя панель статистики
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Initial balance", f"${start_balance:,.0f}")
    c2.metric("Result balance", f"${data['final']:,.0f}")
    c3.metric("Return %", f"{((data['final']-start_balance)/start_balance)*100:.1f}%")
    c4.metric("Max drawdown", f"-{data['mdd']:.1f}%")
    c5.metric("Max cons. losses", data['max_losses'])
    c6.metric("Max cons. wins", data['max_wins'])
    c7.metric("Win trades %", f"{data['win_pct']:.1f}%")

    # Сетка таблиц по годам
    st.write("#### Результаты по месяцам")
    diffs = data['monthly_diffs']
    
    # Разбиваем на годы (по 12 месяцев)
    num_years = int(np.ceil(len(diffs) / 12))
    cols_years = st.columns(min(num_years, 3))
    
    months_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    for y in range(num_years):
        with cols_years[y % 3]:
            year_label = 2025 + y
            year_data = diffs[y*12 : (y+1)*12]
            
            rows = []
            for i, val in enumerate(year_data):
                pct = (val / start_balance) * 100 # Упрощенно от начального
                rows.append({"Month": months_names[i], "Results %": f"{pct:+.1f}%", "Results $": f"${val:,.0f}"})
            
            st.write(f"**Year {year_label}**")
            st.table(pd.DataFrame(rows))

with tab_med: render_scenario(results[idx_median])
with tab_worst: render_scenario(results[idx_worst])
with tab_best: render_scenario(results[idx_best])

st.divider()

# Общий график (паутина)
fig = go.Figure()
for r in results[:100]:
    fig.add_trace(go.Scatter(y=r["history"], mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
fig.update_layout(title="Общая динамика (все симуляции)", template="plotly_dark", height=400)
st.plotly_chart(fig, use_container_width=True)
