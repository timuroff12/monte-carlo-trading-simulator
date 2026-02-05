import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Monte Carlo Trader Sim Pro", layout="wide")

st.title("📊 Симуляция Монте-Карло для трейдеров")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Параметры")
    mode = st.radio("Режим расчета:", ["Проценты (%)", "Доллары ($)"])
    start_balance = st.number_input("Начальный баланс", value=10000)
    win_rate = st.slider("Win Rate (%)", 0, 100, 50)
    
    col1, col2 = st.columns(2)
    risk_val = col1.number_input(f"Риск ({mode[-2]})", value=1.0 if "%" in mode else 100.0)
    reward_val = col2.number_input(f"Прибыль ({mode[-2]})", value=2.0 if "%" in mode else 200.0)
    
    variability = st.slider("Вариативность RR (%)", 0, 100, 20)
    num_trades = st.number_input("Количество сделок", value=100)
    num_sims = st.number_input("Количество симуляций", value=100)

# --- ЛОГИКА ---
def run_simulation():
    all_runs = []
    final_balances = []
    max_drawdowns = []

    for _ in range(num_sims):
        balance = start_balance
        history = [balance]
        peak = balance
        mdd = 0
        
        for _ in range(num_trades):
            if balance <= 0:
                balance = 0
                history.append(balance)
                break
                
            is_win = np.random.random() < (win_rate / 100)
            v_factor = np.random.normal(1, variability / 100)
            
            if is_win:
                change = (balance * (reward_val * v_factor / 100)) if "%" in mode else (reward_val * v_factor)
                balance += max(0, change)
            else:
                change = (balance * (risk_val * v_factor / 100)) if "%" in mode else (risk_val * v_factor)
                balance -= max(0, change)
            
            history.append(balance)
            # Расчет просадки
            if balance > peak: peak = balance
            dd = (peak - balance) / peak if peak > 0 else 0
            if dd > mdd: mdd = dd

        all_runs.append(history)
        final_balances.append(balance)
        max_drawdowns.append(mdd * 100)
        
    return all_runs, final_balances, max_drawdowns

runs, finals, drawdowns = run_simulation()

# --- ВИЗУАЛИЗАЦИЯ ---
col_left, col_right = st.columns([2, 1])

with col_left:
    # График эквити
    fig_main = go.Figure()
    for r in runs[:100]: # Рисуем первые 100 для скорости
        fig_main.add_trace(go.Scatter(y=r, mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
    fig_main.update_layout(title="Динамика баланса", template="plotly_dark", height=450)
    st.plotly_chart(fig_main, use_container_width=True)

with col_right:
    # Метрики
    ruin_p = (finals.count(0) / num_sims) * 100
    st.metric("Риск обнуления", f"{ruin_p:.1f}%", delta_color="inverse")
    st.metric("Средняя просадка", f"{np.mean(drawdowns):.1f}%")
    st.metric("Медианный доход", f"${np.median(finals):,.0f}")

st.divider()

# Гистограмма финальных результатов
fig_hist = px.histogram(
    finals, 
    nbins=30, 
    title="Распределение финального баланса (Вероятности)",
    labels={'value': 'Финальный баланс', 'count': 'Количество исходов'},
    color_discrete_sequence=['#00CC96'],
    template="plotly_dark"
)
st.plotly_chart(fig_hist, use_container_width=True)
