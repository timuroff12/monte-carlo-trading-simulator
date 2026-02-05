import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Monte Carlo Trader Sim Pro", layout="wide")

st.title("Симуляция Монте-Карло для трейдеров")

# Исправленная функция расчета просадки
def calculate_single_mdd(history):
    if not history or len(history) < 2:
        return 0.0
    h = np.array(history)
    # Исключаем деление на ноль, если баланс везде 0
    if np.all(h <= 0):
        return 100.0
    peaks = np.maximum.accumulate(h)
    # Добавляем небольшое смещение, чтобы не делить на ноль
    drawdowns = (peaks - h) / (peaks + 1e-9)
    return float(np.max(drawdowns) * 100)

# --- SIDEBAR ---
with st.sidebar:
    st.header("Параметры")
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

    for _ in range(int(num_sims)):
        balance = float(start_balance)
        history = [balance]
        peak = balance
        mdd = 0.0
        
        for _ in range(int(num_trades)):
            if balance <= 0:
                balance = 0.0
                history.append(balance)
                break
                
            is_win = np.random.random() < (win_rate / 100)
            v_factor = np.random.normal(1, variability / 100)
            
            if is_win:
                change = (balance * (reward_val * v_factor / 100)) if "%" in mode else (reward_val * v_factor)
                balance += max(0.0, float(change))
            else:
                change = (balance * (risk_val * v_factor / 100)) if "%" in mode else (risk_val * v_factor)
                balance -= max(0.0, float(change))
            
            history.append(balance)
            if balance > peak: peak = balance
            dd = (peak - balance) / peak if peak > 0 else 0
            if dd > mdd: mdd = dd

        all_runs.append(history)
        final_balances.append(balance)
        max_drawdowns.append(mdd * 100)
        
    return all_runs, final_balances, max_drawdowns

runs, finals, drawdowns = run_simulation()

# --- АНАЛИЗ СЦЕНАРИЕВ ---
finals_arr = np.array(finals)
idx_best = int(np.argmax(finals_arr))
idx_worst = int(np.argmin(finals_arr))
median_val = np.median(finals_arr)
idx_most_likely = int((np.abs(finals_arr - median_val)).argmin())

scenarios = [
    {"title": "MOST POSSIBLE", "color": "#3B82F6", "run": runs[idx_most_likely], "final": finals[idx_most_likely]},
    {"title": "WORST CASE", "color": "#EF4444", "run": runs[idx_worst], "final": finals[idx_worst]},
    {"title": "BEST CASE", "color": "#10B981", "run": runs[idx_best], "final": finals[idx_best]}
]

# --- ВИЗУАЛИЗАЦИЯ ---
col_left, col_right = st.columns([2, 1])

with col_left:
    fig_main = go.Figure()
    # Ограничиваем количество линий для отрисовки, чтобы не перегружать браузер
    display_count = min(len(runs), 100)
    for r in runs[:display_count]:
        fig_main.add_trace(go.Scatter(y=r, mode='lines', line=dict(width=1), opacity=0.3, showlegend=False))
    fig_main.update_layout(title="Динамика баланса", template="plotly_dark", height=400)
    st.plotly_chart(fig_main, use_container_width=True)

with col_right:
    st.subheader("Общая статистика")
    ruin_p = (finals.count(0) / len(finals)) * 100
    st.metric("Риск обнуления", f"{ruin_p:.1f}%")
    st.metric("Средняя просадка", f"{np.mean(drawdowns):.1f}%")
    st.metric("Медианный доход", f"${np.median(finals):,.0f}")

st.divider()

# --- БЛОК СЦЕНАРИЕВ ---
cols = st.columns(3)
for i, sc in enumerate(scenarios):
    with cols[i]:
        st.markdown(f"""
            <div style="background-color: {sc['color']}; padding: 15px; border-radius: 10px 10px 0 0; color: white; text-align: center; font-weight: bold;">
                {sc['title']}
            </div>
        """, unsafe_allow_html=True) # Исправлено: allow_html вместо allow_index
        
        with st.container(border=True):
            ret_pct = ((sc["final"] - start_balance) / start_balance) * 100
            mdd_sc = calculate_single_mdd(sc["run"])
            
            st.write(f"**Initial balance:** ${start_balance:,.0f}")
            st.write(f"**Result balance:** ${sc['final']:,.0f}")
            st.write(f"**Total Return:** {ret_pct:.1f}%")
            st.write(f"**Max Drawdown:** -{mdd_sc:.1f}%")

st.divider()

fig_hist = px.histogram(
    finals, nbins=30, title="Распределение финального баланса",
    labels={'value': 'Баланс', 'count': 'Частота'},
    color_discrete_sequence=['#00CC96'], template="plotly_dark"
)
st.plotly_chart(fig_hist, use_container_width=True)
