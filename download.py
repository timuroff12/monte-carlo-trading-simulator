import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Professional Monte Carlo", layout="wide")

# --- CSS: МАКСИМАЛЬНАЯ ЧИСТОТА И КОМПАКТНОСТЬ ---
st.markdown("""
    <style>
    [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-list"] {
        display: flex; justify-content: center; gap: 8px;
        padding-bottom: 10px; border: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px; width: 180px; border-radius: 4px;
        font-weight: bold; font-size: 13px; color: white !important;
        border: none !important;
    }
    /* Цвета кнопок как на референсе */
    div[data-baseweb="tab-list"] button:nth-child(1) { background-color: #3B82F6 !important; }
    div[data-baseweb="tab-list"] button:nth-child(2) { background-color: #EF4444 !important; }
    div[data-baseweb="tab-list"] button:nth-child(3) { background-color: #10B981 !important; }
    
    /* Компактные метрики */
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem !important; opacity: 0.8; }
    
    /* Убираем лишние отступы */
    .stTable td, .stTable th { padding: 3px !important; font-size: 12px !important; }
    </style>
""", unsafe_allow_html=True)

# --- ФУНКЦИИ ---
def calculate_single_mdd(history):
    h = np.array(history)
    peaks = np.maximum.accumulate(h)
    drawdowns = (peaks - h) / (peaks + 1e-9)
    return float(np.max(drawdowns) * 100)

def get_consecutive(results):
    max_w, max_l = 0, 0
    cw, cl = 0, 0
    for r in results:
        if r == 1: cw += 1; cl = 0
        elif r == -1: cl += 1; cw = 0
        else: cw, cl = 0, 0
        max_w = max(max_w, cw); max_l = max(max_l, cl)
    return max_w, max_l

# --- SIDEBAR (УЛЬТРА-КОМПАКТ) ---
with st.sidebar:
    st.caption("PARAM SETTINGS")
    mode = st.segmented_control("Mode", ["%", "$"], default="%")
    start_balance = st.number_input("Balance", value=10000)
    
    col1, col2 = st.columns(2)
    win_rate = col1.number_input("Win%", value=55)
    be_rate = col2.number_input("BE%", value=5)
    
    col3, col4 = st.columns(2)
    risk = col3.number_input("Risk", value=1.0 if mode=="%" else 100.0)
    reward = col4.number_input("Reward", value=2.0 if mode=="%" else 200.0)
    
    num_sims = st.select_slider("Simulations", options=[10, 20, 50, 100, 200], value=50)
    
    col5, col6 = st.columns(2)
    t_pm = col5.number_input("Tr/Mo", value=20)
    months = col6.number_input("Months", value=24)
    var = st.slider("Variability %", 0, 100, 20)

# --- ЛОГИКА ---
def run_simulation():
    runs = []
    total_t = int(months * t_pm)
    for _ in range(num_sims):
        bal = float(start_balance)
        hist, res, diffs = [bal], [], []
        m_start = bal
        m_diffs = []
        for t in range(1, total_t + 1):
            if bal <= 0: 
                bal = 0.0; hist.append(0.0); res.append(-1); diffs.append(0); continue
            rn = np.random.random() * 100
            vf = np.random.normal(1, var / 100)
            if rn < win_rate:
                ch = (bal * (reward * vf / 100)) if mode=="%" else (reward * vf)
                bal += max(0.0, float(ch)); res.append(1); diffs.append(ch)
            elif rn < (win_rate + be_rate):
                res.append(0); diffs.append(0)
            else:
                ch = (bal * (risk * vf / 100)) if mode=="%" else (risk * vf)
                bal -= max(0.0, float(ch)); res.append(-1); diffs.append(-ch)
            hist.append(bal)
            if t % t_pm == 0:
                m_diffs.append(bal - m_start); m_start = bal
        
        mdd = calculate_single_mdd(hist)
        w_vals = [v for v in diffs if v > 0]
        l_vals = [abs(v) for v in diffs if v < 0]
        mw, ml = get_consecutive(res)
        runs.append({
            "history": hist, "final": bal, "mdd": mdd, "max_wins": mw, "max_losses": ml,
            "pf": sum(w_vals)/sum(l_vals) if sum(l_vals)>0 else 0,
            "rf": (bal - start_balance)/(start_balance * (mdd/100)) if mdd > 0 else 0,
            "exp": sum(diffs)/len(res) if res else 0,
            "wp": (res.count(1)/len(res))*100, "m_diffs": m_diffs
        })
    return runs

results = run_simulation()
finals = [r["final"] for r in results]
idx_b, idx_w = int(np.argmax(finals)), int(np.argmin(finals))
idx_m = int((np.abs(np.array(finals) - np.median(finals))).argmin())

# --- ГРАФИК: ОБЛАКО СИМУЛЯЦИЙ ---
fig = go.Figure()
# Рисуем все линии серым цветом
for r in results:
    fig.add_trace(go.Scatter(y=r["history"], mode='lines', 
                             line=dict(color='rgba(128, 128, 128, 0.15)', width=1), 
                             showlegend=False, hoverinfo='skip'))

# Накладываем основные сценарии
fig.add_trace(go.Scatter(y=results[idx_m]["history"], name="MOST POSSIBLE", line=dict(color="#3B82F6", width=2.5)))
fig.add_trace(go.Scatter(y=results[idx_w]["history"], name="WORST CASE", line=dict(color="#EF4444", width=2.5)))
fig.add_trace(go.Scatter(y=results[idx_b]["history"], name="BEST CASE", line=dict(color="#10B981", width=2.5)))

fig.update_layout(template="plotly_dark", height=400, margin=dict(l=10, r=10, t=10, b=10), 
                  legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"))
st.plotly_chart(fig, use_container_width=True)

# --- АНАЛИЗ ---
st.markdown("<h3 style='text-align: center;'>Детальный анализ сценариев</h3>", unsafe_allow_html=True)
t_m, t_w, t_b = st.tabs(["MOST POSSIBLE", "WORST", "BEST"])

def show_data(d):
    # Чистый блок без заливки
    with st.container(border=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final Balance", f"${d['final']:,.0f}")
        c2.metric("Return", f"{((d['final']-start_balance)/start_balance)*100:+.1f}%")
        c3.metric("Max Drawdown", f"-{d['mdd']:.1f}%")
        c4.metric("Profit Factor", f"{d['pf']:.2f}")
        c5.metric("Win Rate", f"{d['wp']:.1f}%")
        
        c6, c7, c8, c9, c10 = st.columns(5)
        c6.metric("Recovery Factor", f"{d['rf']:.2f}")
        c7.metric("Expectancy", f"${d['exp']:,.1f}" if mode=="$" else f"{d['exp']:,.1f}%")
        c8.metric("Cons. Wins", d['max_wins'])
        c9.metric("Cons. Losses", d['max_losses'])
        c10.metric("Initial", f"${start_balance:,.0f}")

    # Таблицы месяцев
    ms = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    cols = st.columns(3)
    for y in range(int(np.ceil(len(d['m_diffs'])/12))):
        with cols[y % 3]:
            y_d = d['m_diffs'][y*12 : (y+1)*12]
            df = pd.DataFrame([{"Month": ms[i], "Res %": f"{(v/start_balance)*100:+.1f}%", "Res $": f"${v:+,.0f}"} for i, v in enumerate(y_d)])
            st.caption(f"Year {2026+y}")
            st.table(df)

with t_m: show_data(results[idx_m])
with t_w: show_data(results[idx_w])
with t_b: show_data(results[idx_b])
