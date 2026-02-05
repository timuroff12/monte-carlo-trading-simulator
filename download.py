import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Professional Monte Carlo Sim", layout="wide")

# Session state initialization
if 'lang' not in st.session_state:
    st.session_state.lang = 'ru'
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

lang = st.session_state.lang

# Translations
texts = {
    'ru': {
        'title': "Симуляция Монте-Карло для трейдеров by timuroff",
        'settings': "Настройки",
        'mode': "Режим:",
        'modes': ["Проценты (%)", "Доллары ($)"],
        'start_balance': "Начальный баланс",
        'win_trades': "Winning trades %",
        'be_trades': "Break even trades %",
        'risk': "Риск",
        'reward': "Прибыль",
        'num_sims': "Количество симуляций",
        'trades_per_month': "Сделок в месяц",
        'num_months': "Срок (месяцев)",
        'variability': "Вариативность RR (%)",
        'language': "Язык",
        'ru': "Русский",
        'en': "English",
        'theme': "Тема",
        'dark': "Тёмная",
        'light': "Светлая",
        'position_sizing': "Модель размера позиции",
        'position_options': ["Fixed", "Kelly"],
        'ruin_threshold': "Порог разорения %",
        'run_sim': "Запустить симуляцию",
        'error_win_be': "Сумма Win% + BE% не может превышать 100%",
        'error_positive': "Риск и прибыль должны быть положительными",
        'running': "Запуск симуляций...",
        'balance_dynamics': "Динамика баланса",
        'detailed_analysis': "Детальный анализ сценариев",
        'most_possible': "MOST POSSIBLE",
        'worst': "WORST",
        'best': "BEST",
        'initial_balance': "Initial balance",
        'result_balance': "Result balance",
        'return_pct': "Return %",
        'max_drawdown': "Max drawdown",
        'max_cons_loss': "Max cons. loss",
        'max_cons_win': "Max cons. win",
        'win_trades_pct': "Win trades %",
        'monthly_results': "Результаты по месяцам",
        'year': "Year",
        'month': "Month",
        'results_pct': "Results %",
        'results_dol': "Results $",
        'ror': "Риск разорения",
        'sensitivity_analysis': "Анализ чувствительности",
        'param_to_vary': "Параметр для варьирования",
        'min_val': "Минимальное значение",
        'max_val': "Максимальное значение",
        'steps': "Количество шагов",
        'run_sens': "Запустить анализ",
        'running_sens': "Выполнение анализа...",
        'sens_title': "Чувствительность {param} к среднему финальному балансу",
        'avg_final_balance': "Средний финальный баланс",
        'months': ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
    },
    'en': {
        'title': "Monte Carlo Simulation for Traders by timuroff",
        'settings': "Settings",
        'mode': "Mode:",
        'modes': ["Percent (%)", "Dollars ($)"],
        'start_balance': "Starting Balance",
        'win_trades': "Winning trades %",
        'be_trades': "Break even trades %",
        'risk': "Risk",
        'reward': "Profit",
        'num_sims': "Number of Simulations",
        'trades_per_month': "Trades per Month",
        'num_months': "Period (Months)",
        'variability': "RR Variability (%)",
        'language': "Language",
        'ru': "Russian",
        'en': "English",
        'theme': "Theme",
        'dark': "Dark",
        'light': "Light",
        'position_sizing': "Position Sizing Model",
        'position_options': ["Fixed", "Kelly"],
        'ruin_threshold': "Ruin Threshold %",
        'run_sim': "Run Simulation",
        'error_win_be': "Sum of Win% + BE% cannot exceed 100%",
        'error_positive': "Risk and reward must be positive",
        'running': "Running simulations...",
        'balance_dynamics': "Balance Dynamics",
        'detailed_analysis': "Detailed Scenario Analysis",
        'most_possible': "MOST POSSIBLE",
        'worst': "WORST",
        'best': "BEST",
        'initial_balance': "Initial balance",
        'result_balance': "Result balance",
        'return_pct': "Return %",
        'max_drawdown': "Max drawdown",
        'max_cons_loss': "Max cons. loss",
        'max_cons_win': "Max cons. win",
        'win_trades_pct': "Win trades %",
        'monthly_results': "Monthly Results",
        'year': "Year",
        'month': "Month",
        'results_pct': "Results %",
        'results_dol': "Results $",
        'ror': "Risk of Ruin",
        'sensitivity_analysis': "Sensitivity Analysis",
        'param_to_vary': "Parameter to Vary",
        'min_val': "Minimum Value",
        'max_val': "Maximum Value",
        'steps': "Number of Steps",
        'run_sens': "Run Analysis",
        'running_sens': "Running Analysis...",
        'sens_title': "Sensitivity of {param} to Average Final Balance",
        'avg_final_balance': "Average Final Balance",
        'months': ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    }
}

t = texts[lang]

# CSS with responsive design
st.markdown("""
    <style>
    /* Existing CSS */
    [data-baseweb="tab-highlight"] {
        display: none !important;
    }
   
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 12px;
        padding-bottom: 20px;
        border: none !important;
    }
   
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        width: 250px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 18px;
        color: white !important;
        border: none !important;
        transition: all 0.2s ease;
    }
    div[data-baseweb="tab-list"] button:nth-child(1) { background-color: #3B82F6 !important; } /* Blue */
    div[data-baseweb="tab-list"] button:nth-child(2) { background-color: #EF4444 !important; } /* Red */
    div[data-baseweb="tab-list"] button:nth-child(3) { background-color: #10B981 !important; } /* Green */
   
    .stTabs [aria-selected="true"] {
        filter: brightness(1.2);
        transform: scale(1.02);
        box-shadow: 0px 5px 15px rgba(0,0,0,0.3);
    }
   
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: none !important;
    }
    
    /* Responsive design */
    @media (max-width: 600px) {
        .stTabs [data-baseweb="tab-list"] {
            flex-direction: column;
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            width: 100%;
            height: 50px;
            font-size: 16px;
        }
        .stMetric {
            font-size: 14px;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.title(t['title'])

# Functions
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
        max_wins = max(max_wins, cur_wins); max_losses = max(max_losses, cur_losses)
    return max_wins, max_losses

# Sidebar
with st.sidebar:
    st.header(t['settings'])
    
    # Language select
    lang_options = [t['ru'], t['en']]
    lang_index = 0 if lang == 'ru' else 1
    lang_select = st.selectbox(t['language'], lang_options, index=lang_index, key='lang_select')
    new_lang = 'ru' if lang_select == t['ru'] else 'en'
    if new_lang != lang:
        st.session_state.lang = new_lang
        st.experimental_rerun()
    
    # Theme radio
    theme_options = [t['dark'], t['light']]
    theme_index = 0 if st.session_state.theme == 'dark' else 1
    theme_select = st.radio(t['theme'], theme_options, index=theme_index, key='theme_select')
    new_theme = 'dark' if theme_select == t['dark'] else 'light'
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.experimental_rerun()
    
    mode = st.radio(t['mode'], t['modes'])
    unit = '%' if '(%)' in mode else '$'
    
    start_balance = st.number_input(t['start_balance'], value=10000, step=1000, format="%d")
    
    col_win, col_be = st.columns(2)
    win_rate = col_win.number_input(t['win_trades'], value=55, format="%d")
    be_rate = col_be.number_input(t['be_trades'], value=5, format="%d")
    
    col_r, col_p = st.columns(2)
    default_risk = 1 if unit == '%' else 100
    default_reward = 2 if unit == '%' else 200
    risk_val = col_r.number_input(f"{t['risk']} ({unit})", value=default_risk, format="%d")
    reward_val = col_p.number_input(f"{t['reward']} ({unit})", value=default_reward, format="%d")
    
    num_sims = st.number_input(t['num_sims'], value=50, step=1, format="%d")
    trades_per_month = st.slider(t['trades_per_month'], 1, 50, 20)
    num_months = st.number_input(t['num_months'], value=24, step=1, format="%d")
    variability = st.slider(t['variability'], 0, 100, 20)
    
    position_sizing = st.selectbox(t['position_sizing'], t['position_options'])
    ruin_threshold = st.number_input(t['ruin_threshold'], value=0, min_value=0, max_value=100, step=1)

# Validation
valid = True
if win_rate + be_rate > 100:
    st.error(t['error_win_be'])
    valid = False
if risk_val <= 0 or reward_val <= 0:
    st.error(t['error_positive'])
    valid = False

# Simulation function
def run_simulation(start_balance, mode, win_rate, be_rate, risk_val, reward_val, num_sims, trades_per_month, num_months, variability, position_sizing, ruin_threshold, progress=True):
    all_runs = []
    total_trades = int(num_months * trades_per_month)
    unit = '%' if '(%)' in mode else '$'
    rr = reward_val / risk_val if risk_val > 0 else 0
    p = win_rate / 100.0
    be = be_rate / 100.0
    loss_p = 1 - p - be
    kelly = (p * rr - loss_p) / rr if rr > 0 else 0
    kelly = max(0, min(1, kelly))
    
    prog = st.progress(0) if progress else None
    
    for sim_i in range(num_sims):
        balance = float(start_balance)
        history = [balance]
        trade_results = []
        monthly_diffs = []
        current_month_start_bal = balance
        for t in range(1, total_trades + 1):
            if balance <= 0:
                balance = 0.0
                history.append(balance)
                trade_results.append(-1)
                continue
            rn = np.random.random() * 100
            v_factor = np.random.normal(1, variability / 100.0)
            if position_sizing == "Kelly":
                risk_amount = kelly * balance
                win_change = risk_amount * rr * v_factor
                loss_change = risk_amount * v_factor
            else:
                if unit == '%':
                    risk_amount = balance * (risk_val / 100.0)
                    win_change = balance * (reward_val * v_factor / 100.0)
                    loss_change = balance * (risk_val * v_factor / 100.0)
                else:
                    win_change = reward_val * v_factor
                    loss_change = risk_val * v_factor
            if rn < win_rate:
                balance += max(0.0, win_change)
                trade_results.append(1)
            elif rn < win_rate + be_rate:
                trade_results.append(0)
            else:
                balance -= max(0.0, loss_change)
                trade_results.append(-1)
            history.append(balance)
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
            "win_pct": (trade_results.count(1) / len(trade_results)) * 100 if trade_results else 0,
            "monthly_diffs": monthly_diffs
        })
        if progress:
            prog.progress((sim_i + 1) / num_sims)
    
    ror_count = sum(min(r["history"]) <= (ruin_threshold / 100.0 * start_balance) for r in all_runs)
    ror = (ror_count / num_sims) * 100 if num_sims > 0 else 0
    return all_runs, ror

# Run simulation
if st.button(t['run_sim']) and valid:
    with st.spinner(t['running']):
        results, ror = run_simulation(start_balance, mode, win_rate, be_rate, risk_val, reward_val, num_sims, trades_per_month, num_months, variability, position_sizing, ruin_threshold)
        
        finals = [r["final"] for r in results]
        idx_best = int(np.argmax(finals))
        idx_worst = int(np.argmin(finals))
        idx_median = int(np.argmin(np.abs(np.array(finals) - np.median(finals))))
        
        COLOR_BEST, COLOR_WORST, COLOR_MEDIAN = "#10B981", "#EF4444", "#3B82F6"
        
        template = "plotly_dark" if st.session_state.theme == 'dark' else "plotly_white"
        
        # Graph
        fig = go.Figure()
        for i, r in enumerate(results[:100]):
            if i not in [idx_best, idx_worst, idx_median]:
                fig.add_trace(go.Scatter(y=r["history"], mode='lines', line=dict(width=1, color="gray"), opacity=0.1, showlegend=False))
        fig.add_trace(go.Scatter(y=results[idx_median]["history"], name=t['most_possible'], line=dict(color=COLOR_MEDIAN, width=2)))
        fig.add_trace(go.Scatter(y=results[idx_worst]["history"], name=t['worst'], line=dict(color=COLOR_WORST, width=2)))
        fig.add_trace(go.Scatter(y=results[idx_best]["history"], name=t['best'], line=dict(color=COLOR_BEST, width=2)))
        fig.update_layout(title=t['balance_dynamics'], template=template, height=450, legend=dict(orientation="h", x=0.5, xanchor="center", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        
        st.metric(t['ror'], f"{ror:.1f}%")
        
        st.divider()
        
        # Detailed analysis
        st.write(f"<h2 style='text-align: center;'>{t['detailed_analysis']}</h2>", unsafe_allow_html=True)
        tab_med, tab_worst, tab_best = st.tabs([t['most_possible'], t['worst'], t['best']])
        
        def style_table(df):
            def color_vals(val):
                if isinstance(val, str) and '-' in val: return 'color: #EF4444'
                if isinstance(val, str) and '+' in val and val != '+0.0%': return 'color: #10B981'
                return ''
            return df.style.applymap(color_vals)
        
        def render_scenario(data):
            with st.container(border=True):
                c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
                c1.metric(t['initial_balance'], f"${start_balance:,.0f}")
                c2.metric(t['result_balance'], f"${data['final']:,.0f}")
                c3.metric(t['return_pct'], f"{((data['final']-start_balance)/start_balance)*100:.1f}%")
                c4.metric(t['max_drawdown'], f"-{data['mdd']:.1f}%")
                c5.metric(t['max_cons_loss'], data['max_losses'])
                c6.metric(t['max_cons_win'], data['max_wins'])
                c7.metric(t['win_trades_pct'], f"{data['win_pct']:.1f}%")
            st.write(f"#### {t['monthly_results']}")
            diffs = data['monthly_diffs']
            num_years = int(np.ceil(len(diffs) / 12))
            cols_years = st.columns(min(num_years, 3))
            months = t['months']
            
            for y in range(num_years):
                with cols_years[y % 3]:
                    year_data = diffs[y*12 : (y+1)*12]
                    rows = []
                    for i, val in enumerate(year_data):
                        pct = (val / start_balance) * 100 if start_balance != 0 else 0
                        rows.append({t['month']: months[i], t['results_pct']: f"{pct:+.1f}%", t['results_dol']: f"${val:,.0f}"})
                    st.write(f"**{t['year']} {2026 + y}**")
                    st.table(style_table(pd.DataFrame(rows)))
        
        with tab_med: render_scenario(results[idx_median])
        with tab_worst: render_scenario(results[idx_worst])
        with tab_best: render_scenario(results[idx_best])

# Sensitivity Analysis
st.header(t['sensitivity_analysis'])

param_options = {
    'win_rate': {'label': t['win_trades'], 'min': 0, 'max': 100, 'step': 1},
    'be_rate': {'label': t['be_trades'], 'min': 0, 'max': 100, 'step': 1},
    'risk_val': {'label': t['risk'], 'min': 0.1, 'max': 10, 'step': 0.1},
    'reward_val': {'label': t['reward'], 'min': 0.1, 'max': 20, 'step': 0.1},
    'variability': {'label': t['variability'], 'min': 0, 'max': 100, 'step': 1}
}

param_key = st.selectbox(t['param_to_vary'], list(param_options.keys()), format_func=lambda k: param_options[k]['label'])

min_val = st.number_input(t['min_val'], value=param_options[param_key]['min'], step=param_options[param_key]['step'])

max_val = st.number_input(t['max_val'], value=param_options[param_key]['max'], step=param_options[param_key]['step'])

steps = st.number_input(t['steps'], value=10, min_value=2, step=1)

base_kwargs = {
    'start_balance': start_balance,
    'mode': mode,
    'win_rate': win_rate,
    'be_rate': be_rate,
    'risk_val': risk_val,
    'reward_val': reward_val,
    'num_sims': num_sims,
    'trades_per_month': trades_per_month,
    'num_months': num_months,
    'variability': variability,
    'position_sizing': position_sizing,
    'ruin_threshold': ruin_threshold
}

def run_sensitivity(param, min_v, max_v, steps, base_kwargs):
    values = np.linspace(min_v, max_v, steps)
    avgs = []
    prog = st.progress(0)
    for j, val in enumerate(values):
        kwargs = base_kwargs.copy()
        kwargs[param] = val
        res, _ = run_simulation(**kwargs, progress=False)
        avg_final = np.mean([r['final'] for r in res])
        avgs.append(avg_final)
        prog.progress((j + 1) / steps)
    return values, avgs

if st.button(t['run_sens']) and valid:
    with st.spinner(t['running_sens']):
        values, avgs = run_sensitivity(param_key, min_val, max_val, steps, base_kwargs)
        template = "plotly_dark" if st.session_state.theme == 'dark' else "plotly_white"
        fig_sens = go.Figure(go.Scatter(x=values, y=avgs, mode='lines'))
        fig_sens.update_layout(
            title=t['sens_title'].format(param=param_options[param_key]['label']),
            xaxis_title=param_options[param_key]['label'],
            yaxis_title=t['avg_final_balance'],
            template=template
        )
        st.plotly_chart(fig_sens, use_container_width=True)
