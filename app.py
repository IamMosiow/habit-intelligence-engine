import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Habit Intelligence & Life OS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enhanced UI Styling
st.markdown("""
<style>
    .main-header { font-size: 2.3rem; font-weight: 800; color: #1E88E5; margin-bottom: 0px; }
    .sub-header { font-size: 1.05rem; color: #555; margin-bottom: 25px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #F0F4F8; border-radius: 8px 8px 0px 0px; padding-left: 20px; padding-right: 20px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #1E88E5 !important; color: white !important; }
    
    .insight-card { background-color: #FFFFFF; border-radius: 12px; padding: 18px; border: 1px solid #E0E0E0; box-shadow: 0 4px 6px rgba(0,0,0,0.04); margin-bottom: 15px; }
    .warning-box { background-color: #FFF3E0; border-radius: 10px; padding: 14px; border-left: 6px solid #FF9800; margin-bottom: 12px; color: #8C3B00; font-size: 0.95rem; }
    .success-box { background-color: #E8F5E9; border-radius: 10px; padding: 14px; border-left: 6px solid #4CAF50; margin-bottom: 12px; color: #1B5E20; font-size: 0.95rem; }
    .info-box { background-color: #E3F2FD; border-radius: 10px; padding: 14px; border-left: 6px solid #2196F3; margin-bottom: 12px; color: #0D47A1; font-size: 0.95rem; }
    .recommend-box { background-color: #F3E5F5; border-radius: 10px; padding: 14px; border-left: 6px solid #9C27B0; margin-bottom: 12px; color: #4A148C; font-size: 0.95rem; }
    .recovery-box { background-color: #FFEBEE; border-radius: 10px; padding: 14px; border-left: 6px solid #F44336; margin-bottom: 12px; color: #B71C1C; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Personal Habit Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Life OS with Dynamic Recovery Mode, Reality-Aware Goal Seeker & Burnout Protection</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📂 Data Import & Controls")
    uploaded_file = st.file_uploader("Upload tracking CSV", type=['csv'])
    st.markdown("---")
    st.markdown("### 📅 Simulation Date")
    selected_date = st.date_input("Target Date", datetime.date.today())
    day_name = selected_date.strftime("%A")
    st.info(f"📆 Day of Week: **{day_name}**")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Controls")
    enable_sleep_mode = st.sidebar.toggle("Track Exact Sleep Duration", value=True)
    enable_secondary = st.sidebar.toggle("Track Secondary Habits", value=True)

def parse_time_to_hours(val_str):
    if pd.isna(val_str) or str(val_str).strip() == '':
        return np.nan
    parts = str(val_str).strip().split(':')
    try:
        h = float(parts[0])
        m = float(parts[1]) if len(parts) > 1 else 0.0
        return h + m / 60.0
    except:
        return np.nan

@st.cache_data
def load_and_preprocess(file):
    df = pd.read_csv(file, on_bad_lines='skip')
    df['Date_parsed'] = pd.to_datetime(df['Date'], format='%m/%d/%y', errors='coerce')
    df['Activity_clean'] = df['Activity name'].str.strip()
    df['Value_num'] = pd.to_numeric(df['Value'], errors='coerce')

    core_habits = [
        'Mood At Night', 'Mood In Morning', 'Physical Health', 'Lunch Quality', 'Dinner Quality',
        'Do Exercise', 'Book', 'English', 'Quran', 'Water Intake', 'Social Media',
        'Wake up early', 'Sleep early', 'Morning Prayer', 'Night Prayer', 'No Soda', 'No J', 'Zekr',
        'Bedtime', 'Wake time'
    ]
    df_core = df[df['Activity_clean'].isin(core_habits)].copy()

    df_val = df_core.pivot_table(index='Date_parsed', columns='Activity_clean', values='Value_num', aggfunc='first')
    df_state = df_core.pivot_table(index='Date_parsed', columns='Activity_clean', values='State', aggfunc='first')
    df_raw_val = df_core.pivot_table(index='Date_parsed', columns='Activity_clean', values='Value', aggfunc='first')

    binary_cols = ['Wake up early', 'Sleep early', 'Morning Prayer', 'Night Prayer', 'No Soda', 'No J', 'Zekr']
    for col in binary_cols:
        if col in df_state.columns:
            df_val[f'{col}_done'] = (df_state[col] == 'Done').astype(int)

    # Sleep Duration Calculation
    if 'Bedtime' in df_raw_val.columns:
        df_val['Bedtime_hour'] = df_raw_val['Bedtime'].apply(parse_time_to_hours)
    else:
        df_val['Bedtime_hour'] = np.nan

    if 'Wake time' in df_raw_val.columns:
        df_val['Waketime_hour'] = df_raw_val['Wake time'].apply(parse_time_to_hours)
    else:
        df_val['Waketime_hour'] = np.nan

    df_val['Sleep_Duration_Hours'] = np.where(
        df_val['Bedtime_hour'].notna() & df_val['Waketime_hour'].notna(),
        (df_val['Waketime_hour'] - df_val['Bedtime_hour']) % 24.0,
        np.nan
    )
    
    fallback_sleep = pd.Series(
        np.where(df_val.get('Sleep early_done', 0) == 1, 7.5, 6.0),
        index=df_val.index
    )
    df_val['Sleep_Duration_Hours'] = df_val['Sleep_Duration_Hours'].fillna(fallback_sleep)

    # Native 1-10 Scale Conversion
    for mood_col in ['Mood At Night', 'Mood In Morning']:
        if mood_col in df_val.columns:
            is_old_scale = (df_val.index < pd.Timestamp('2026-08-13')) & (df_val[mood_col] <= 5.0) & (df_val[mood_col] > 0)
            df_val[mood_col] = np.where(is_old_scale, df_val[mood_col] * 2.0, df_val[mood_col])

    # Rolling Momentum
    for col in ['Mood At Night', 'Mood In Morning', 'Physical Health', 'Do Exercise', 'English', 'Book', 'Sleep_Duration_Hours']:
        if col in df_val.columns:
            df_val[f'{col}_roll3'] = df_val[col].shift(1).rolling(3, min_periods=1).mean()
            df_val[f'{col}_roll7'] = df_val[col].shift(1).rolling(7, min_periods=1).mean()
            df_val[f'{col}_roll14'] = df_val[col].shift(1).rolling(14, min_periods=1).mean()

    # Streaks
    if 'Do Exercise' in df_val.columns:
        ex_done = (df_val['Do Exercise'].fillna(0) > 0).astype(int)
        df_val['Exercise_Streak'] = ex_done.groupby((ex_done != ex_done.shift()).cumsum()).cumsum().shift(1).fillna(0)

    if 'English' in df_val.columns:
        eng_done = (df_val['English'].fillna(0) > 0).astype(int)
        df_val['English_Streak'] = eng_done.groupby((eng_done != eng_done.shift()).cumsum()).cumsum().shift(1).fillna(0)

    df_val['DayOfWeek'] = df_val.index.dayofweek
    df_val['DayName'] = df_val.index.day_name()
    return df_val

if uploaded_file is not None:
    df_proc = load_and_preprocess(uploaded_file)
    valid_history = df_proc.dropna(subset=['Mood At Night']).copy()
    valid_history = valid_history[valid_history['Mood At Night'] > 0]

    # Time-based Baselines & Velocity (Recent 7 days vs Previous 7 days)
    recent_7 = valid_history.tail(7)
    prev_7 = valid_history.iloc[-14:-7] if len(valid_history) >= 14 else recent_7
    yesterday_row = valid_history.iloc[-1]
    
    dow_map = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
    sim_dow_num = dow_map.get(day_name, 0)
    same_weekday_df = valid_history[valid_history['DayOfWeek'] == sim_dow_num]

    # Calculate current baselines and deltas
    def get_velocity(col_name, default_val):
        curr = recent_7[col_name].mean() if col_name in recent_7 else default_val
        prev = prev_7[col_name].mean() if col_name in prev_7 else curr
        return curr, curr - prev

    base_m_mood, delta_m_mood = get_velocity('Mood In Morning', 8.0)
    base_n_mood, delta_n_mood = get_velocity('Mood At Night', 8.0)
    base_health, delta_health = get_velocity('Physical Health', 8.5)
    base_ex, delta_ex = get_velocity('Do Exercise', 20.0)
    base_eng, delta_eng = get_velocity('English', 15.0)
    base_sleep, delta_sleep = get_velocity('Sleep_Duration_Hours', 7.5)

    curr_ex_streak = int(recent_7['Exercise_Streak'].iloc[-1]) if 'Exercise_Streak' in recent_7 else 0
    curr_eng_streak = int(recent_7['English_Streak'].iloc[-1]) if 'English_Streak' in recent_7 else 0

    dow_historical_avg = same_weekday_df['Mood At Night'].mean() if len(same_weekday_df) > 0 else 8.0
    dow_ex_avg = same_weekday_df['Do Exercise'].mean() if len(same_weekday_df) > 0 else 20.0

    # Cognitive vs Physical Load Ratio
    cog_load_7 = recent_7['English'].mean() + recent_7['Book'].mean() * 3.0
    phys_load_7 = recent_7['Do Exercise'].mean()
    burnout_ratio = (cog_load_7 / (phys_load_7 + 1.0))

    # Tracking Consistency Score
    days_in_last_30 = len(valid_history[valid_history.index >= (valid_history.index.max() - pd.Timedelta(days=30))])
    consistency_rate = min((days_in_last_30 / 30.0) * 100, 100)

    # State check: Is the user feeling unwell? (Score <= 6.5)
    is_recovery_mode_auto = yesterday_row.get('Mood In Morning', 8.0) <= 6.5 or yesterday_row.get('Mood At Night', 8.0) <= 6.5

    # Train Model
    feature_cols = [c for c in valid_history.columns if c not in ['Mood At Night', 'DayName', 'Bedtime', 'Wake time']]
    X = valid_history[feature_cols].fillna(0)
    y = valid_history['Mood At Night']
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    tab_auto, tab_sim, tab_opt, tab_analytics = st.tabs([
        "🔮 Automated Recommendations", 
        "⚙️ Scenario Simulation", 
        "🎯 Goal Seeker",
        "📈 Analytics & Velocity"
    ])

    # =========================================================================
    # TAB 1: AUTOMATED RECOMMENDATIONS
    # =========================================================================
    with tab_auto:
        st.markdown(f"## 🤖 Passive Life OS Recommendations ({day_name})")
        
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("7-Day Morning Mood", f"{base_m_mood:.1f} / 10", delta=f"{delta_m_mood:+.1f} vs Last Wk")
        a2.metric("7-Day Exercise Avg", f"{base_ex:.0f} mins", delta=f"{delta_ex:+.0f} mins")
        a3.metric("7-Day Sleep Avg", f"{base_sleep:.1f} hrs", delta=f"{delta_sleep:+.1f} hrs")
        a4.metric("30-Day Tracking Consistency", f"{consistency_rate:.0f}%", delta="Optimal" if consistency_rate>80 else "Needs logging")

        st.markdown("---")
        auto_w, auto_p, auto_r = st.columns(3)

        auto_warnings = []
        y_sleep_hours = yesterday_row.get('Sleep_Duration_Hours', 7.5)
        if y_sleep_hours < 6.0:
            auto_warnings.append(f"⚠️ **Sleep Deficit**: Only **{y_sleep_hours:.1f} hours** of sleep last night.")
        
        # Only issue streak threats if NOT in recovery mode
        if not is_recovery_mode_auto and curr_ex_streak >= 3 and yesterday_row.get('Do Exercise', 0) == 0:
            auto_warnings.append(f"⚠️ **Streak Threat**: Your {curr_ex_streak}-day workout streak is at risk.")
            
        if burnout_ratio > 2.5 and not is_recovery_mode_auto:
            auto_warnings.append(f"🧠 **Burnout Alert**: Mental load heavily exceeds physical recovery. Prioritize rest or light exercise.")

        with auto_w:
            st.subheader("🚨 Risk Alerts")
            if is_recovery_mode_auto:
                st.markdown('<div class="recovery-box">🩺 **Recovery Mode Active**: You recently logged a low mood score. All streak warnings and productivity threats are paused. Focus on your health today.</div>', unsafe_allow_html=True)
            elif auto_warnings:
                for w in auto_warnings:
                    st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ No passive risk flags detected!</div>', unsafe_allow_html=True)

        auto_positives = []
        if delta_health > 0.5:
            auto_positives.append(f"📈 **Health Velocity**: Physical health is actively trending up (+{delta_health:.1f} vs last week).")
        if y_sleep_hours >= 7.5:
            auto_positives.append(f"🌙 **Optimal Sleep**: You secured **{y_sleep_hours:.1f}h** of sleep.")
        if curr_ex_streak >= 3 and not is_recovery_mode_auto:
            auto_positives.append(f"🔥 **Exercise Multiplier**: You are on a {curr_ex_streak}-day workout streak!")

        with auto_p:
            st.subheader("🌟 Active Multipliers")
            if auto_positives:
                for p in auto_positives:
                    st.markdown(f'<div class="success-box">{p}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">💡 Focus on basic recovery and sleep to rebuild momentum.</div>', unsafe_allow_html=True)

        auto_recs = []
        if is_recovery_mode_auto:
            auto_recs.append("🩺 **Rest is Progress**: It is completely fine to break a streak today. Your only goals are hydration and deep sleep.")
            auto_recs.append("💧 **Hydration Focus**: Ensure you get 8+ glasses of water today.")
        else:
            auto_recs.append(f"📅 **{day_name} Benchmark**: Target **{dow_ex_avg:.0f} mins** of exercise today.")
            if day_name == 'Wednesday':
                auto_recs.append("🎯 **Wednesday Shield**: Lock in 15+ minutes of book reading early in the day to counter mid-week slump.")
        
        if y_sleep_hours < 6.0:
            auto_recs.append("🎯 **Sleep Debt Recovery**: Plan a bedtime before 23:30 tonight.")

        with auto_r:
            st.subheader(f"💡 Recommended Plan")
            for r in auto_recs:
                st.markdown(f'<div class="recommend-box">{r}</div>', unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: SIMULATION & CHECK-IN
    # =========================================================================
    with tab_sim:
        st.markdown("## ⚙️ Interactive Scenario Simulation")
        
        with st.expander("⚙️ Adjust Today's Inputs", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("### 🌅 Morning State")
                plan_m_mood = st.slider("Morning Mood (1-10)", 1.0, 10.0, float(round(base_m_mood, 1)), 0.5)
                plan_health = st.slider("Physical Health (1-10)", 1.0, 10.0, float(round(base_health, 1)), 0.5)
                
                st.markdown("---")
                has_night_mood = st.checkbox("Log Actual Night Mood")
                if has_night_mood:
                    plan_n_mood = st.slider("Actual Night Mood Logged (1-10)", 1.0, 10.0, float(round(base_n_mood, 1)), 0.5)
                else:
                    plan_n_mood = None

            with col2:
                st.markdown("### 💤 Sleep Target")
                if enable_sleep_mode:
                    plan_bed_time = st.time_input("Planned Bedtime", datetime.time(2, 30))
                    plan_wake_time = st.time_input("Planned Wake Time", datetime.time(10, 30))
                    b_dec = plan_bed_time.hour + plan_bed_time.minute / 60.0
                    w_dec = plan_wake_time.hour + plan_wake_time.minute / 60.0
                    calc_sleep_dur = (w_dec - b_dec) % 24.0
                    st.info(f"⏱️ Sleep Duration: **{calc_sleep_dur:.1f} Hours**")
                else:
                    b_dec, w_dec, calc_sleep_dur = 23.5, 7.0, 7.5

                plan_sleep_early = st.checkbox("Slept Early Last Night", value=bool(b_dec <= 24.0 or b_dec <= 1.0))
                plan_wake_early = st.checkbox("Woke Up Early Today", value=bool(w_dec <= 7.5))

            with col3:
                st.markdown("### 📚 Productivity")
                plan_exercise = st.slider("Exercise (Minutes)", 0, 180, int(base_ex), 5)
                plan_english = st.slider("English Study (Minutes)", 0, 180, int(base_eng), 5)
                plan_book = st.slider("Book Reading (Pages)", 0, 50, 10, 1)

            with col4:
                st.markdown("### 🥗 Nutrition & Discp.")
                plan_dinner = st.slider("Dinner Quality (1-10)", 1.0, 10.0, 8.0, 0.5)
                plan_lunch = st.slider("Lunch Quality (1-10)", 1.0, 10.0, 8.0, 0.5)
                
                if enable_secondary:
                    plan_water = st.slider("Water (Glasses)", 0, 12, 8, 1)
                    plan_no_soda = st.checkbox("No Soda Today", value=True)
                else:
                    plan_water, plan_no_soda = 8, True

        scenario_dict = yesterday_row[feature_cols].to_dict()
        scenario_dict.update({
            'Mood In Morning': plan_m_mood,
            'Physical Health': plan_health,
            'Do Exercise': plan_exercise,
            'English': plan_english,
            'Book': plan_book,
            'Lunch Quality': plan_lunch,
            'Dinner Quality': plan_dinner,
            'Water Intake': plan_water * 250,
            'Bedtime_hour': b_dec,
            'Waketime_hour': w_dec,
            'Sleep_Duration_Hours': calc_sleep_dur,
            'Sleep early_done': int(plan_sleep_early),
            'Wake up early_done': int(plan_wake_early),
            'No Soda_done': int(plan_no_soda),
            'DayOfWeek': sim_dow_num
        })

        scenario_df = pd.DataFrame([scenario_dict]).reindex(columns=feature_cols, fill_value=0)
        simulated_pred_mood = rf.predict(scenario_df)[0]

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        with m1:
            if has_night_mood:
                st.metric("Actual Logged Evening Mood", f"{plan_n_mood:.1f} / 10.0", delta=f"{plan_n_mood - simulated_pred_mood:+.1f} vs Predicted")
            else:
                st.metric("Predicted Evening Mood", f"{simulated_pred_mood:.1f} / 10.0", delta=f"{simulated_pred_mood - dow_historical_avg:+.1f} vs {day_name} Avg")
        with m2:
            st.metric(f"Historical {day_name} Avg Mood", f"{dow_historical_avg:.1f} / 10.0")
        with m3:
            st.metric("7-Day Baseline Mood", f"{base_m_mood:.1f} / 10.0")

        # RECOVERY MODE CHECK FOR SIMULATION
        is_recovery_sim = plan_m_mood <= 6.5

        st.markdown("---")
        st.subheader("💡 Reactive Sensitivities")
        r1, r2 = st.columns(2)
        
        with r1:
            if is_recovery_sim:
                st.markdown('<div class="recovery-box">🩺 **Recovery Protocol Engaged**: Your morning mood is low. Standard productivity targets are muted.</div>', unsafe_allow_html=True)
                if plan_exercise > 0:
                    st.markdown('<div class="info-box">💡 **Rest over Streaks**: You planned exercise, but it is entirely okay to change this to 0 mins to recover your baseline.</div>', unsafe_allow_html=True)
                if plan_english > 0 or plan_book > 0:
                    st.markdown('<div class="info-box">🧠 **Cognitive Rest**: Keep mental load light today. Skip heavy English or reading if you feel exhausted.</div>', unsafe_allow_html=True)
            else:
                if plan_exercise < 30:
                    test_df = scenario_df.copy()
                    test_df['Do Exercise'] += 15
                    bump = rf.predict(test_df)[0] - simulated_pred_mood
                    if bump > 0.05:
                        st.markdown(f'<div class="recommend-box">💡 **Exercise ROI**: +15 mins exercise = **+{bump:.2f} predicted mood**.</div>', unsafe_allow_html=True)
                if plan_english < 20:
                    test_df = scenario_df.copy()
                    test_df['English'] += 15
                    bump = rf.predict(test_df)[0] - simulated_pred_mood
                    if bump > 0.05:
                        st.markdown(f'<div class="info-box">📚 **Study ROI**: +15 mins English = **+{bump:.2f} predicted mood**.</div>', unsafe_allow_html=True)

        with r2:
            if calc_sleep_dur < 6.0:
                st.markdown('<div class="warning-box">⚠️ **Sleep Deficit**: Planned sleep is low. Prioritize rest tonight.</div>', unsafe_allow_html=True)
            if not is_recovery_sim and plan_exercise < dow_ex_avg:
                st.markdown(f'<div class="recommend-box">📅 **{day_name} Benchmark**: You are below your usual {day_name} exercise average ({dow_ex_avg:.0f}m).</div>', unsafe_allow_html=True)
            if is_recovery_sim:
                st.markdown(f'<div class="success-box">💧 **Nourishment Focus**: Hydration ({plan_water} glasses) and high dinner quality ({plan_dinner}/10) are your only real goals today.</div>', unsafe_allow_html=True)

        # Plan Summary Export
        st.markdown("---")
        checklist_text = f"""# Action Plan - {selected_date.strftime('%Y-%m-%d')}
Predicted Mood: {simulated_pred_mood:.1f} / 10.0
- [ ] Exercise: {plan_exercise} Min
- [ ] English: {plan_english} Min
- [ ] Book: {plan_book} Pages
- [ ] Sleep Target: {calc_sleep_dur:.1f}h
"""
        st.download_button("📥 Download Checklist (.md)", data=checklist_text, file_name=f"Plan_{selected_date.strftime('%Y-%m-%d')}.md")

    # =========================================================================
    # TAB 3: REALITY-AWARE GOAL SEEKER
    # =========================================================================
    with tab_opt:
        st.markdown("## 🎯 Reality-Aware Goal Seeker")
        st.caption("Select your desired evening mood score. If you are unwell, the optimizer automatically adjusts to find the best possible realistic outcome.")

        target_score = st.slider("Desired Evening Mood (1-10)", 6.0, 10.0, 8.5, 0.1)

        if st.button("🚀 Find Optimal Habit Plan"):
            best_target_plan = None
            best_max_plan = None
            max_possible_mood = -1
            min_effort = 999999
            
            # If morning mood is low, penalize heavy physical/mental effort heavily in the solver
            is_sick = plan_m_mood <= 6.5

            for ex_val in [0, 15, 30, 45, 60]:
                for eng_val in [0, 15, 30, 45]:
                    for din_val in [7.0, 8.5, 10.0]:
                        for bk_val in [0, 5, 15]:
                            test_dict = yesterday_row[feature_cols].to_dict()
                            test_dict.update({
                                'Mood In Morning': plan_m_mood,
                                'Physical Health': plan_health,
                                'Do Exercise': ex_val,
                                'English': eng_val,
                                'Book': bk_val,
                                'Dinner Quality': din_val,
                                'Lunch Quality': 8.0,
                                'Sleep_Duration_Hours': 7.5,
                                'Sleep early_done': 1,
                                'Wake up early_done': 1,
                                'DayOfWeek': sim_dow_num
                            })
                            t_df = pd.DataFrame([test_dict]).reindex(columns=feature_cols, fill_value=0)
                            p_mood = rf.predict(t_df)[0]
                            
                            # Track absolute maximum achievable mood
                            if p_mood > max_possible_mood:
                                max_possible_mood = p_mood
                                best_max_plan = {'Exercise': ex_val, 'English': eng_val, 'Book': bk_val, 'Dinner': din_val, 'Predicted Mood': p_mood}

                            # Track minimum effort to reach user target
                            if p_mood >= target_score - 0.15:
                                if is_sick:
                                    # Penalize effort heavily if sick so the solver prefers 0 exercise
                                    effort = (ex_val * 3) + (eng_val * 2) + (bk_val * 2)
                                else:
                                    effort = ex_val + eng_val + (bk_val * 2)
                                    
                                if effort < min_effort:
                                    min_effort = effort
                                    best_target_plan = {'Exercise': ex_val, 'English': eng_val, 'Book': bk_val, 'Dinner': din_val, 'Predicted Mood': p_mood}

            if best_target_plan:
                st.success(f"🎯 **Optimal Minimum Plan Found!** Expected Evening Mood: **{best_target_plan['Predicted Mood']:.1f} / 10.0**")
                o1, o2, o3, o4 = st.columns(4)
                o1.metric("Exercise Required", f"{best_target_plan['Exercise']} min")
                o2.metric("English Study Required", f"{best_target_plan['English']} min")
                o3.metric("Book Reading Required", f"{best_target_plan['Book']} pages")
                o4.metric("Dinner Quality Target", f"{best_target_plan['Dinner']:.1f}/10")
                if best_target_plan['Exercise'] == 0 and is_sick:
                    st.info("🩺 The optimizer correctly determined that keeping exercise at 0 mins minimizes effort while you recover.")
            else:
                st.warning(f"⚠️ Your target of {target_score} is mathematically out of reach today due to your morning baseline ({plan_m_mood:.1f}/10).")
                st.info(f"However, here is the absolute BEST realistic plan to reach the maximum possible score of **{max_possible_mood:.1f} / 10.0**:")
                
                o1, o2, o3, o4 = st.columns(4)
                o1.metric("Optimal Exercise", f"{best_max_plan['Exercise']} min")
                o2.metric("Optimal English", f"{best_max_plan['English']} min")
                o3.metric("Optimal Reading", f"{best_max_plan['Book']} pages")
                o4.metric("Dinner Quality Requirement", f"{best_max_plan['Dinner']:.1f}/10")

    # =========================================================================
    # TAB 4: ANALYTICS
    # =========================================================================
    with tab_analytics:
        st.markdown("## 📈 Deep Analytics & Velocity")
        
        fig, ax = plt.subplots(figsize=(10, 3))
        recent_60 = valid_history.tail(60)
        ax.plot(recent_60.index, recent_60['Mood At Night'], label='Daily Mood', alpha=0.35, color='gray', linestyle='--')
        ax.plot(recent_60.index, recent_60['Mood At Night_roll7'], label='7-Day Trend', color='#4CAF50', linewidth=2.5)
        ax.set_title("60-Day Rolling Momentum Trends")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)

else:
    st.info("👆 Upload your habit tracking CSV file in the sidebar to launch the system.")
