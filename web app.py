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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🧠 Personal Habit Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Life OS with Precision Sleep Duration Engine, Native 1-10 Mood Scaling & Automated Prescriptions</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📂 Data Import & Controls")
    uploaded_file = st.file_uploader("Upload updated tracking CSV", type=['csv'])
    st.markdown("---")
    st.markdown("### 📅 Simulation Date")
    selected_date = st.date_input("Simulation Date", datetime.date.today())
    day_name = selected_date.strftime("%A")
    st.info(f"📆 Day of Week: **{day_name}**")
    
    st.markdown("---")
    st.markdown("### ⚙️ System Controls")
    enable_sleep_mode = st.sidebar.toggle("Track Exact Bedtime & Sleep Duration", value=True)
    enable_secondary = st.sidebar.toggle("Track Secondary Habits (Soda, Water, Screen)", value=True)

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

    # Pivot tables
    df_val = df_core.pivot_table(index='Date_parsed', columns='Activity_clean', values='Value_num', aggfunc='first')
    df_state = df_core.pivot_table(index='Date_parsed', columns='Activity_clean', values='State', aggfunc='first')
    df_raw_val = df_core.pivot_table(index='Date_parsed', columns='Activity_clean', values='Value', aggfunc='first')

    # Binary flags
    binary_cols = ['Wake up early', 'Sleep early', 'Morning Prayer', 'Night Prayer', 'No Soda', 'No J', 'Zekr']
    for col in binary_cols:
        if col in df_state.columns:
            df_val[f'{col}_done'] = (df_state[col] == 'Done').astype(int)

    # 1. Exact Bedtime & Wake Time Sleep Duration Calculation
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
    
    # Safe Fillna using pd.Series to prevent TypeError
    fallback_sleep = pd.Series(
        np.where(df_val.get('Sleep early_done', 0) == 1, 7.5, 6.0),
        index=df_val.index
    )
    df_val['Sleep_Duration_Hours'] = df_val['Sleep_Duration_Hours'].fillna(fallback_sleep)

    # 2. Standardize All Historical Mood Entries to Unified 1-10 Scale
    for mood_col in ['Mood At Night', 'Mood In Morning']:
        if mood_col in df_val.columns:
            is_old_scale = (df_val.index < pd.Timestamp('2026-08-13')) & (df_val[mood_col] <= 5.0) & (df_val[mood_col] > 0)
            df_val[mood_col] = np.where(is_old_scale, df_val[mood_col] * 2.0, df_val[mood_col])

    # 3. Rolling Momentum Features
    for col in ['Mood At Night', 'Mood In Morning', 'Physical Health', 'Do Exercise', 'English', 'Book', 'Sleep_Duration_Hours']:
        if col in df_val.columns:
            df_val[f'{col}_roll3'] = df_val[col].shift(1).rolling(3, min_periods=1).mean()
            df_val[f'{col}_roll7'] = df_val[col].shift(1).rolling(7, min_periods=1).mean()
            df_val[f'{col}_roll14'] = df_val[col].shift(1).rolling(14, min_periods=1).mean()

    # 4. Habit Streaks
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
    latest_date = df_proc.index.max()

    valid_history = df_proc.dropna(subset=['Mood At Night']).copy()
    valid_history = valid_history[valid_history['Mood At Night'] > 0]

    # Pre-compute baselines
    recent_14 = valid_history.tail(14)
    recent_7 = valid_history.tail(7)
    yesterday_row = valid_history.iloc[-1]
    
    dow_map = {'Monday':0, 'Tuesday':1, 'Wednesday':2, 'Thursday':3, 'Friday':4, 'Saturday':5, 'Sunday':6}
    sim_dow_num = dow_map.get(day_name, 0)
    same_weekday_df = valid_history[valid_history['DayOfWeek'] == sim_dow_num]

    base_m_mood = recent_14['Mood In Morning'].mean() if 'Mood In Morning' in recent_14 else 8.0
    base_n_mood = recent_14['Mood At Night'].mean() if 'Mood At Night' in recent_14 else 8.0
    base_health = recent_14['Physical Health'].mean() if 'Physical Health' in recent_14 else 8.5
    base_ex = recent_14['Do Exercise'].mean() if 'Do Exercise' in recent_14 else 20.0
    base_eng = recent_14['English'].mean() if 'English' in recent_14 else 15.0
    base_sleep_hours = recent_14['Sleep_Duration_Hours'].mean() if 'Sleep_Duration_Hours' in recent_14 else 7.5
    curr_ex_streak = int(recent_14['Exercise_Streak'].iloc[-1]) if 'Exercise_Streak' in recent_14 else 0
    curr_eng_streak = int(recent_14['English_Streak'].iloc[-1]) if 'English_Streak' in recent_14 else 0

    dow_historical_avg = same_weekday_df['Mood At Night'].mean() if len(same_weekday_df) > 0 else 8.0
    dow_ex_avg = same_weekday_df['Do Exercise'].mean() if len(same_weekday_df) > 0 else 20.0
    dow_eng_avg = same_weekday_df['English'].mean() if len(same_weekday_df) > 0 else 15.0

    # Cognitive vs Physical Load Ratio (Burnout Index)
    cog_load_7 = recent_7['English'].mean() + recent_7['Book'].mean() * 3.0
    phys_load_7 = recent_7['Do Exercise'].mean()
    burnout_ratio = (cog_load_7 / (phys_load_7 + 1.0))

    # Train Random Forest Regressor
    feature_cols = [c for c in valid_history.columns if c not in ['Mood At Night', 'DayName', 'Bedtime', 'Wake time']]
    X = valid_history[feature_cols].fillna(0)
    y = valid_history['Mood At Night']

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Navigation Tabs
    tab_auto, tab_sim, tab_opt, tab_analytics = st.tabs([
        "🔮 Automated Recommendations", 
        "⚙️ Scenario Simulation & Recommendations", 
        "🎯 Target Goal Seeker",
        "📈 Deep Analytics & Burnout Engine"
    ])

    # =========================================================================
    # TAB 1: AUTOMATED RECOMMENDATIONS (PASSIVE INTELLIGENCE)
    # =========================================================================
    with tab_auto:
        st.markdown(f"## 🤖 Automated Life OS Recommendations for Today ({day_name})")
        st.caption("Generated automatically using exact sleep hours, 1-10 scaled mood baselines, and historical weekday patterns.")

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Today's Target Evening Mood", f"{dow_historical_avg:.1f} / 10.0")
        a2.metric(f"Past {day_name}s Exercise Avg", f"{dow_ex_avg:.0f} mins")
        a3.metric(f"Past {day_name}s English Avg", f"{dow_eng_avg:.0f} mins")
        a4.metric("Active Workout Streak", f"{curr_ex_streak} days")

        st.markdown("---")
        auto_w, auto_p, auto_r = st.columns(3)

        auto_warnings = []
        y_sleep_hours = yesterday_row.get('Sleep_Duration_Hours', 7.5)
        if y_sleep_hours < 6.0:
            auto_warnings.append(f"⚠️ **Sleep Deficit Detected**: You logged only **{y_sleep_hours:.1f} hours** of sleep last night. Expect lower physical stamina today.")
        if curr_ex_streak >= 3 and yesterday_row.get('Do Exercise', 0) == 0:
            auto_warnings.append(f"⚠️ **Workout Streak Threat**: Your active {curr_ex_streak}-day exercise streak is at risk of breaking.")
        if yesterday_row.get('Mood In Morning', 8.0) < base_m_mood - 1.5:
            auto_warnings.append(f"⚠️ **Morning Energy Dip**: Recent morning mood ({yesterday_row.get('Mood In Morning'):.1f}/10) is below your 14-day baseline.")
        if day_name == 'Wednesday':
            auto_warnings.append("📅 **Wednesday Mid-Week Pattern**: Wednesdays historically record your lowest weekly average night mood (~7.8/10).")
        if burnout_ratio > 2.5:
            auto_warnings.append(f"🧠 **Burnout Alert (Index {burnout_ratio:.1f})**: Mental study load exceeds physical recovery. Prioritize a workout today.")

        with auto_w:
            st.subheader("🚨 Automated Risk Alerts")
            if auto_warnings:
                for w in auto_warnings:
                    st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ No passive risk flags detected for today!</div>', unsafe_allow_html=True)

        auto_positives = []
        if y_sleep_hours >= 7.5:
            auto_positives.append(f"🌙 **Optimal Sleep Duration**: You secured **{y_sleep_hours:.1f} hours** of sleep, providing a solid biological foundation.")
        if yesterday_row.get('Do Exercise', 0) >= 20:
            auto_positives.append(f"🏋️ **Workout Momentum**: Completed {yesterday_row.get('Do Exercise'):.0f} mins of exercise yesterday.")
        if yesterday_row.get('Dinner Quality', 0) >= 8.5:
            auto_positives.append("🥗 **High Dinner Quality Anchor**: Dinner quality yesterday was high (>=8.5), anchoring morning energy.")
        if curr_ex_streak >= 3:
            auto_positives.append(f"🔥 **Exercise Compound Multiplier**: You are on a {curr_ex_streak}-day workout streak!")

        with auto_p:
            st.subheader("🌟 Active Compound Multipliers")
            if auto_positives:
                for p in auto_positives:
                    st.markdown(f'<div class="success-box">{p}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">💡 Complete workout and study goals today to trigger positive multipliers.</div>', unsafe_allow_html=True)

        auto_recs = []
        auto_recs.append(f"📅 **{day_name} Exercise Benchmark**: Target at least **{dow_ex_avg:.0f} minutes** of exercise to match your historical benchmark.")
        auto_recs.append(f"📚 **{day_name} Study Target**: Complete at least **{dow_eng_avg:.0f} minutes** of English study today.")
        if y_sleep_hours < 6.0:
            auto_recs.append("🎯 **Sleep Debt Recovery**: Avoid heavy exhaustion today. Schedule a short nap or plan a bedtime before 23:30 tonight.")
        if day_name == 'Wednesday':
            auto_recs.append("🎯 **Wednesday Shield**: Lock in 15+ minutes of book reading early in the day to counter mid-week slump.")
        if curr_ex_streak >= 3 and yesterday_row.get('Do Exercise', 0) == 0:
            auto_recs.append("🎯 **Streak Protector**: Do a quick 10-minute walk or stretching routine today to preserve your streak.")

        with auto_r:
            st.subheader(f"💡 Recommended Plan for {day_name}")
            for r in auto_recs:
                st.markdown(f'<div class="recommend-box">{r}</div>', unsafe_allow_html=True)

    # =========================================================================
    # TAB 2: INTERACTIVE SCENARIO SIMULATION & LIVE RECOMMENDATIONS
    # =========================================================================
    with tab_sim:
        st.markdown("## ⚙️ Interactive Scenario Simulation (1-10 Scale & Precision Sleep)")
        st.caption("Adjust planned habit inputs below. The engine calculates real-time ML predictions and provides reactive optimization advice.")

        with st.expander("⚙️ Today's Simulation Inputs", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown("### 🌅 Mood & Health (1-10 Scale)")
                plan_m_mood = st.slider("Morning Mood (1-10)", 1.0, 10.0, float(round(base_m_mood, 1)), 0.5)
                plan_health = st.slider("Physical Health (1-10)", 1.0, 10.0, float(round(base_health, 1)), 0.5)
                
                st.markdown("---")
                has_night_mood = st.checkbox("Log Actual Night Mood (End of Day Check-in)")
                if has_night_mood:
                    plan_n_mood = st.slider("Actual Night Mood Logged (1-10)", 1.0, 10.0, float(round(base_n_mood, 1)), 0.5)
                else:
                    plan_n_mood = None

            with col2:
                st.markdown("### 💤 Sleep Schedule & Duration")
                if enable_sleep_mode:
                    plan_bed_time = st.time_input("Planned Bedtime", datetime.time(2, 30))
                    plan_wake_time = st.time_input("Planned Wake Time", datetime.time(10, 30))
                    
                    b_dec = plan_bed_time.hour + plan_bed_time.minute / 60.0
                    w_dec = plan_wake_time.hour + plan_wake_time.minute / 60.0
                    calc_sleep_dur = (w_dec - b_dec) % 24.0
                    st.info(f"⏱️ Calculated Sleep: **{calc_sleep_dur:.1f} Hours**")
                else:
                    b_dec = 23.5
                    w_dec = 7.0
                    calc_sleep_dur = 7.5

                plan_sleep_early = st.checkbox("Slept Early Last Night", value=bool(b_dec <= 24.0 or b_dec <= 1.0))
                plan_wake_early = st.checkbox("Woke Up Early Today", value=bool(w_dec <= 7.5))

            with col3:
                st.markdown("### 📚 Productivity & Study")
                plan_exercise = st.slider("Exercise (Minutes)", 0, 180, int(base_ex), 5)
                plan_english = st.slider("English Study (Minutes)", 0, 180, int(base_eng), 5)
                plan_book = st.slider("Book Reading (Pages)", 0, 50, 10, 1)
                plan_quran = st.slider("Quran (Pages)", 0, 10, 1, 1)

            with col4:
                st.markdown("### 🥗 Meals & Disciplines")
                plan_lunch = st.slider("Lunch Quality (1-10)", 1.0, 10.0, 8.0, 0.5)
                plan_dinner = st.slider("Dinner Quality (1-10)", 1.0, 10.0, 8.0, 0.5)
                if enable_secondary:
                    plan_water = st.slider("Water Intake (Glasses)", 0, 12, 8, 1)
                    plan_screen = st.slider("Screen Time (Min)", 0, 180, 30, 5)
                    plan_no_soda = st.checkbox("No Soda Today", value=True)
                    plan_no_j = st.checkbox("No J Today", value=True)
                else:
                    plan_water = 8
                    plan_screen = 30
                    plan_no_soda = True
                    plan_no_j = True

                plan_m_prayer = st.checkbox("Morning Prayer Completed", value=True)
                plan_n_prayer = st.checkbox("Night Prayer Completed", value=True)
                plan_zekr = st.checkbox("Zekr Completed", value=True)

        scenario_dict = yesterday_row[feature_cols].to_dict()
        scenario_dict.update({
            'Mood In Morning': plan_m_mood,
            'Physical Health': plan_health,
            'Do Exercise': plan_exercise,
            'English': plan_english,
            'Book': plan_book,
            'Quran': plan_quran,
            'Lunch Quality': plan_lunch,
            'Dinner Quality': plan_dinner,
            'Water Intake': plan_water * 250,
            'Social Media': plan_screen,
            'Bedtime_hour': b_dec,
            'Waketime_hour': w_dec,
            'Sleep_Duration_Hours': calc_sleep_dur,
            'Sleep early_done': int(plan_sleep_early),
            'Wake up early_done': int(plan_wake_early),
            'Morning Prayer_done': int(plan_m_prayer),
            'Night Prayer_done': int(plan_n_prayer),
            'Zekr_done': int(plan_zekr),
            'No Soda_done': int(plan_no_soda),
            'No J_done': int(plan_no_j),
            'DayOfWeek': sim_dow_num
        })

        scenario_df = pd.DataFrame([scenario_dict]).reindex(columns=feature_cols, fill_value=0)
        simulated_pred_mood = rf.predict(scenario_df)[0]

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            if has_night_mood:
                st.metric("Actual Logged Evening Mood", f"{plan_n_mood:.1f} / 10.0", delta=f"{plan_n_mood - simulated_pred_mood:+.1f} vs Predicted")
            else:
                st.metric("Predicted Evening Mood", f"{simulated_pred_mood:.1f} / 10.0", delta=f"{simulated_pred_mood - dow_historical_avg:+.1f} vs {day_name} Avg")
        with m2:
            st.metric(f"Historical {day_name} Avg Mood", f"{dow_historical_avg:.1f} / 10.0")
        with m3:
            st.metric("14-Day Baseline Mood", f"{base_m_mood:.1f} / 10.0")
        with m4:
            st.metric("Training Set Size", f"{len(valid_history)} days")

        # Live Recommendations & Sensitivity Tips
        st.markdown("---")
        sim_w, sim_s, sim_r = st.columns(3)

        sim_warnings = []
        if calc_sleep_dur < 6.0:
            sim_warnings.append(f"⚠️ **Sleep Duration Deficit**: Planned sleep is only **{calc_sleep_dur:.1f}h** (recommended: 7.0–8.5h).")
        if plan_health < (base_health - 1.5):
            sim_warnings.append(f"⚠️ **Physical Health Drop**: Planned health score ({plan_health:.1f}) is 1.5+ points below your 14-day average ({base_health:.1f}).")
        if plan_exercise == 0 and curr_ex_streak >= 3:
            sim_warnings.append(f"⚠️ **Workout Streak Threat**: 0 minutes of exercise will break your active {curr_ex_streak}-day workout streak.")
        if plan_m_mood < base_m_mood - 1.5:
            sim_warnings.append(f"⚠️ **Low Morning Mood Baseline**: Morning mood ({plan_m_mood:.1f}/10) is below recent average ({base_m_mood:.1f}).")
        if enable_secondary and not plan_no_soda:
            sim_warnings.append("🥤 **Soda Consumption**: Consuming soda historically correlates with a 50% decrease in exercise output.")

        with sim_w:
            st.subheader("🚨 Planned Schedule Warnings")
            if sim_warnings:
                for w in sim_warnings:
                    st.markdown(f'<div class="warning-box">{w}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">✅ Planned schedule has zero behavioral risk flags!</div>', unsafe_allow_html=True)

        sim_positives = []
        if calc_sleep_dur >= 7.5:
            sim_positives.append(f"🌙 **Healthy Sleep Duration**: Planned {calc_sleep_dur:.1f}h sleep anchors physical recovery.")
        if plan_exercise >= 20:
            sim_positives.append(f"🏋️ **Solid Workout Session**: Logging {plan_exercise} mins of exercise elevates evening mood.")
        if plan_english >= 15:
            sim_positives.append(f"📖 **English Goal Met**: Completing {plan_english} mins of English study boosts overall daily satisfaction.")
        if plan_dinner >= 8.5:
            sim_positives.append("🥗 **High Dinner Quality**: High dinner quality is a top predictor for tomorrow morning's energy.")

        with sim_s:
            st.subheader("🌟 Simulated Multipliers")
            if sim_positives:
                for p in sim_positives:
                    st.markdown(f'<div class="success-box">{p}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-box">💡 Increase planned exercise or study goals to trigger positive multipliers.</div>', unsafe_allow_html=True)

        sim_recommendations = []
        if plan_exercise < 30:
            test_dict = scenario_dict.copy()
            test_dict['Do Exercise'] = plan_exercise + 15
            test_df = pd.DataFrame([test_dict]).reindex(columns=feature_cols, fill_value=0)
            bump_pred = rf.predict(test_df)[0]
            diff = bump_pred - simulated_pred_mood
            if diff > 0.1:
                sim_recommendations.append(f"💡 **Exercise Opportunity**: Increasing exercise from {plan_exercise}m to {plan_exercise+15}m elevates predicted evening mood by **+{diff:.2f} points**.")

        if plan_english < 20:
            test_dict = scenario_dict.copy()
            test_dict['English'] = plan_english + 15
            test_df = pd.DataFrame([test_dict]).reindex(columns=feature_cols, fill_value=0)
            bump_eng = rf.predict(test_df)[0]
            diff_eng = bump_eng - simulated_pred_mood
            if diff_eng > 0.1:
                sim_recommendations.append(f"📚 **English Leverage**: Adding 15 mins of English study boosts predicted evening satisfaction by **+{diff_eng:.2f} points**.")

        if plan_exercise < dow_ex_avg:
            sim_recommendations.append(f"📅 **{day_name} Benchmark**: Today's planned exercise ({plan_exercise}m) is below your historical {day_name} average ({dow_ex_avg:.0f}m).")

        if plan_m_mood <= 6.0:
            sim_recommendations.append("🎯 **Morning Deficit Offset**: Morning mood is low. Prioritize reading or English today to prevent an evening mood drop.")

        with sim_r:
            st.subheader("💡 Reactive Simulation Recommendations")
            if sim_recommendations:
                for r in sim_recommendations:
                    st.markdown(f'<div class="recommend-box">{r}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">🎯 Planned inputs are already in the top statistical tier!</div>', unsafe_allow_html=True)

        # Plan Summary Export
        st.markdown("---")
        st.subheader("📋 Export Today's Optimized Action Checklist")
        checklist_text = f"""# Daily Habit Action Plan - {selected_date.strftime('%Y-%m-%d')} ({day_name})
Predicted Night Mood: {simulated_pred_mood:.1f} / 10.0

## Sleep & Recovery:
- [ ] Planned Bedtime: {plan_bed_time.strftime('%H:%M')}
- [ ] Planned Wake Time: {plan_wake_time.strftime('%H:%M')} (Total: {calc_sleep_dur:.1f} Hours)

## Core Action Targets:
- [ ] Exercise: {plan_exercise} Minutes
- [ ] English Study: {plan_english} Minutes
- [ ] Book Reading: {plan_book} Pages
- [ ] Quran Reading: {plan_quran} Pages
- [ ] Lunch Quality Target: {plan_lunch}/10
- [ ] Dinner Quality Target: {plan_dinner}/10
- [ ] Water Intake: {plan_water} Glasses
- [ ] Screen Time Limit: <={plan_screen} Minutes

## Morning / Night Anchors:
- [ ] Morning Prayer Completed
- [ ] Night Prayer Completed
- [ ] Zekr Completed
- [ ] No Soda Maintained
- [ ] No J Maintained
"""
        st.download_button(
            label="📥 Download Daily Action Checklist (.md)",
            data=checklist_text,
            file_name=f"Habit_Plan_{selected_date.strftime('%Y-%m-%d')}.md",
            mime="text/markdown"
        )

    # =========================================================================
    # TAB 3: TARGET GOAL SEEKER (INVERSE OPTIMIZATION)
    # =========================================================================
    with tab_opt:
        st.markdown("## 🎯 Target Goal Seeker (One-Click Habit Optimizer)")
        st.caption("Select your desired evening mood score (1-10). The optimizer will determine the minimum viable habit combination to achieve it.")

        target_score = st.slider("Desired Evening Mood (1-10)", 7.0, 10.0, 8.8, 0.1)

        if st.button("🚀 Find Optimal Habit Plan"):
            best_plan = None
            min_effort = 999999
            
            ex_options = [0, 15, 30, 45, 60]
            eng_options = [0, 15, 30, 45]
            dinner_options = [7.0, 8.0, 9.0, 10.0]
            book_options = [5, 10, 15]

            for ex_val in ex_options:
                for eng_val in eng_options:
                    for din_val in dinner_options:
                        for bk_val in book_options:
                            test_dict = yesterday_row[feature_cols].to_dict()
                            test_dict.update({
                                'Mood In Morning': base_m_mood,
                                'Physical Health': base_health,
                                'Do Exercise': ex_val,
                                'English': eng_val,
                                'Book': bk_val,
                                'Dinner Quality': din_val,
                                'Lunch Quality': 8.0,
                                'Sleep_Duration_Hours': 7.5,
                                'Sleep early_done': 1,
                                'Wake up early_done': 1,
                                'No Soda_done': 1,
                                'DayOfWeek': sim_dow_num
                            })
                            t_df = pd.DataFrame([test_dict]).reindex(columns=feature_cols, fill_value=0)
                            p_mood = rf.predict(t_df)[0]
                            if p_mood >= target_score - 0.15:
                                effort = ex_val + eng_val + bk_val * 2
                                if effort < min_effort:
                                    min_effort = effort
                                    best_plan = {
                                        'Exercise': ex_val,
                                        'English': eng_val,
                                        'Book': bk_val,
                                        'Dinner': din_val,
                                        'Predicted Mood': p_mood
                                    }

            if best_plan:
                st.success(f"🎯 **Optimal Minimum Plan Found!** Expected Evening Mood: **{best_plan['Predicted Mood']:.1f} / 10.0**")
                o1, o2, o3, o4 = st.columns(4)
                o1.metric("Exercise Required", f"{best_plan['Exercise']} min")
                o2.metric("English Study Required", f"{best_plan['English']} min")
                o3.metric("Book Reading Required", f"{best_plan['Book']} pages")
                o4.metric("Dinner Quality Target", f"{best_plan['Dinner']:.1f}/10")
            else:
                st.warning("Could not reach that target with standard routine. Try ensuring optimal sleep duration and morning baseline first.")

    # =========================================================================
    # TAB 4: DEEP ANALYTICS & BURNOUT ENGINE
    # =========================================================================
    with tab_analytics:
        st.markdown("## 📈 Deep Analytics, Burnout Engine & Historical Trends")

        c1, c2, c3 = st.columns(3)
        c1.metric("Cognitive vs Physical Load Ratio", f"{burnout_ratio:.2f}", delta="Optimal < 2.0" if burnout_ratio <= 2.0 else "Burnout Risk > 2.5", delta_color="inverse")
        c2.metric("14-Day Average Sleep Duration", f"{base_sleep_hours:.1f} Hours", delta="Healthy >= 7.5h" if base_sleep_hours >= 7.5 else "Sleep Deficit", delta_color="normal")
        c3.metric(f"Historical {day_name}s Sampled", f"{len(same_weekday_df)} days")

        st.markdown("---")
        g1, g2 = st.columns(2)

        with g1:
            st.subheader(f"Historical {day_name} Performance Distribution (1-10 Scale)")
            fig_dow, ax_dow = plt.subplots(figsize=(8, 3.5))
            ax_dow.plot(same_weekday_df.index, same_weekday_df['Mood At Night'], marker='o', color='#9C27B0', label=f'Past {day_name} Moods')
            ax_dow.axhline(dow_historical_avg, color='#FF9800', linestyle='--', label=f'{day_name} Mean ({dow_historical_avg:.1f})')
            ax_dow.set_ylabel("Mood (1-10)")
            ax_dow.grid(True, linestyle='--', alpha=0.5)
            ax_dow.legend()
            st.pyplot(fig_dow)

        with g2:
            st.subheader("Top Model Feature Leverage Weights")
            importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
            fig_imp, ax_imp = plt.subplots(figsize=(8, 3.5))
            importances.plot(kind='barh', ax=ax_imp, color='#2196F3')
            ax_imp.invert_yaxis()
            ax_imp.set_xlabel("Predictive Weight")
            st.pyplot(fig_imp)

        st.subheader("60-Day Rolling Momentum Trends (Native 1-10 Scale)")
        fig, ax = plt.subplots(figsize=(10, 3))
        recent_60 = valid_history.tail(60)
        ax.plot(recent_60.index, recent_60['Mood At Night'], label='Daily Night Mood', alpha=0.35, color='gray', linestyle='--')
        ax.plot(recent_60.index, recent_60['Mood At Night_roll7'], label='7-Day Rolling Trend', color='#4CAF50', linewidth=2.5)
        ax.plot(recent_60.index, recent_60['Mood At Night_roll14'], label='14-Day Rolling Momentum', color='#FF9800', linewidth=2.5)
        ax.set_ylabel("Score (1-10)")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)

else:
    st.info("👆 Upload your habit tracking CSV file in the sidebar to launch the system.")