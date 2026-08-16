# Personal Habit Intelligence Engine & Life OS

An offline, machine-learning-powered personal analytics engine and interactive decision-support system built with **Python**, **Scikit-Learn**, and **Streamlit**.

[cite_start]This application ingests longitudinal habit tracking CSV exports (spanning 800+ days) [cite: 711][cite_start], parses multi-scale metrics (1–5 / 1–10) [cite: 874, 1143][cite_start], computes multi-window rolling momentums (3-day, 7-day, 14-day) [cite: 1144][cite_start], calculates sleep duration and circadian alignment [cite: 599, 620][cite_start], and forecasts end-of-day well-being[cite: 713]. [cite_start]It acts as a smart life assistant by flagging behavioral risks, calculating streak multipliers [cite: 886][cite_start], and prescribing real-time habit optimizations[cite: 933, 944].

---

## Key Features

- [cite_start]** Automated Passive Recommendations**: Automatically analyzes historical same-weekday patterns (e.g., past Wednesdays) and 14-day rolling baselines without requiring manual daily input[cite: 407, 440, 441].
- [cite_start]** Interactive Scenario Simulator**: Adjust planned daily parameters (exercise duration, study targets, meal quality, bedtime) to forecast evening mood scores in real time[cite: 958, 1075].
- [cite_start]** Precision Sleep & Circadian Duration**: Computes continuous sleep hours across midnight rollover boundaries and flags mismatched sleep patterns[cite: 599, 917].
- [cite_start]** Target Goal Seeker (Inverse Optimization)**: Specify a target evening score (e.g., 9.0/10) to compute the minimum viable habit combination required to reach it[cite: 523, 524].
- [cite_start]** Cognitive vs. Physical Load Balance (Burnout Index)**: Tracks mental load (reading and study) against physical recovery to detect impending burnout[cite: 526, 527].
- [cite_start]** Feature Leverage & SHAP Importance**: Quantifies which habits provide the highest marginal return on investment for your daily mood and health[cite: 719, 721].
- [cite_start]** One-Click Markdown Checklist Export**: Download a structured daily action plan (`.md`) formatted for Obsidian, Notion, or local to-do managers[cite: 529].

---

## Tech Stack & Dependencies

- **Language**: Python 3.9+
- **Frontend / Dashboard**: Streamlit
- [cite_start]**Machine Learning**: Scikit-Learn (Random Forest Regressor) [cite: 834]
- [cite_start]**Data Engineering**: Pandas, NumPy (Vectorized lag, rolling window, and time conversions) [cite: 740, 741, 1144]
- **Visualizations**: Matplotlib, Seaborn

---

##  Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/habit-intelligence-engine.git](https://github.com/your-username/habit-intelligence-engine.git)
cd habit-intelligence-engine
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run "web app.py"
```

[cite_start]Open `http://localhost:8501` in your browser and upload your habit tracking CSV file[cite: 1048, 1049].

---

##  Repository Structure

```text
habit-intelligence-engine/
│
├── app.py                     # Streamlit application UI and ML pipeline
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files to exclude from version control
├── LICENSE                    # Open-source license (e.g., MIT)
└── README.md                  # Project documentation
```

---

## 🛡️ Privacy & Local Execution

[cite_start]This dashboard operates **100% locally and offline**[cite: 948, 993]. [cite_start]No personal activity, habit data, or health metrics are sent to third-party cloud servers[cite: 948, 993].

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.