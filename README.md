# The Exhaustion Paradox: How Fragmented Timetables Drain Student Energy

An exploratory data analysis investigating how gap-heavy, fragmented class timetables contribute to physical fatigue and reduced productivity among university students.

---

## Problem Statement

Does "free time" between classes actually leave students feeling rested — or does it quietly drain their energy? This study examines how short, isolated gaps (1–2 hours) embedded in a student's daily schedule drive end-of-day exhaustion, and whether the common coping strategy of walking back to the hostel during those gaps makes the problem worse.

## Key Findings

| Metric | Value |
|--------|-------|
| Mean Fatigue Score (out of 10) | ~7.0 |
| Mean Productivity During Gaps (out of 10) | 3.3 |
| Correlation: Single Gaps → Fatigue | **0.55** (p ≈ 0) |
| Correlation: Single Gaps → Wasted Hours | **0.60** |
| Fatigue Increase per Extra Hostel Trip | +0.4 points |
| Students Preferring Compact Schedules | **67%** |

### Highlights

- **Each additional 1-hour gap** in a student's day is associated with roughly a **1-point increase** in fatigue (on a 10-point scale).
- Students who walked back to hostels during gaps reported the **highest fatigue** (8.0/10) — the walk costs more than the rest provides.
- Students who stayed in **air-conditioned classrooms** reported the **lowest fatigue** (5.9/10).
- Two-hour gaps did **not** lead to better time utilisation than one-hour gaps — they just doubled the opportunity to waste time.

---

## Dataset

- **Source**: Primary data collected via a Google Forms survey (19 questions across 4 sections)
- **Sample**: 60 valid responses from 3rd-year students at IIIT Sri City (March 2026)
- **Demographics**: CSE (39), AI&DS (12), ECE (9) | Male (48), Female (12)
- **Scope**: Each response covers the student's **2 busiest days** of the week

### Variables Collected

| Category | Variables |
|----------|-----------|
| Demographics | Branch, Gender, Year of Study |
| Schedule Load | Total classes across 2 busiest days |
| Fragmentation | Single gaps (1 hr), Double gaps (2 hr), Long gaps (3+ hr) |
| Coping Strategy | Gap location (Library / AC Classroom / Hostel / Canteen), Hostel trips |
| Outcomes | Productivity (1–10), Fatigue (1–10), Wasted hours |
| Opinions | Academic impact of gaps, Schedule preference |

---

## Data Preprocessing

The raw survey data required several cleaning steps before analysis:

1. **Text-to-Numeric Conversion** — Regex extraction + dictionary mapping for entries like "eight" → 8
2. **Standardisation** — Capitalisation normalisation on categorical columns to prevent group-by errors
3. **Missing Value Imputation** — Median imputation for 5 blank cells (median chosen over mean to resist outlier influence)
4. **Outlier Treatment** — IQR-based detection flagged an impossible hostel trip count of 15; replaced with median (2)
5. **Min-Max Scaling** — 7 numeric columns normalised to [0, 1] for the correlation heatmap
6. **Feature Engineering** — 3 derived metrics:
   - `Total_Gap_Score` — weighted composite of all gap types (1× single, 2× double, 3× long)
   - `Efficiency_Ratio` — Productivity / Total Gap Score
   - `Fatigue_Prod_Ratio` — Fatigue / Productivity (the "cost of the day")

---

## Exploratory Data Analysis

Nine visualisations were generated to investigate different angles of the fatigue problem:

| Plot | Description |
|------|-------------|
| `plot1_distributions.png` | Histograms of key numerical variables (classes, gaps, fatigue, productivity) |
| `plot2_correlation_heatmap.png` | Pearson correlation heatmap across all numeric features |
| `plot3_location_boxplots.png` | Fatigue & productivity boxplots grouped by gap-period location |
| `plot4_regression_analysis.png` | Scatter + regression — Single Gaps vs Fatigue, Total Gaps vs Productivity |
| `plot5_gender_analysis.png` | Gender-wise location preferences and fatigue comparison |
| `plot6_branch_analysis.png` | Branch-wise (CSE/ECE/AI&DS) fatigue and productivity |
| `plot7_hostel_trip_penalty.png` | Hostel round-trips vs fatigue — the "trip penalty" |
| `plot8_opinions.png` | Student preferences for compact vs fragmented schedules |
| `plot9_waste_vs_gaps.png` | Wasted hours compared across single and double gap counts |

---

## Project Structure

```
Student_Timetable_fatigue_analysis/
│
├── Dataset/
│   ├── raw_data.csv              # Original survey export (60 responses)
│   └── cleaned_data.csv          # Preprocessed dataset with engineered features
│
├── EDA/
│   ├── plot1_distributions.png
│   ├── plot2_correlation_heatmap.png
│   ├── plot3_location_boxplots.png
│   ├── plot4_regression_analysis.png
│   ├── plot5_gender_analysis.png
│   ├── plot6_branch_analysis.png
│   ├── plot7_hostel_trip_penalty.png
│   ├── plot8_opinions.png
│   └── plot9_waste_vs_gaps.png
│
├── Code(Preprocessing + EDA).py  # Full pipeline: cleaning → analysis → visualisation
├── Questionaries.pdf             # Screenshots of the Google Form survey
├── Report.pdf                    # Detailed written analysis and interpretation
└── README.md
```

---

## Tech Stack

- **Python 3** — Core language
- **Pandas** — Data loading, cleaning, transformation, and aggregation
- **NumPy** — Numerical operations
- **Matplotlib** — Base plotting and figure composition
- **Seaborn** — Statistical visualisations (heatmaps, regression plots, boxplots)
- **SciPy** — Statistical tests (Pearson correlation, p-values)

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scipy

# Run the full pipeline (preprocessing + EDA)
cd Student_Timetable_fatigue_analysis/Dataset
python "../Code(Preprocessing + EDA).py"
```

All 9 plots will be saved to the `EDA/` directory, and a `cleaned_data.csv` will be generated in `Dataset/`.

---

## Limitations

- **Self-reported data** — Fatigue and productivity scores are subjective (one person's 8 may be another's 5)
- **Single cohort** — Only 3rd-year students; no comparison with freshmen or 2nd-years
- **Small subgroups** — ECE (n=9) and AI&DS (n=12) subsamples are too small for robust branch-level conclusions
- **Uncontrolled variables** — Temperature, AC availability, sleep habits, and caffeine intake were not measured

---

## Author

**Vignesh Balamurugan M.B**
B.Tech AI & Data Science, IIIT Sri City

---

*This project was completed as part of the Introduction to Data Handling and Visualization (IDHV) course.*
