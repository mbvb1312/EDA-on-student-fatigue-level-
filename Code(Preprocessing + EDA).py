"""
IDHV Assignment: Data Preprocessing & Exploratory Data Analysis
Title: Impact of Timetable Fragmentation on Student Productivity & Fatigue
Author: Vignesh Balamurugan M.B (S20230030422)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# ============================================================
# PHASE 2: DATA ORGANIZATION & PREPROCESSING
# ============================================================
print("=" * 60)
print("PHASE 2: DATA ORGANIZATION & PREPROCESSING")
print("=" * 60)

# Load raw data
df = pd.read_csv('raw_data.csv')

# Shorten column names for easier handling
col_map = {
    'Timestamp': 'Timestamp',
    'Name': 'Name',
    'Institute Email ID (must end with @iiits.in)': 'Email',
    'Year of Study': 'Year',
    'Gender': 'Gender',
    'Branch': 'Branch',
    'Total number of classes on your most 2 busiest days of the week combined (Enter a number)': 'Total_Classes',
    'How many "Single Period Gaps" (exactly 1 hour free between classes) do you have across these 2 days?': 'Single_Gaps',
    'How many "Double Period Gaps" (2 hours free between classes) do you have across these 2 days?': 'Double_Gaps',
    'Do you have any "Long Gaps" (3+ hours between classes) on these 2 days?': 'Has_Long_Gap',
    'If yes, how many long gaps (3+ hours)?': 'Long_Gap_Count',
    'Where do you primarily spend your short gaps (1-2 hours)?': 'Short_Gap_Location',
    'Where do you primarily spend your longer gaps (3+ hours)?': 'Long_Gap_Location',
    'How many times do you walk back to the hostel during gaps on these 2 busiest days?': 'Hostel_Trips',
    'What do you typically do during gap hours? (Select all that apply)': 'Gap_Activities',
    'On a scale of 1-10, how productive are your gap hours? (1 = Not at all, 10 = Very productive)': 'Productivity',
    'How many hours across these 2 busiest days do you feel were "wasted" (neither restful nor productive)? Enter a number': 'Waste_Hours',
    'Rate your physical fatigue level at 6 PM on your busiest day (1 = Energetic, 10 = Completely drained)': 'Fatigue',
    'Do you feel the gaps in your timetable significantly decrease your overall academic focus?': 'Academic_Impact',
    'Would you prefer a more compact/continuous class schedule?': 'Schedule_Preference',
}
df.rename(columns=col_map, inplace=True)

print(f"\n--- Raw Dataset Info ---")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nData Types:\n{df.dtypes}")
print(f"\n--- Missing Values (Before Cleaning) ---")
missing = df.isnull().sum()
print(missing[missing > 0])

# --------------------------------------------------------
# STEP 1: Handle text-in-numeric fields
# --------------------------------------------------------
print("\n--- Step 1: Cleaning Text in Numeric Fields ---")

# Total_Classes: fix "7 classes", "eight"
print(f"  Before: Total_Classes non-numeric entries:")
print(f"    Row 10: {df.loc[10, 'Total_Classes']}")
print(f"    Row 35: {df.loc[35, 'Total_Classes']}")

text_to_num = {'eight': 8, 'seven': 7, 'six': 6, 'five': 5}
def clean_total_classes(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip().lower()
    if val_str in text_to_num:
        return text_to_num[val_str]
    # Extract number from strings like "7 classes"
    import re
    nums = re.findall(r'\d+', val_str)
    if nums:
        return int(nums[0])
    try:
        return float(val_str)
    except:
        return np.nan

df['Total_Classes'] = df['Total_Classes'].apply(clean_total_classes)
print(f"  After cleaning: {df['Total_Classes'].dtype}")

# Waste_Hours: fix "around 3"
print(f"\n  Before: Waste_Hours non-numeric entry:")
print(f"    Row 20: {df.loc[20, 'Waste_Hours']}")

def clean_waste_hours(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip().lower()
    import re
    nums = re.findall(r'[\d.]+', val_str)
    if nums:
        return float(nums[0])
    try:
        return float(val_str)
    except:
        return np.nan

df['Waste_Hours'] = df['Waste_Hours'].apply(clean_waste_hours)
print(f"  After cleaning: {df['Waste_Hours'].dtype}")

# --------------------------------------------------------
# STEP 2: Handle Missing Values
# --------------------------------------------------------
print("\n--- Step 2: Handling Missing Values ---")

# Convert numeric columns
numeric_cols = ['Total_Classes', 'Single_Gaps', 'Double_Gaps', 'Hostel_Trips',
                'Productivity', 'Waste_Hours', 'Fatigue']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

missing_after = df[numeric_cols].isnull().sum()
print(f"Missing values in numeric columns:\n{missing_after[missing_after > 0]}")

# Impute with MEDIAN (robust to outliers)
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        print(f"  Imputing '{col}' missing values with median: {median_val}")
        df[col].fillna(median_val, inplace=True)

# --------------------------------------------------------
# STEP 3: Handle Outliers (IQR Method)
# --------------------------------------------------------
print("\n--- Step 3: Outlier Detection & Treatment ---")

print(f"  Hostel_Trips max before: {df['Hostel_Trips'].max()}")
Q1 = df['Hostel_Trips'].quantile(0.25)
Q3 = df['Hostel_Trips'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
outliers = df[df['Hostel_Trips'] > upper_bound]
print(f"  IQR: {IQR}, Upper bound: {upper_bound}")
print(f"  Outliers found: {len(outliers)} (values > {upper_bound})")
df.loc[df['Hostel_Trips'] > upper_bound, 'Hostel_Trips'] = df['Hostel_Trips'].median()
print(f"  Hostel_Trips max after capping: {df['Hostel_Trips'].max()}")

# --------------------------------------------------------
# STEP 4: Standardize Categorical Values
# --------------------------------------------------------
print("\n--- Step 4: Standardizing Categorical Values ---")

# Standardize Has_Long_Gap
df['Has_Long_Gap'] = df['Has_Long_Gap'].str.strip().str.capitalize()
df['Has_Long_Gap'] = df['Has_Long_Gap'].replace({'No': 'No', 'Yes': 'Yes'})
print(f"  Has_Long_Gap values: {df['Has_Long_Gap'].unique()}")

# Standardize Schedule_Preference
df['Schedule_Preference'] = df['Schedule_Preference'].str.strip().str.title()
df['Schedule_Preference'] = df['Schedule_Preference'].replace({
    'Strongly Agree': 'Strongly Agree',
    'Agree': 'Agree',
    'Neutral': 'Neutral',
    'Disagree': 'Disagree',
    'Strongly Disagree': 'Strongly Disagree'
})
print(f"  Schedule_Preference values: {df['Schedule_Preference'].unique()}")

# Standardize Academic_Impact
df['Academic_Impact'] = df['Academic_Impact'].str.strip().str.capitalize()
print(f"  Academic_Impact values: {df['Academic_Impact'].unique()}")

# --------------------------------------------------------
# STEP 5: Normalization (Min-Max)
# --------------------------------------------------------
print("\n--- Step 5: Min-Max Normalization ---")

norm_cols = ['Total_Classes', 'Single_Gaps', 'Double_Gaps', 'Hostel_Trips',
             'Productivity', 'Waste_Hours', 'Fatigue']

for col in norm_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    df[f'Norm_{col}'] = (df[col] - col_min) / (col_max - col_min) if col_max != col_min else 0
    print(f"  {col}: min={col_min}, max={col_max} -> Normalized to [0, 1]")

# --------------------------------------------------------
# STEP 6: Derived Variables
# --------------------------------------------------------
print("\n--- Step 6: Creating Derived Variables ---")

# Total Gap Score: weighted combination
df['Total_Gap_Score'] = df['Single_Gaps'] * 1 + df['Double_Gaps'] * 2 + df['Long_Gap_Count'].fillna(0) * 3
print(f"  Total_Gap_Score: mean={df['Total_Gap_Score'].mean():.2f}, std={df['Total_Gap_Score'].std():.2f}")

# Productivity Efficiency Ratio
df['Efficiency_Ratio'] = df['Productivity'] / (df['Total_Gap_Score'] + 1)
print(f"  Efficiency_Ratio: mean={df['Efficiency_Ratio'].mean():.2f}")

# Fatigue-to-Productivity Ratio
df['Fatigue_Prod_Ratio'] = df['Fatigue'] / (df['Productivity'] + 1)
print(f"  Fatigue_Prod_Ratio: mean={df['Fatigue_Prod_Ratio'].mean():.2f}")

# --------------------------------------------------------
# SAVE CLEANED DATA
# --------------------------------------------------------
df.to_csv('cleaned_data.csv', index=False)
print(f"\n{'='*60}")
print(f"cleaned_data.csv saved with {len(df)} rows and {len(df.columns)} columns")
print(f"{'='*60}")

# ============================================================
# PHASE 3: DESCRIPTIVE STATISTICS & EDA
# ============================================================
print(f"\n\n{'='*60}")
print("PHASE 3: DESCRIPTIVE STATISTICS & EDA")
print(f"{'='*60}")

# --------------------------------------------------------
# 3.1 DESCRIPTIVE STATISTICS
# --------------------------------------------------------
print("\n--- 3.1 Descriptive Statistics ---")
desc_cols = ['Total_Classes', 'Single_Gaps', 'Double_Gaps', 'Hostel_Trips',
             'Productivity', 'Waste_Hours', 'Fatigue']
desc_stats = df[desc_cols].describe()
print(desc_stats.round(2))

# Add skewness and kurtosis
print("\nSkewness:")
for col in desc_cols:
    print(f"  {col}: {df[col].skew():.3f}")

print("\nKurtosis:")
for col in desc_cols:
    print(f"  {col}: {df[col].kurtosis():.3f}")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n--- Generating Visualizations ---")

# --------------------------------------------------------
# PLOT 1: Distribution of Key Variables (Histograms)
# --------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Distribution of Key Numerical Variables', fontsize=16, fontweight='bold')

plot_cols = ['Total_Classes', 'Single_Gaps', 'Double_Gaps', 'Hostel_Trips', 'Productivity', 'Fatigue']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

for ax, col, color in zip(axes.flatten(), plot_cols, colors):
    ax.hist(df[col], bins=8, color=color, alpha=0.7, edgecolor='black')
    ax.set_title(col.replace('_', ' '), fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.1f}')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plot1_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot1_distributions.png")

# --------------------------------------------------------
# PLOT 2: Correlation Heatmap
# --------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))
corr_cols = ['Total_Classes', 'Single_Gaps', 'Double_Gaps', 'Hostel_Trips',
             'Productivity', 'Waste_Hours', 'Fatigue', 'Total_Gap_Score']
corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, ax=ax,
            cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Heatmap: Gaps vs Productivity vs Fatigue', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot2_correlation_heatmap.png")

# --------------------------------------------------------
# PLOT 3: Box Plot - Fatigue by Short Gap Location
# --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Impact of Gap Location on Fatigue & Productivity', fontsize=14, fontweight='bold')

sns.boxplot(x='Short_Gap_Location', y='Fatigue', data=df, ax=axes[0], palette='Set2')
axes[0].set_title('Fatigue Level by Short Gap Location')
axes[0].set_xlabel('Primary Location (1-2 hr gaps)')
axes[0].set_ylabel('Fatigue Level (1-10)')
axes[0].tick_params(axis='x', rotation=25)

sns.boxplot(x='Short_Gap_Location', y='Productivity', data=df, ax=axes[1], palette='Set3')
axes[1].set_title('Productivity by Short Gap Location')
axes[1].set_xlabel('Primary Location (1-2 hr gaps)')
axes[1].set_ylabel('Productivity Score (1-10)')
axes[1].tick_params(axis='x', rotation=25)

plt.tight_layout()
plt.savefig('plot3_location_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot3_location_boxplots.png")

# --------------------------------------------------------
# PLOT 4: Scatter + Regression - Single Gaps vs Fatigue
# --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Regression Analysis: Gaps → Fatigue & Productivity', fontsize=14, fontweight='bold')

sns.regplot(x='Single_Gaps', y='Fatigue', data=df, ax=axes[0],
            scatter_kws={'alpha': 0.6, 's': 50}, color='#e74c3c')
r1, p1 = stats.pearsonr(df['Single_Gaps'], df['Fatigue'])
axes[0].set_title(f'Single Gaps vs Fatigue (r={r1:.3f}, p={p1:.4f})')
axes[0].set_xlabel('Number of Single Period Gaps')
axes[0].set_ylabel('Fatigue Level')

sns.regplot(x='Total_Gap_Score', y='Productivity', data=df, ax=axes[1],
            scatter_kws={'alpha': 0.6, 's': 50}, color='#3498db')
r2, p2 = stats.pearsonr(df['Total_Gap_Score'], df['Productivity'])
axes[1].set_title(f'Total Gap Score vs Productivity (r={r2:.3f}, p={p2:.4f})')
axes[1].set_xlabel('Total Gap Score (weighted)')
axes[1].set_ylabel('Productivity Score')

plt.tight_layout()
plt.savefig('plot4_regression_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot4_regression_analysis.png")

# --------------------------------------------------------
# PLOT 5: Gender-wise analysis
# --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Gender-wise Analysis', fontsize=14, fontweight='bold')

# Gender vs Location (stacked bar)
gender_loc = pd.crosstab(df['Gender'], df['Short_Gap_Location'], normalize='index') * 100
gender_loc.plot(kind='bar', stacked=True, ax=axes[0], colormap='Set2')
axes[0].set_title('Short Gap Location by Gender (%)')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Percentage')
axes[0].legend(title='Location', bbox_to_anchor=(1.0, 1.0), fontsize=8)
axes[0].tick_params(axis='x', rotation=0)

# Gender vs Fatigue
sns.boxplot(x='Gender', y='Fatigue', data=df, ax=axes[1], palette='pastel')
axes[1].set_title('Fatigue Level by Gender')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Fatigue Level')

plt.tight_layout()
plt.savefig('plot5_gender_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot5_gender_analysis.png")

# --------------------------------------------------------
# PLOT 6: Branch-wise analysis
# --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Branch-wise Analysis', fontsize=14, fontweight='bold')

sns.barplot(x='Branch', y='Fatigue', data=df, ax=axes[0], palette='viridis', ci='sd')
axes[0].set_title('Average Fatigue by Branch')
axes[0].set_xlabel('Branch')
axes[0].set_ylabel('Fatigue Level')

sns.barplot(x='Branch', y='Waste_Hours', data=df, ax=axes[1], palette='magma', ci='sd')
axes[1].set_title('Average Waste Hours by Branch')
axes[1].set_xlabel('Branch')
axes[1].set_ylabel('Waste Hours')

plt.tight_layout()
plt.savefig('plot6_branch_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot6_branch_analysis.png")

# --------------------------------------------------------
# PLOT 7: Hostel Trips vs Fatigue
# --------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x='Hostel_Trips', y='Fatigue', data=df, ax=ax,
            scatter_kws={'alpha': 0.6, 's': 60, 'color': '#e67e22'},
            line_kws={'color': '#c0392b'})
r3, p3 = stats.pearsonr(df['Hostel_Trips'], df['Fatigue'])
ax.set_title(f'Hostel Trip Penalty: Trips vs Fatigue (r={r3:.3f}, p={p3:.4f})',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Hostel Trips (during gaps)')
ax.set_ylabel('Fatigue Level at 6 PM')
plt.tight_layout()
plt.savefig('plot7_hostel_trip_penalty.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot7_hostel_trip_penalty.png")

# --------------------------------------------------------
# PLOT 8: Schedule Preference Pie Chart
# --------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Student Opinions on Timetable', fontsize=14, fontweight='bold')

pref_counts = df['Schedule_Preference'].value_counts()
colors_pie = ['#2ecc71', '#27ae60', '#f1c40f', '#e74c3c', '#c0392b']
axes[0].pie(pref_counts, labels=pref_counts.index, autopct='%1.1f%%',
            colors=colors_pie[:len(pref_counts)], startangle=90)
axes[0].set_title('Prefer Compact Schedule?')

impact_counts = df['Academic_Impact'].value_counts()
colors_pie2 = ['#e74c3c', '#f39c12', '#2ecc71']
axes[1].pie(impact_counts, labels=impact_counts.index, autopct='%1.1f%%',
            colors=colors_pie2[:len(impact_counts)], startangle=90)
axes[1].set_title('Gaps Decrease Academic Focus?')

plt.tight_layout()
plt.savefig('plot8_opinions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot8_opinions.png")

# --------------------------------------------------------
# PLOT 9: Waste Hours Distribution by Gap Type
# --------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
df_melt = df[['Single_Gaps', 'Double_Gaps', 'Waste_Hours']].copy()
sns.scatterplot(x='Single_Gaps', y='Waste_Hours', data=df, label='vs Single Gaps',
                s=80, alpha=0.6, color='#3498db', ax=ax)
sns.scatterplot(x='Double_Gaps', y='Waste_Hours', data=df, label='vs Double Gaps',
                s=80, alpha=0.6, color='#e74c3c', marker='s', ax=ax)
ax.set_title('Waste Hours vs Gap Types', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Gaps')
ax.set_ylabel('Waste Hours')
ax.legend()
plt.tight_layout()
plt.savefig('plot9_waste_vs_gaps.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: plot9_waste_vs_gaps.png")

# --------------------------------------------------------
# PRINT KEY FINDINGS
# --------------------------------------------------------
print(f"\n{'='*60}")
print("KEY FINDINGS SUMMARY")
print(f"{'='*60}")

print(f"\n1. CORRELATION RESULTS:")
print(f"   • Single Gaps ↔ Fatigue: r = {r1:.3f} (p = {p1:.4f}) {'*** Significant' if p1 < 0.05 else ''}")
print(f"   • Total Gap Score ↔ Productivity: r = {r2:.3f} (p = {p2:.4f}) {'*** Significant' if p2 < 0.05 else ''}")
print(f"   • Hostel Trips ↔ Fatigue: r = {r3:.3f} (p = {p3:.4f}) {'*** Significant' if p3 < 0.05 else ''}")

print(f"\n2. LOCATION ANALYSIS:")
loc_fatigue = df.groupby('Short_Gap_Location')['Fatigue'].mean()
for loc, fat in loc_fatigue.items():
    print(f"   • {loc}: avg fatigue = {fat:.2f}")

print(f"\n3. GENDER INSIGHTS:")
gender_stats = df.groupby('Gender')[['Fatigue', 'Productivity', 'Hostel_Trips']].mean()
print(gender_stats.round(2))

print(f"\n4. OVERALL:")
print(f"   • Average fatigue: {df['Fatigue'].mean():.2f}")
print(f"   • Average productivity: {df['Productivity'].mean():.2f}")
print(f"   • Average waste hours: {df['Waste_Hours'].mean():.2f}")
print(f"   • Students preferring compact schedule: {(df['Schedule_Preference'].isin(['Strongly Agree', 'Agree'])).sum()}/{len(df)} ({(df['Schedule_Preference'].isin(['Strongly Agree', 'Agree'])).sum()/len(df)*100:.1f}%)")

print(f"\n{'='*60}")
print("All files generated successfully!")
print(f"{'='*60}")
print(f"\nFiles in project folder:")
print(f"  1. raw_data.csv          - Raw collected data (60 responses)")
print(f"  2. cleaned_data.csv      - Cleaned & preprocessed data")
print(f"  3. plot1_distributions.png     - Distribution histograms")
print(f"  4. plot2_correlation_heatmap.png - Correlation heatmap")
print(f"  5. plot3_location_boxplots.png  - Location vs Fatigue/Productivity")
print(f"  6. plot4_regression_analysis.png - Regression plots")
print(f"  7. plot5_gender_analysis.png    - Gender-wise analysis")
print(f"  8. plot6_branch_analysis.png    - Branch-wise analysis")
print(f"  9. plot7_hostel_trip_penalty.png - Hostel trip penalty")
print(f" 10. plot8_opinions.png           - Student opinions pie charts")
print(f" 11. plot9_waste_vs_gaps.png      - Waste hours vs gap types")
