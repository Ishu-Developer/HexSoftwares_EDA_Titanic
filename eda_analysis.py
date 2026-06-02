# ======================================
# Task 1: EDA on Titanic Dataset
# @HexSoftwares Internship — Ishu
# ======================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. Load Dataset 
df = pd.read_csv('titanic.csv')
print("=== Dataset Shape ===")
print(df.shape)           #rows&columns 
print("=== First 5 Rows ===")
print(df.head())
print("=== Column Names ===")
print(df.columns.tolist())
print("=== Data Types ===")
print(df.dtypes)

# 2. Missing Values Check
print("=== Missing Values Count ===")
print(df.isnull().sum())
print("=== Missing Values Percentage ===")
missing_percent = (df.isnull().sum() / len(df)) * 100
print(missing_percent[missing_percent > 0].round(2))
# Handle Missing Values
# Age column: mean -> average age
df['Age'] = df['Age'].fillna(df['Age'].median())
# Embarked column: by most common value
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# Cabin column: more missing, drop them
df = df.drop(columns=['Cabin'])
print("=== After Cleaning — Missing Values ===")
print(df.isnull().sum())
print("Data cleaning complete! ✅")

# 3. Statistical Analysis 

print("=== Basic Statistics ===")
print(df.describe())  # Mean, std, min, max
print("=== Survival Count ===")
print(df['Survived'].value_counts()) # 0 = Not Survived, 1 = Survived
print("=== Survival Rate (%) ===")
survival_rate = df['Survived'].mean() * 100
print(f"Overall Survival Rate: {survival_rate:.2f}%")
print("=== Survival by Gender ===")
print(df.groupby('Sex')['Survived'].mean() * 100)
print("=== Survival by Passenger Class ===")
print(df.groupby('Pclass')['Survived'].mean() * 100)
print("=== Average Age by Survived ===")
print(df.groupby('Survived')['Age'].mean().round(2))

# 4. Visualizations & Charts
sns.set_theme(style="whitegrid", palette="muted")

fig, axes = plt.subplots(3, 2, figsize=(15, 16))
fig.suptitle('Titanic Dataset — Improved EDA', fontsize=18, fontweight='bold', y=0.98)

# 1) Survival Count + % labels
surv_counts = df['Survived'].value_counts().sort_index()
labels = ['Not Survived', 'Survived']
colors = ['#e05c5c', '#4f98a3']

axes[0,0].bar(labels, surv_counts, color=colors)
axes[0,0].set_title('Survival Count with Percentage')
axes[0,0].set_ylabel('Count')

total = len(df)
for i, v in enumerate(surv_counts):
    axes[0,0].text(i, v + 10, f'{v/total*100:.1f}%', ha='center', fontsize=10)

# 2) Survival Rate by Gender (%)
gender_surv = df.groupby('Sex')['Survived'].mean().reset_index()
sns.barplot(data=gender_surv, x='Sex', y='Survived', ax=axes[0,1], palette=colors)
axes[0,1].set_title('Survival Rate by Gender')
axes[0,1].set_ylabel('Survival Rate')
axes[0,1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
axes[0,1].set_ylim(0, 1)

# 3) Age Distribution by Survival
sns.histplot(data=df, x='Age', hue='Survived',
             bins=30, kde=True, ax=axes[1,0],
             palette=colors, multiple='stack')
axes[1,0].set_title('Age Distribution by Survival')
axes[1,0].set_xlabel('Age')

# 4) Survival Rate by Passenger Class (%)
pclass_surv = df.groupby('Pclass')['Survived'].mean().reset_index()
sns.barplot(data=pclass_surv, x='Pclass', y='Survived', ax=axes[1,1], palette=colors)
axes[1,1].set_title('Survival Rate by Passenger Class')
axes[1,1].set_ylabel('Survival Rate')
axes[1,1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
axes[1,1].set_ylim(0, 1)

# 5) Fare vs Survival 
df_box = df.copy()
upper = df_box['Fare'].quantile(0.98) 
df_box['Fare_clipped'] = df_box['Fare'].clip(upper=upper)

sns.violinplot(
    data=df,
    x='Survived',
    y='Fare',
    ax=axes[2,0],
    palette=colors,
    cut=0,
    inner='quartile'
)
axes[2,0].set_title('Fare Distribution by Survival')
axes[2,0].set_xticklabels(['Not Survived', 'Survived'])
axes[2,0].set_ylabel('Fare')

# 6) Correlation Heatmap 
corr_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr = df[corr_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2,1])
axes[2,1].set_title('Correlation Heatmap (Selected Features)')

plt.tight_layout()
plt.savefig('titanic_eda_charts_v2.png', dpi=150, bbox_inches='tight')
plt.show()