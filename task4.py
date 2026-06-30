# =============================================
# Project 4: Unveiling the Android App Market
# Dataset: Google Play Store Data
# Internship: Oasis Infobyte - Level 2, Project 4
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------
# STEP 1: Load Datasets
# -----------------------------------------------
apps = pd.read_csv("C:/Users/hp/OneDrive/Desktop/datasets/datasets/apps.csv", index_col=0)
reviews = pd.read_csv("C:/Users/hp/OneDrive/Desktop/datasets/datasets/user_reviews.csv")

print("Apps dataset shape:", apps.shape)
print("Reviews dataset shape:", reviews.shape)
print("\nApps columns:", list(apps.columns))
print("Reviews columns:", list(reviews.columns))

# -----------------------------------------------
# STEP 2: Data Cleaning
# -----------------------------------------------

# Drop duplicates
apps = apps.drop_duplicates(subset='App')

# Clean Rating
apps['Rating'] = pd.to_numeric(apps['Rating'], errors='coerce')

# Clean Reviews
apps['Reviews'] = pd.to_numeric(apps['Reviews'], errors='coerce')

# Clean Size — convert M/k to numeric
apps['Size'] = pd.to_numeric(apps['Size'], errors='coerce')

# Clean Installs
apps['Installs'] = apps['Installs'].str.replace('[+,]', '', regex=True)
apps['Installs'] = pd.to_numeric(apps['Installs'], errors='coerce')

# Clean Price
apps['Price'] = apps['Price'].str.replace('$', '', regex=False)
apps['Price'] = pd.to_numeric(apps['Price'], errors='coerce')

# Drop rows with missing Rating
apps = apps.dropna(subset=['Rating'])

print("\nCleaned Apps shape:", apps.shape)
print("Missing values after cleaning:\n", apps.isnull().sum())

# -----------------------------------------------
# STEP 3: Category Exploration
# -----------------------------------------------
plt.figure(figsize=(12, 6))
category_counts = apps['Category'].value_counts().head(15)
sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis')
plt.title('Top 15 App Categories on Play Store')
plt.xlabel('Number of Apps')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig("plot1_category_distribution.png")
plt.show()

# -----------------------------------------------
# STEP 4: Rating Distribution
# -----------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(apps['Rating'].dropna(), bins=20, kde=True, color='steelblue')
plt.title('App Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("plot2_rating_distribution.png")
plt.show()

# -----------------------------------------------
# STEP 5: Free vs Paid Apps
# -----------------------------------------------
plt.figure(figsize=(6, 4))
type_counts = apps['Type'].value_counts()
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%',
        colors=['#66b3ff', '#ff9999'], startangle=90)
plt.title('Free vs Paid Apps')
plt.tight_layout()
plt.savefig("plot3_free_vs_paid.png")
plt.show()

# -----------------------------------------------
# STEP 6: Top Categories by Average Rating
# -----------------------------------------------
plt.figure(figsize=(12, 6))
avg_rating = apps.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(15)
sns.barplot(x=avg_rating.values, y=avg_rating.index, palette='coolwarm')
plt.title('Top 15 Categories by Average Rating')
plt.xlabel('Average Rating')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig("plot4_category_avg_rating.png")
plt.show()

# -----------------------------------------------
# STEP 7: Installs by Category
# -----------------------------------------------
plt.figure(figsize=(12, 6))
installs_by_cat = apps.groupby('Category')['Installs'].sum().sort_values(ascending=False).head(10)
sns.barplot(x=installs_by_cat.values, y=installs_by_cat.index, palette='magma')
plt.title('Top 10 Categories by Total Installs')
plt.xlabel('Total Installs')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig("plot5_installs_by_category.png")
plt.show()

# -----------------------------------------------
# STEP 8: Sentiment Analysis from Reviews
# -----------------------------------------------
reviews_clean = reviews.dropna(subset=['Sentiment'])

plt.figure(figsize=(6, 4))
sentiment_counts = reviews_clean['Sentiment'].value_counts()
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='Set2')
plt.title('User Review Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("plot6_sentiment_distribution.png")
plt.show()

# -----------------------------------------------
# STEP 9: Sentiment Polarity Distribution
# -----------------------------------------------
plt.figure(figsize=(8, 4))
reviews_clean['Sentiment_Polarity'] = pd.to_numeric(reviews_clean['Sentiment_Polarity'], errors='coerce')
sns.histplot(reviews_clean['Sentiment_Polarity'].dropna(), bins=30, kde=True, color='mediumpurple')
plt.title('Sentiment Polarity Distribution')
plt.xlabel('Polarity Score (-1 Negative to +1 Positive)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("plot7_sentiment_polarity.png")
plt.show()

# -----------------------------------------------
# STEP 10: Merge & Avg Sentiment per App
# -----------------------------------------------
avg_sentiment = reviews_clean.groupby('App')['Sentiment_Polarity'].mean().reset_index()
avg_sentiment.columns = ['App', 'Avg_Sentiment']

merged = apps.merge(avg_sentiment, on='App', how='inner')
print("\nMerged dataset shape:", merged.shape)

# Rating vs Sentiment Polarity
plt.figure(figsize=(7, 5))
sns.scatterplot(x='Avg_Sentiment', y='Rating', data=merged, alpha=0.5, color='teal')
plt.title('App Rating vs Average Sentiment Polarity')
plt.xlabel('Average Sentiment Polarity')
plt.ylabel('App Rating')
plt.tight_layout()
plt.savefig("plot8_rating_vs_sentiment.png")
plt.show()

# -----------------------------------------------
# STEP 11: Summary Stats
# -----------------------------------------------
print("\n=== Summary ===")
print(f"Total Apps Analyzed     : {len(apps)}")
print(f"Total Categories        : {apps['Category'].nunique()}")
print(f"Average App Rating      : {apps['Rating'].mean():.2f}")
print(f"Free Apps               : {(apps['Type'] == 'Free').sum()}")
print(f"Paid Apps               : {(apps['Type'] == 'Paid').sum()}")
print(f"Total Reviews Analyzed  : {len(reviews_clean)}")
print(f"Positive Reviews        : {(reviews_clean['Sentiment'] == 'Positive').sum()}")
print(f"Negative Reviews        : {(reviews_clean['Sentiment'] == 'Negative').sum()}")

print("\nDone! All plots saved.")
