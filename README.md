# oasis-infobyte-internship-project4-level2
# Unveiling the Android App Market

Analyzing Google Play Store data to understand app market dynamics — covering 
categories, ratings, installs, pricing, and user sentiment.

## About the Dataset

Used two datasets — one containing details of around 9,600 apps (category, rating, 
size, installs, price, etc.) and another with over 64,000 user reviews along with 
their sentiment labels.

## What I Did

Started by cleaning both datasets — fixed data types for ratings, reviews, size, 
installs, and price columns, and removed duplicates and missing values.

Explored which categories dominate the Play Store, how ratings are distributed, and 
the split between free and paid apps. Compared categories by average rating and total 
installs to see what users engage with the most.

For sentiment analysis, used the user reviews dataset to check the overall sentiment 
distribution (positive, negative, neutral) and the polarity scores. Merged both 
datasets to check whether higher sentiment polarity correlates with better app ratings.

## Results

Most apps fall into a handful of dominant categories, and the majority are free. 
Family and Game categories have the highest number of apps. Positive reviews 
significantly outnumber negative ones, and there's a mild positive relationship 
between sentiment polarity and app ratings.

## Libraries Used

pandas, numpy, matplotlib, seaborn

## How to Run

1. Place `apps.csv` and `user_reviews.csv` in the project folder
2. Run:

## Internship

Oasis Infobyte — Data Analytics Internship | Level 2, Project 4
