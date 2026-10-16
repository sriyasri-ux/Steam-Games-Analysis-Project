# Project Report: Steam Games Dataset Analysis and Recommendation System

## 1. Introduction
This project demonstrates exploratory analysis of video-game data using Python. It summarizes available game details and, when tags are available, recommends games with similar content.

## 2. Objectives
- Inspect the dataset structure and missing values.
- Explore prices and free/paid game counts.
- Summarize review or rating information when present.
- Identify frequent tags or genres.
- Demonstrate content-based recommendations.

## 3. Technologies
Python, Pandas, Scikit-learn, TF-IDF vectorization, and cosine similarity.

## 4. Methodology
The program loads `games.csv`, standardizes column names, removes duplicate rows, prints an overview, and performs analyses only when recognized columns are present. TF-IDF converts tag text to vectors, and cosine similarity ranks games by tag similarity.

## 5. Results
Results are generated at runtime and depend on the dataset used. The sample dataset is for demonstration only and is not verified live Steam data.

## 6. Limitations
Recommendations rely on tag quality and do not use player behavior. Dataset columns and data quality can vary.

## 7. Conclusion
The project provides a basic example of data analysis and content-based recommendation for game datasets. It can be extended with visualizations, sentiment analysis, and additional game features.
