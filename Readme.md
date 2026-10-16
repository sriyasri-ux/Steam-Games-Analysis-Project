# Steam Games Dataset Analysis and Recommendation System

## Overview
A Python project for exploring a Steam games dataset and recommending games with similar tags.

## Files
- `Codefile.py` — main program.
- `requirements.txt` — Python dependencies.
- `Readme.md` — project instructions.
- `Project_Report.md` — brief project report.
- `games.csv` — optional dataset. If absent, the program creates a small illustrative sample dataset so it can run as a demonstration.

## Features
- Displays the first five records, dataset shape, columns, missing values, and summary.
- Analyzes game prices where a price column is available.
- Analyzes positive/negative reviews or ratings where supported by the dataset columns.
- Counts common tags/genres.
- Uses TF-IDF and cosine similarity for content-based recommendations if scikit-learn is installed.

## Recognized column names
- Game name: `name`, `game_name`, `title`, `app_name`
- Price: `price`, `final_price`, `discounted_price`, `price_usd`
- Tags: `tags`, `genres`, `categories`, `genre`
- Positive reviews: `positive_reviews`, `positive_ratings`, `positive`
- Negative reviews: `negative_reviews`, `negative_ratings`, `negative`
- Rating: `rating`, `user_rating`, `review_score`

Column names are normalized to lowercase with underscores. If your dataset uses different names, update the candidate lists in `Codefile.py`.

## Run
Install dependencies:
```bash
pip install -r requirements.txt
```
Run:
```bash
python Codefile.py
```
When asked, type an exact game name from the list or press Enter to skip recommendations.

## Dataset note
The built-in sample is illustrative test data, not verified live Steam data. For internship submission, use the dataset provided by your instructor/platform and describe the source accurately. Do not present sample statistics as official Steam statistics.
