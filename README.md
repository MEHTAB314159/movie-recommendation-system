# 🎬 CineSuggest - BDA Movie Recommendation System

## 🇮🇳 Bollywood + 🌎 International Movies

CineSuggest is a Big Data Analytics mini project that recommends movies using a hybrid recommendation system.

The project combines Bollywood movie data with the MovieLens movie-rating dataset.

## Features

- 🎬 Movie recommendation
- 🇮🇳 Bollywood movies
- 🌎 International movies
- 🔎 Movie search
- ⭐ Movie ratings
- 🎭 Genre filtering
- 📊 Analytics dashboard
- 🔥 Popular movies
- 🔍 Movie explorer
- 🤖 Content-based recommendation
- 👥 Collaborative filtering
- 🏆 Hybrid recommendation

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Plotly
- Streamlit

## Recommendation System

The system uses two techniques.

### Content-Based Filtering

Movie title, genre, language and industry are converted into features using TF-IDF.

### Collaborative Filtering

MovieLens user-rating data is used to identify similar movies based on user behaviour.

### Hybrid Recommendation

The final score combines both approaches:

Content Score × 55%

-

Collaborative Score × 45%

## Datasets

### MovieLens

MovieLens dataset is provided by GroupLens Research.

https://grouplens.org/datasets/movielens/

### Bollywood

A separate Bollywood movie dataset is included in:

`bollywood_movies.csv`

## Project Structure

```text
movie-recommendation-system/
│
├── app.py
├── requirements.txt
├── bollywood_movies.csv
└── README.md
```
