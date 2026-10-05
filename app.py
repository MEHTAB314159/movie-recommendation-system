import os
import zipfile
import requests
import warnings

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CineSuggest - Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 45px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 30px;
}

.movie-card {
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #dddddd;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MOVIELENS CONFIGURATION
# ============================================================

MOVIELENS_URL = (
    "https://files.grouplens.org/datasets/movielens/"
    "ml-latest-small.zip"
)

DATA_FOLDER = "data"

ZIP_FILE = os.path.join(
    DATA_FOLDER,
    "movielens.zip"
)

MOVIELENS_FOLDER = os.path.join(
    DATA_FOLDER,
    "ml-latest-small"
)


# ============================================================
# DOWNLOAD MOVIELENS
# ============================================================

def download_movielens():

    os.makedirs(
        DATA_FOLDER,
        exist_ok=True
    )

    if os.path.exists(
        MOVIELENS_FOLDER
    ):
        return

    response = requests.get(
        MOVIELENS_URL,
        timeout=60
    )

    response.raise_for_status()

    with open(
        ZIP_FILE,
        "wb"
    ) as file:

        file.write(
            response.content
        )

    with zipfile.ZipFile(
        ZIP_FILE,
        "r"
    ) as zip_ref:

        zip_ref.extractall(
            DATA_FOLDER
        )

    if os.path.exists(
        ZIP_FILE
    ):

        os.remove(
            ZIP_FILE
        )


# ============================================================
# LOAD MOVIELENS
# ============================================================

@st.cache_data
def load_movielens():

    download_movielens()

    movies_file = os.path.join(
        MOVIELENS_FOLDER,
        "movies.csv"
    )

    ratings_file = os.path.join(
        MOVIELENS_FOLDER,
        "ratings.csv"
    )

    movies = pd.read_csv(
        movies_file
    )

    ratings = pd.read_csv(
        ratings_file
    )

    return movies, ratings


# ============================================================
# LOAD BOLLYWOOD DATA
# ============================================================

@st.cache_data
def load_bollywood():

    file_path = "bollywood_movies.csv"

    bollywood = pd.read_csv(
        file_path
    )

    return bollywood


# ============================================================
# LOAD DATA
# ============================================================

with st.spinner(
    "Loading movie database..."
):

    hollywood_movies, ratings = (
        load_movielens()
    )

    bollywood_movies = (
        load_bollywood()
    )


# ============================================================
# PREPARE MOVIELENS DATA
# ============================================================

movie_statistics = (
    ratings
    .groupby("movieId")
    .agg(
        average_rating=("rating", "mean"),
        rating_count=("rating", "count")
    )
    .reset_index()
)


hollywood_movies = hollywood_movies.merge(
    movie_statistics,
    on="movieId",
    how="left"
)


hollywood_movies[
    "average_rating"
] = (
    hollywood_movies[
        "average_rating"
    ]
    .fillna(0)
    .round(2)
)


hollywood_movies[
    "rating_count"
] = (
    hollywood_movies[
        "rating_count"
    ]
    .fillna(0)
    .astype(int)
)


# ============================================================
# STANDARDIZE BOLLYWOOD DATA
# ============================================================

bollywood_movies = (
    bollywood_movies.rename(
        columns={
            "movieId": "movieId",
            "title": "title",
            "genres": "genres",
            "year": "year",
            "language": "language",
            "industry": "industry",
            "rating": "average_rating",
            "rating_count": "rating_count"
        }
    )
)


# ============================================================
# ADD MISSING COLUMNS
# ============================================================

hollywood_movies[
    "language"
] = "English"


hollywood_movies[
    "industry"
] = "International"


hollywood_movies[
    "year"
] = (
    hollywood_movies[
        "title"
    ]
    .str.extract(
        r"\((\d{4})\)"
    )[0]
)


hollywood_movies[
    "year"
] = pd.to_numeric(
    hollywood_movies[
        "year"
    ],
    errors="coerce"
)


# ============================================================
# SELECT COLUMNS
# ============================================================

hollywood_final = (
    hollywood_movies[
        [
            "movieId",
            "title",
            "genres",
            "year",
            "language",
            "industry",
            "average_rating",
            "rating_count"
        ]
    ].copy()
)


bollywood_final = (
    bollywood_movies[
        [
            "movieId",
            "title",
            "genres",
            "year",
            "language",
            "industry",
            "average_rating",
            "rating_count"
        ]
    ].copy()
)


# ============================================================
# COMBINE MOVIES
# ============================================================

movies = pd.concat(
    [
        hollywood_final,
        bollywood_final
    ],
    ignore_index=True
)


# ============================================================
# CLEAN DATA
# ============================================================

movies["title"] = (
    movies["title"]
    .fillna("")
)


movies["genres"] = (
    movies["genres"]
    .fillna("")
)


movies["language"] = (
    movies["language"]
    .fillna("Unknown")
)


movies["industry"] = (
    movies["industry"]
    .fillna("Unknown")
)


movies["average_rating"] = (
    pd.to_numeric(
        movies["average_rating"],
        errors="coerce"
    )
    .fillna(0)
)


movies["rating_count"] = (
    pd.to_numeric(
        movies["rating_count"],
        errors="coerce"
    )
    .fillna(0)
)


movies["year"] = (
    pd.to_numeric(
        movies["year"],
        errors="coerce"
    )
)


# ============================================================
# CONTENT FEATURES
# ============================================================

movies["features"] = (
    movies["title"]
    + " "
    + movies["genres"]
    + " "
    + movies["language"]
    + " "
    + movies["industry"]
)


# ============================================================
# TF-IDF MODEL
# ============================================================

@st.cache_resource
def create_content_model(
    feature_data
):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform(
        feature_data
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    model.fit(
        matrix
    )

    return matrix, model


content_matrix, content_model = (
    create_content_model(
        movies["features"]
    )
)


# ============================================================
# COLLABORATIVE MODEL
# ============================================================

@st.cache_resource
def create_collaborative_model(
    ratings
):

    user_movie = ratings.pivot_table(
        index="movieId",
        columns="userId",
        values="rating",
        fill_value=0
    )

    matrix = csr_matrix(
        user_movie.values
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    model.fit(
        matrix
    )

    movie_to_index = {
        movie_id: index
        for index, movie_id
        in enumerate(
            user_movie.index
        )
    }

    index_to_movie = {
        index: movie_id
        for movie_id, index
        in movie_to_index.items()
    }

    return (
        matrix,
        model,
        movie_to_index,
        index_to_movie
    )


(
    collaborative_matrix,
    collaborative_model,
    movie_to_index,
    index_to_movie
) = create_collaborative_model(
    ratings
)


# ============================================================
# CONTENT RECOMMENDATION
# ============================================================

def content_recommend(
    movie_index,
    number
):

    distances, indices = (
        content_model.kneighbors(
            content_matrix[
                movie_index
            ],
            n_neighbors=min(
                number + 1,
                len(movies)
            )
        )
    )

    result = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        if index == movie_index:
            continue

        score = (
            1 - distance
        ) * 100

        result.append(
            {
                "index": index,
                "content_score": score
            }
        )

    return pd.DataFrame(
        result
    )


# ============================================================
# COLLABORATIVE RECOMMENDATION
# ============================================================

def collaborative_recommend(
    movie_id,
    number
):

    if movie_id not in movie_to_index:

        return pd.DataFrame(
            columns=[
                "movieId",
                "collaborative_score"
            ]
        )

    movie_index = (
        movie_to_index[
            movie_id
        ]
    )

    distances, indices = (
        collaborative_model.kneighbors(
            collaborative_matrix[
                movie_index
            ],
            n_neighbors=min(
                number + 1,
                collaborative_matrix.shape[0]
            )
        )
    )

    result = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        recommended_id = (
            index_to_movie[
                index
            ]
        )

        if recommended_id == movie_id:
            continue

        score = (
            1 - distance
        ) * 100

        result.append(
            {
                "movieId": recommended_id,
                "collaborative_score": score
            }
        )

    return pd.DataFrame(
        result
    )


# ============================================================
# HYBRID RECOMMENDATION
# ============================================================

def recommend_movies(
    selected_index,
    selected_movie_id,
    number
):

    selected_movie = movies.iloc[
        selected_index
    ]

    # --------------------------------------------------------
    # CONTENT
    # --------------------------------------------------------

    content = content_recommend(
        selected_index,
        50
    )

    content[
        "movieId"
    ] = content[
        "index"
    ].apply(
        lambda x:
        movies.iloc[x]["movieId"]
    )

    # --------------------------------------------------------
    # BOLLYWOOD MOVIES
    # --------------------------------------------------------

    if selected_movie["industry"] == "Bollywood":

        content = content[
            content[
                "index"
            ] != selected_index
        ]

    # --------------------------------------------------------
    # COLLABORATIVE
    # --------------------------------------------------------

    collaborative = (
        collaborative_recommend(
            selected_movie_id,
            50
        )
    )

    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    result = pd.merge(
        content[
            [
                "movieId",
                "content_score"
            ]
        ],
        collaborative,
        on="movieId",
        how="outer"
    )

    result[
        "content_score"
    ] = (
        result[
            "content_score"
        ]
        .fillna(0)
    )

    result[
        "collaborative_score"
    ] = (
        result[
            "collaborative_score"
        ]
        .fillna(0)
    )

    # --------------------------------------------------------
    # HYBRID SCORE
    # --------------------------------------------------------

    result[
        "hybrid_score"
    ] = (
        result[
            "content_score"
        ] * 0.55
        +
        result[
            "collaborative_score"
        ] * 0.45
    )

    # --------------------------------------------------------
    # JOIN MOVIE DATA
    # --------------------------------------------------------

    result = result.merge(
        movies[
            [
                "movieId",
                "title",
                "genres",
                "year",
                "language",
                "industry",
                "average_rating",
                "rating_count"
            ]
        ],
        on="movieId",
        how="left"
    )

    # --------------------------------------------------------
    # REMOVE SELECTED MOVIE
    # --------------------------------------------------------

    result = result[
        result["title"]
        != selected_movie["title"]
    ]

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    result = result.sort_values(
        [
            "hybrid_score",
            "average_rating"
        ],
        ascending=False
    )

    return result.head(
        number
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🎬 CineSuggest'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    '🇮🇳 Bollywood + 🌎 International Movie '
    'Recommendation System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🎯 Navigation"
)

page = st.sidebar.radio(
    "Choose a module",
    [
        "🏠 Recommendation",
        "📊 Analytics",
        "🔥 Popular Movies",
        "🔍 Movie Explorer",
        "ℹ️ About Project"
    ]
)


# ============================================================
# RECOMMENDATION PAGE
# ============================================================

if page == "🏠 Recommendation":

    st.header(
        "🎬 Movie Recommendation"
    )

    st.sidebar.subheader(
        "Movie Category"
    )

    category = st.sidebar.selectbox(
        "Choose category",
        [
            "All Movies",
            "🇮🇳 Bollywood",
            "🌎 International"
        ]
    )

    if category == "🇮🇳 Bollywood":

        filtered = movies[
            movies["industry"]
            == "Bollywood"
        ]

    elif category == "🌎 International":

        filtered = movies[
            movies["industry"]
            == "International"
        ]

    else:

        filtered = movies.copy()

    search = st.text_input(
        "🔎 Search movie",
        placeholder="Example: 3 Idiots, Dangal, Toy Story..."
    )

    if search:

        filtered = filtered[
            filtered["title"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    if len(filtered) == 0:

        st.warning(
            "No movie found."
        )

        st.stop()

    selected_title = st.selectbox(
        "🎥 Select a movie",
        filtered["title"].tolist()
    )

    selected_index = movies.index[
        movies["title"]
        == selected_title
    ][0]

    selected_movie = movies.iloc[
        selected_index
    ]

    selected_movie_id = (
        selected_movie["movieId"]
    )

    st.subheader(
        "🎥 Movie Information"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "⭐ Rating",
            f"{selected_movie['average_rating']:.2f}"
        )

    with c2:

        st.metric(
            "👥 Ratings",
            f"{int(selected_movie['rating_count']):,}"
        )

    with c3:

        st.metric(
            "🎭 Genre",
            selected_movie[
                "genres"
            ].split("|")[0]
        )

    with c4:

        year = selected_movie[
            "year"
        ]

        if pd.notna(year):

            st.metric(
                "📅 Year",
                int(year)
            )

        else:

            st.metric(
                "📅 Year",
                "N/A"
            )

    st.write(
        "**Industry:**",
        selected_movie["industry"]
    )

    st.write(
        "**Language:**",
        selected_movie["language"]
    )

    st.write(
        "**Genres:**",
        selected_movie["genres"]
    )

    number = st.slider(
        "Number of recommendations",
        5,
        15,
        10
    )

    if st.button(
        "🚀 Generate Recommendations",
        use_container_width=True
    ):

        with st.spinner(
            "Finding the best movies for you..."
        ):

            recommendations = (
                recommend_movies(
                    selected_index,
                    selected_movie_id,
                    number
                )
            )

        st.success(
            "Recommendations generated!"
        )

        for position, (
            index,
            row
        ) in enumerate(
            recommendations.iterrows(),
            start=1
        ):

            st.markdown(
                f"""
                <div class="movie-card">

                <h3>
                #{position} 🎬 {row['title']}
                </h3>

                <b>Industry:</b>
                {row['industry']}
                <br>

                <b>Language:</b>
                {row['language']}
                <br>

                <b>Genre:</b>
                {row['genres']}
                <br><br>

                ⭐ <b>Rating:</b>
                {row['average_rating']:.2f}

                &nbsp;&nbsp;&nbsp;

                👥 <b>Ratings:</b>
                {int(row['rating_count']):,}

                <br><br>

                🤖 <b>Content Score:</b>
                {row['content_score']:.2f}%

                &nbsp;&nbsp;&nbsp;

                👥 <b>Collaborative Score:</b>
                {row['collaborative_score']:.2f}%

                <br><br>

                🏆 <b>Hybrid Score:</b>
                {row['hybrid_score']:.2f}%

                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.header(
        "📊 Movie Analytics Dashboard"
    )

    total_movies = len(
        movies
    )

    total_users = ratings[
        "userId"
    ].nunique()

    total_ratings = len(
        ratings
    )

    average_rating = ratings[
        "rating"
    ].mean()

    bollywood_count = len(
        movies[
            movies["industry"]
            == "Bollywood"
        ]
    )

    international_count = len(
        movies[
            movies["industry"]
            == "International"
        ]
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "🎬 Total Movies",
            f"{total_movies:,}"
        )

    with c2:

        st.metric(
            "🇮🇳 Bollywood",
            f"{bollywood_count:,}"
        )

    with c3:

        st.metric(
            "🌎 International",
            f"{international_count:,}"
        )

    with c4:

        st.metric(
            "👤 Users",
            f"{total_users:,}"
        )

    with c5:

        st.metric(
            "⭐ Ratings",
            f"{total_ratings:,}"
        )

    st.divider()

    # INDUSTRY CHART

    industry_count = (
        movies[
            "industry"
        ]
        .value_counts()
        .reset_index()
    )

    industry_count.columns = [
        "Industry",
        "Movies"
    ]

    fig1 = px.pie(
        industry_count,
        names="Industry",
        values="Movies",
        title="Bollywood vs International Movies"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # RATING DISTRIBUTION

    st.subheader(
        "⭐ Rating Distribution"
    )

    rating_distribution = (
        ratings[
            "rating"
        ]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    rating_distribution.columns = [
        "Rating",
        "Count"
    ]

    fig2 = px.bar(
        rating_distribution,
        x="Rating",
        y="Count",
        title="Movie Rating Distribution"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # GENRES

    st.subheader(
        "🎭 Popular Genres"
    )

    genre_data = (
        movies[
            "genres"
        ]
        .str.split("|")
        .explode()
    )

    genre_count = (
        genre_data
        .value_counts()
        .head(15)
        .reset_index()
    )

    genre_count.columns = [
        "Genre",
        "Movies"
    ]

    fig3 = px.bar(
        genre_count,
        x="Movies",
        y="Genre",
        orientation="h",
        title="Top 15 Genres"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )


# ============================================================
# POPULAR MOVIES
# ============================================================

elif page == "🔥 Popular Movies":

    st.header(
        "🔥 Popular Movies"
    )

    industry_filter = st.selectbox(
        "Select industry",
        [
            "All",
            "Bollywood",
            "International"
        ]
    )

    minimum_ratings = st.slider(
        "Minimum number of ratings",
        0,
        500,
        20
    )

    popular = movies[
        movies[
            "rating_count"
        ]
        >= minimum_ratings
    ].copy()

    if industry_filter != "All":

        popular = popular[
            popular[
                "industry"
            ]
            == industry_filter
        ]

    popular = popular.sort_values(
        [
            "average_rating",
            "rating_count"
        ],
        ascending=False
    ).head(20)

    st.dataframe(
        popular[
            [
                "title",
                "year",
                "industry",
                "language",
                "genres",
                "average_rating",
                "rating_count"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MOVIE EXPLORER
# ============================================================

elif page == "🔍 Movie Explorer":

    st.header(
        "🔍 Explore Movie Database"
    )

    col1, col2 = st.columns(2)

    with col1:

        industry = st.selectbox(
            "Industry",
            [
                "All",
                "Bollywood",
                "International"
            ]
        )

    with col2:

        language = st.selectbox(
            "Language",
            [
                "All"
            ]
            + sorted(
                movies[
                    "language"
                ]
                .dropna()
                .unique()
                .tolist()
            )
        )

    result = movies.copy()

    if industry != "All":

        result = result[
            result[
                "industry"
            ]
            == industry
        ]

    if language != "All":

        result = result[
            result[
                "language"
            ]
            == language
        ]

    st.write(
        f"Showing **{len(result):,} movies**"
    )

    st.dataframe(
        result[
            [
                "title",
                "year",
                "language",
                "industry",
                "genres",
                "average_rating",
                "rating_count"
            ]
        ].sort_values(
            "average_rating",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About Project":

    st.header(
        "ℹ️ About the Project"
    )

    st.markdown("""
## 🎬 CineSuggest

### Big Data Movie Recommendation System

This BDA mini project combines movie information
and user-rating data to recommend movies.

### 🇮🇳 Bollywood

The system includes a separate Bollywood movie
dataset containing Hindi movies.

### 🌎 International Movies

International movie information and user ratings
are obtained from the MovieLens dataset.

### 🤖 Recommendation Methods

#### Content-Based Filtering

Uses:

- Movie title
- Genre
- Language
- Industry

TF-IDF is used to convert movie information
into numerical features.

#### Collaborative Filtering

Uses user-movie rating patterns to identify
similar movies.

#### Hybrid Recommendation

The final recommendation score combines:

**55% Content Score**

+

**45% Collaborative Score**

### 📊 Analytics

The dashboard provides:

- Movie statistics
- Industry comparison
- Rating distribution
- Genre analysis
- Popular movies
- Movie explorer

### 🚀 Future Scope

The project can be extended with:

- PySpark
- Hadoop
- HDFS
- Hive
- Apache Flume
- Kafka
- Real-time recommendations
- Movie posters
- Sentiment analysis
- Deep learning
""")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BDA Mini Project | CineSuggest | "
    "Bollywood + International Movie Recommendation System"
)