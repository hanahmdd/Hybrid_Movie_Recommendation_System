import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import os
import re
from collections import defaultdict
import streamlit as st

np.random.seed(42)


class HybridRecommendationSystem:
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.id_to_title = {}
        self.title_to_id = {}
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.svd_model = None

    def load_data(self, data_path='./data'):
        """
        Load movie and rating data from local CSVs, uploaded files, or create sample data if not found.
        """
        # Check for direct file uploads through Streamlit
        if 'movies_uploaded' in st.session_state and 'ratings_uploaded' in st.session_state:
            self.movies_df = st.session_state.movies_uploaded
            self.ratings_df = st.session_state.ratings_uploaded
            print("Using uploaded files")
        else:
            # Try local path or generate sample data
            movies_path = os.path.join(data_path, 'movies.csv')
            ratings_path = os.path.join(data_path, 'ratings.csv')

            if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
                print("CSV files not found, creating sample data...")
                self._create_sample_data(data_path)

            # Load data
            self.movies_df = pd.read_csv(movies_path)
            self.ratings_df = pd.read_csv(ratings_path)

        # Create mappings
        self.id_to_title = dict(zip(self.movies_df['movie_id'], self.movies_df['title']))
        self.title_to_id = dict(zip(self.movies_df['title'], self.movies_df['movie_id']))

        print(f"Loaded {len(self.ratings_df)} ratings for {len(self.movies_df)} movies")

    def _create_sample_data(self, data_path):
        os.makedirs(data_path, exist_ok=True)

        movies = {
            'movie_id': list(range(1, 21)),
            'title': [
                'Toy Story (1995)', 'Jumanji (1995)', 'Grumpier Old Men (1995)',
                'Waiting to Exhale (1995)', 'Father of the Bride Part II (1995)',
                'Heat (1995)', 'Sabrina (1995)', 'Tom and Huck (1995)',
                'Sudden Death (1995)', 'GoldenEye (1995)', 'American President, The (1995)',
                'Dracula: Dead and Loving It (1995)', 'Balto (1995)', 'Nixon (1995)',
                'Cutthroat Island (1995)', 'Casino (1995)', 'Sense and Sensibility (1995)',
                'Four Rooms (1995)', 'Ace Ventura: When Nature Calls (1995)', 'Money Train (1995)'
            ],
            'genres': [
                'Adventure|Animation|Children|Comedy|Fantasy',
                'Adventure|Children|Fantasy',
                'Comedy|Romance',
                'Comedy|Drama|Romance',
                'Comedy',
                'Action|Crime|Thriller',
                'Comedy|Romance',
                'Adventure|Children',
                'Action',
                'Action|Adventure|Thriller',
                'Comedy|Drama|Romance',
                'Comedy|Horror',
                'Adventure|Animation|Children',
                'Drama',
                'Action|Adventure|Romance',
                'Crime|Drama',
                'Drama|Romance',
                'Comedy',
                'Comedy',
                'Action|Comedy|Crime|Thriller'
            ]
        }

        ratings = {
            'user_id': np.random.randint(1, 101, 1000),
            'movie_id': np.random.randint(1, 21, 1000),
            'rating': np.random.randint(1, 6, 1000),
            'timestamp': np.random.randint(1_000_000_000, 1_100_000_000, 1000)
        }

        pd.DataFrame(movies).to_csv(os.path.join(data_path, 'movies.csv'), index=False)
        pd.DataFrame(ratings).to_csv(os.path.join(data_path, 'ratings.csv'), index=False)

        print("Sample data created.")

    def preprocess_data(self):
        self.movies_df['genres'] = self.movies_df['genres'].fillna('')
        self.movies_df['clean_title'] = self.movies_df['title'].apply(lambda x: re.sub(r"\s*\(\d{4}\)$", "", x))

    @st.cache_data
    def compute_similarity_matrix(_self, _tfidf_matrix):
        """Compute and cache the cosine similarity matrix."""
        return cosine_similarity(_tfidf_matrix, _tfidf_matrix)

    def build_content_based_model(self):
        """Build the content-based recommendation model with optimizations for Streamlit Cloud."""
        # Use max_features to limit vocabulary size for better performance
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['genres'])

        # Use cached computation for the similarity matrix
        self.cosine_sim = self.compute_similarity_matrix(self.tfidf_matrix)
        print("Content-based model built.")

    def get_content_based_recommendations(self, movie_title, top_n=10):
        if movie_title not in self.title_to_id:
            print(f"Movie '{movie_title}' not found.")
            return pd.DataFrame()

        try:
            movie_id = self.title_to_id[movie_title]
            idx = self.movies_df[self.movies_df['movie_id'] == movie_id].index[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
            indices = [i[0] for i in sim_scores]
            return self.movies_df.iloc[indices][['movie_id', 'title', 'genres']]
        except Exception as e:
            print(f"Error getting content-based recommendations: {str(e)}")
            return pd.DataFrame()

    def train_collaborative_model(self):
        try:
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(self.ratings_df[['user_id', 'movie_id', 'rating']], reader)
            trainset = data.build_full_trainset()

            # Reduce number of factors and epochs for better performance on Streamlit Cloud
            self.svd_model = SVD(n_factors=50, n_epochs=10, lr_all=0.005, reg_all=0.02)
            self.svd_model.fit(trainset)
            print("Collaborative filtering model trained.")
        except Exception as e:
            print(f"Error training collaborative model: {str(e)}")
            raise

    def get_collaborative_recommendations(self, user_id, top_n=10):
        try:
            seen = self.ratings_df[self.ratings_df['user_id'] == user_id]['movie_id'].unique()
            unseen = self.movies_df[~self.movies_df['movie_id'].isin(seen)]['movie_id'].values

            # Use a smaller sample for very large datasets to improve performance
            if len(unseen) > 10000:
                unseen = np.random.choice(unseen, 10000, replace=False)

            predictions = [(mid, self.svd_model.predict(user_id, mid).est) for mid in unseen]
            predictions.sort(key=lambda x: x[1], reverse=True)
            top = predictions[:top_n]
            top_movie_ids = [movie_id for movie_id, _ in top]
            top_scores = [score for _, score in top]
            df = self.movies_df[self.movies_df['movie_id'].isin(top_movie_ids)].copy()
            df['predicted_rating'] = df['movie_id'].map(dict(zip(top_movie_ids, top_scores)))
            return df[['movie_id', 'title', 'genres', 'predicted_rating']].sort_values('predicted_rating',
                                                                                       ascending=False)
        except Exception as e:
            print(f"Error getting collaborative recommendations: {str(e)}")
            return pd.DataFrame()

    def get_hybrid_recommendations(self, user_id, movie_title=None, content_weight=0.4, collab_weight=0.6, top_n=10):
        try:
            if movie_title and movie_title in self.title_to_id:
                # Get content and collaborative recommendations
                content_recs = self.get_content_based_recommendations(movie_title, top_n=50)
                collab_recs = self.get_collaborative_recommendations(user_id, top_n=50)

                content_ids = set(content_recs['movie_id'])
                collab_dict = dict(zip(collab_recs['movie_id'], collab_recs['predicted_rating']))

                hybrid_scores = defaultdict(float)
                for i, row in enumerate(content_recs.itertuples()):
                    score = (len(content_recs) - i) / len(content_recs)  # normalize to [0,1]
                    hybrid_scores[row.movie_id] += content_weight * score

                for row in collab_recs.itertuples():
                    score = (row.predicted_rating - 1) / 4  # normalize rating from 1-5 to [0,1]
                    hybrid_scores[row.movie_id] += collab_weight * score

                top_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
                top_ids = [movie_id for movie_id, _ in top_hybrid]

                results = self.movies_df[self.movies_df['movie_id'].isin(top_ids)].copy()
                results['hybrid_score'] = results['movie_id'].map(dict(top_hybrid))
                return results[['movie_id', 'title', 'genres', 'hybrid_score']].sort_values('hybrid_score',
                                                                                            ascending=False)
            else:
                return self.get_collaborative_recommendations(user_id, top_n=top_n)
        except Exception as e:
            print(f"Error getting hybrid recommendations: {str(e)}")
            return pd.DataFrame()

    def evaluate_models(self, test_size=0.2):
        try:
            # Use a smaller sample for evaluation if dataset is very large
            if len(self.ratings_df) > 100000:
                eval_df = self.ratings_df.sample(n=100000, random_state=42)
            else:
                eval_df = self.ratings_df

            train_df, test_df = train_test_split(eval_df, test_size=test_size, random_state=42)
            reader = Reader(rating_scale=(1, 5))
            train_data = Dataset.load_from_df(train_df[['user_id', 'movie_id', 'rating']], reader)
            test_data = Dataset.load_from_df(test_df[['user_id', 'movie_id', 'rating']], reader)

            svd = SVD(n_factors=50, n_epochs=10, lr_all=0.005, reg_all=0.02)
            trainset = train_data.build_full_trainset()
            svd.fit(trainset)

            testset = test_data.build_full_trainset().build_testset()
            predictions = svd.test(testset)

            actual = [pred.r_ui for pred in predictions]
            predicted = [pred.est for pred in predictions]

            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mae = mean_absolute_error(actual, predicted)

            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            return {"RMSE": rmse, "MAE": mae}
        except Exception as e:
            print(f"Error evaluating models: {str(e)}")
            return {"RMSE": 0.0, "MAE": 0.0}


# ---------------- Main Execution ----------------

def main():
    recommender = HybridRecommendationSystem()

    # Load and prepare data
    recommender.load_data()
    recommender.preprocess_data()

    # Train models
    try:
        recommender.build_content_based_model()
        recommender.train_collaborative_model()
    except Exception as e:
        print(f"Error building models: {str(e)}")
        return

    # Evaluate
    metrics = recommender.evaluate_models()

    # Show example recommendations
    print("\nContent-based Recommendations for 'Toy Story (1995)':")
    print(recommender.get_content_based_recommendations("Toy Story (1995)"))

    print("\nCollaborative Recommendations for user_id=1:")
    print(recommender.get_collaborative_recommendations(user_id=1))

    print("\nHybrid Recommendations for user_id=1 and 'Toy Story (1995)':")
    print(recommender.get_hybrid_recommendations(user_id=1, movie_title="Toy Story (1995)"))


if __name__ == "__main__":
    main()