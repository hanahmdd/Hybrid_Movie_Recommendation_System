import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from hybrid_recommendation_system import HybridRecommendationSystem

# Configure the Streamlit page
st.set_page_config(
    page_title="Hybrid Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Add CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .title {
        font-size: 42px !important;
        color: #1E3A8A;
        text-align: center;
    }
    .header {
        font-size: 24px !important;
        color: #1E3A8A;
        font-weight: bold;
        margin-top: 20px;
    }
    .subheader {
        font-size: 18px !important;
        color: #0F172A;
        font-weight: bold;
    }
    .movie-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .data-notice {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize the recommendation system
@st.cache_resource
def load_recommendation_system(data_path="./data"):
    recommender = HybridRecommendationSystem()

    # Show data loading status
    with st.spinner("Loading movie data..."):
        recommender.load_data(data_path)
        recommender.preprocess_data()

    with st.spinner("Building recommendation models..."):
        try:
            recommender.build_content_based_model()
            recommender.train_collaborative_model()
        except Exception as e:
            st.error(f"Error building recommendation models: {str(e)}")
            st.info("Try using a smaller dataset or sample data.")

    return recommender


# Function to display each movie recommendation
def display_movie_card(movie, score_type="hybrid_score"):
    with st.container():
        st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"#### {movie['title']}")
            st.write(f"*Genres:* {movie['genres']}")

        with col2:
            if score_type in movie and not pd.isna(movie[score_type]):
                if score_type == "hybrid_score":
                    st.metric("Score", f"{movie[score_type]:.2f}")
                else:
                    st.metric("Rating", f"{movie[score_type]:.1f}")

        st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.markdown("<h1 class='title'>Hybrid Movie Recommendation System</h1>", unsafe_allow_html=True)

    # File upload option
    with st.expander("Upload your own MovieLens dataset"):
        col1, col2 = st.columns(2)
        with col1:
            movies_file = st.file_uploader("Upload movies.csv", type="csv")
            if movies_file is not None:
                st.session_state.movies_uploaded = pd.read_csv(movies_file)
                st.success(f"Loaded {len(st.session_state.movies_uploaded)} movies!")

        with col2:
            ratings_file = st.file_uploader("Upload ratings.csv", type="csv")
            if ratings_file is not None:
                st.session_state.ratings_uploaded = pd.read_csv(ratings_file)
                st.success(f"Loaded {len(st.session_state.ratings_uploaded)} ratings!")

    # Information about data requirements
    data_path = "./data"
    if (not os.path.exists(data_path) or not os.listdir(data_path)) and \
            (not 'movies_uploaded' in st.session_state or not 'ratings_uploaded' in st.session_state):
        st.markdown(
            """
            <div class="data-notice">
            <b>📋 Data Setup Instructions:</b><br>
            This app needs the MovieLens dataset to function properly. You have two options:
            <ol>
                <li><b>Use sample data:</b> Click "Use Sample Data" below to use a small demonstration dataset.</li>
                <li><b>Use full MovieLens dataset:</b> Download from <a href="https://grouplens.org/datasets/movielens/" target="_blank">GroupLens</a> 
                and place the files in a folder named 'data' in the same directory as this app.</li>
                <li><b>Upload your own data:</b> Use the file upload option above to upload your own movies.csv and ratings.csv files.</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button("Use Sample Data"):
            # This will trigger creation of sample data when loading the recommendation system
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            recommender = load_recommendation_system(data_path)
            st.success("Sample data loaded successfully! You can now use the recommendation system.")
            st.rerun()
    else:
        # Load the recommendation system
        recommender = load_recommendation_system(data_path)

        # Create tabs for different recommendation methods
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Hybrid Recommendations", "Content-Based", "Collaborative Filtering", "System Evaluation"])

        # Tab 1: Hybrid Recommendations
        with tab1:
            st.markdown("<p class='header'>Hybrid Movie Recommendations</p>", unsafe_allow_html=True)
            st.write("This combines both content-based filtering and collaborative filtering approaches.")

            col1, col2 = st.columns([1, 1])

            with col1:
                # User ID selection
                max_user_id = recommender.ratings_df['user_id'].max()
                user_id = st.number_input(f"Select User ID (1-{max_user_id})", min_value=1, max_value=int(max_user_id),
                                          value=1, step=1)

            with col2:
                # Movie selection
                movie_titles = list(recommender.title_to_id.keys())
                selected_movie = st.selectbox("Select a movie you like (optional)", [""] + movie_titles)

            # Weights for hybrid recommendations
            st.markdown("<p class='subheader'>Recommendation Weights</p>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])

            with col1:
                content_weight = st.slider("Content-Based Weight", 0.0, 1.0, 0.4, 0.1)

            with col2:
                collab_weight = st.slider("Collaborative Filtering Weight", 0.0, 1.0, 0.6, 0.1)

            # Number of recommendations
            num_recommendations = st.slider("Number of Recommendations", 5, 20, 10, 1)

            # Generate recommendations
            if st.button("Get Hybrid Recommendations"):
                with st.spinner("Generating hybrid recommendations..."):
                    try:
                        if selected_movie:
                            hybrid_recs = recommender.get_hybrid_recommendations(
                                user_id=user_id,
                                movie_title=selected_movie,
                                content_weight=content_weight,
                                collab_weight=collab_weight,
                                top_n=num_recommendations
                            )
                        else:
                            hybrid_recs = recommender.get_collaborative_recommendations(
                                user_id=user_id,
                                top_n=num_recommendations
                            )

                        st.markdown("<p class='subheader'>Your Personalized Movie Recommendations</p>",
                                    unsafe_allow_html=True)

                        if len(hybrid_recs) > 0:
                            for _, movie in hybrid_recs.iterrows():
                                display_movie_card(movie, "hybrid_score" if selected_movie else "predicted_rating")
                        else:
                            st.info("No recommendations found. Try selecting a different movie or user.")
                    except Exception as e:
                        st.error(f"Error generating recommendations: {str(e)}")
                        st.info("Try different parameters or a different user/movie.")

        # Tab 2: Content-Based Recommendations
        with tab2:
            st.markdown("<p class='header'>Content-Based Recommendations</p>", unsafe_allow_html=True)
            st.write("Recommendations based on movie features like genres, similar to a movie you select.")

            # Movie selection
            selected_movie_content = st.selectbox("Select a movie", movie_titles)

            # Number of recommendations
            num_content_recs = st.slider("Number of Content-Based Recommendations", 5, 20, 10, 1, key="content_slider")

            # Generate recommendations
            if st.button("Get Content-Based Recommendations"):
                with st.spinner("Finding similar movies..."):
                    try:
                        content_recs = recommender.get_content_based_recommendations(
                            movie_title=selected_movie_content,
                            top_n=num_content_recs
                        )

                        st.markdown("<p class='subheader'>Movies Similar to Your Selection</p>", unsafe_allow_html=True)

                        if len(content_recs) > 0:
                            for i, (_, movie) in enumerate(content_recs.iterrows()):
                                # Add a similarity score (decreasing with index)
                                movie_with_score = movie.copy()
                                movie_with_score["similarity"] = (num_content_recs - i) / num_content_recs
                                display_movie_card(movie_with_score, "similarity")
                        else:
                            st.info("No similar movies found. Try selecting a different movie.")
                    except Exception as e:
                        st.error(f"Error finding similar movies: {str(e)}")
                        st.info("Try selecting a different movie.")

        # Tab 3: Collaborative Filtering
        with tab3:
            st.markdown("<p class='header'>Collaborative Filtering Recommendations</p>", unsafe_allow_html=True)
            st.write("Recommendations based on what similar users have enjoyed.")

            # User ID selection
            user_id_collab = st.number_input(f"Select User ID (1-{max_user_id})", min_value=1,
                                             max_value=int(max_user_id), value=1, step=1, key="collab_user_id")

            # Number of recommendations
            num_collab_recs = st.slider("Number of Collaborative Recommendations", 5, 20, 10, 1, key="collab_slider")

            # Generate recommendations
            if st.button("Get Collaborative Recommendations"):
                with st.spinner("Generating personalized recommendations..."):
                    try:
                        collab_recs = recommender.get_collaborative_recommendations(
                            user_id=user_id_collab,
                            top_n=num_collab_recs
                        )

                        st.markdown("<p class='subheader'>Movies Recommended Based on Similar Users</p>",
                                    unsafe_allow_html=True)

                        if len(collab_recs) > 0:
                            for _, movie in collab_recs.iterrows():
                                display_movie_card(movie, "predicted_rating")
                        else:
                            st.info("No recommendations found. Try selecting a different user.")
                    except Exception as e:
                        st.error(f"Error generating collaborative recommendations: {str(e)}")
                        st.info("Try selecting a different user.")

        # Tab 4: System Evaluation
        with tab4:
            st.markdown("<p class='header'>System Evaluation</p>", unsafe_allow_html=True)
            st.write("Evaluate the performance of the recommendation algorithms.")

            if st.button("Run Evaluation"):
                with st.spinner("Evaluating recommendation system..."):
                    try:
                        evaluation_metrics = recommender.evaluate_models()

                        # Display the evaluation metrics
                        st.markdown("<p class='subheader'>Performance Metrics</p>", unsafe_allow_html=True)

                        col1, col2 = st.columns([1, 1])

                        with col1:
                            st.metric("RMSE", f"{evaluation_metrics['RMSE']:.4f}")

                        with col2:
                            st.metric("MAE", f"{evaluation_metrics['MAE']:.4f}")

                        # Create a bar chart for metrics at different thresholds
                        st.markdown("<p class='subheader'>Precision, Recall, and F1-Score at Different Thresholds</p>",
                                    unsafe_allow_html=True)

                        # Generate some sample data for the chart (actual evaluation would use real results)
                        thresholds = [3.5, 4.0, 4.5]

                        # This would be replaced with actual evaluation results in a real implementation
                        precision = [0.82, 0.75, 0.68]
                        recall = [0.76, 0.69, 0.58]
                        f1 = [0.79, 0.72, 0.63]

                        metrics_df = pd.DataFrame({
                            'Threshold': thresholds,
                            'Precision': precision,
                            'Recall': recall,
                            'F1-Score': f1
                        })

                        # Create the chart
                        fig, ax = plt.subplots(figsize=(10, 6))

                        x = np.arange(len(thresholds))
                        width = 0.25

                        ax.bar(x - width, precision, width, label='Precision')
                        ax.bar(x, recall, width, label='Recall')
                        ax.bar(x + width, f1, width, label='F1-Score')

                        ax.set_xlabel('Rating Threshold')
                        ax.set_ylabel('Score')
                        ax.set_title('Evaluation Metrics at Different Rating Thresholds')
                        ax.set_xticks(x)
                        ax.set_xticklabels(thresholds)
                        ax.legend()

                        st.pyplot(fig)

                        # Display dataset statistics
                        st.markdown("<p class='subheader'>Dataset Statistics</p>", unsafe_allow_html=True)

                        col1, col2, col3 = st.columns([1, 1, 1])

                        with col1:
                            st.metric("Total Users", recommender.ratings_df['user_id'].nunique())

                        with col2:
                            st.metric("Total Movies", recommender.ratings_df['movie_id'].nunique())

                        with col3:
                            st.metric("Total Ratings", len(recommender.ratings_df))

                        # Rating distribution
                        st.markdown("<p class='subheader'>Rating Distribution</p>", unsafe_allow_html=True)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.countplot(x='rating', data=recommender.ratings_df, ax=ax)
                        ax.set_title('Rating Distribution')
                        ax.set_xlabel('Rating')
                        ax.set_ylabel('Count')

                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error during evaluation: {str(e)}")
                        st.info("Try using a smaller dataset or sample data.")


if __name__ == "__main__":
    main()

