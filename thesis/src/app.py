import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import faiss
from catboost import Pool
import matplotlib.pyplot as plt

# --- Load Data and Models ---
@st.cache_resource
def load_data_and_models():
    users_df = pd.read_csv('test_users_data.csv')
    events_df = pd.read_csv("test_events_data.csv")
    interactions_df = pd.read_csv('test_interactions_data.csv')
    # Rename columns as in your pipeline
    users_df.rename(columns={
        'lat': 'user_lat', 'lng': 'user_lon', 'location': 'user_city',
        'indoor_outdoor_preference': 'user_weather_preference', 'joinedAt': 'signup_date'
    }, inplace=True)
    events_df.rename(columns={
        'category': 'event_type', 'lat': 'event_lat', 'lng': 'event_lon',
        'city': 'event_city', 'weather_description': 'weather_condition',
        'temperature_2m_mean': 'temperature'
    }, inplace=True)
    interactions_df.rename(columns={
        'distance_to_event': 'interaction_distance_to_event'
    }, inplace=True)
    # Load models
    query_model = tf.keras.models.load_model('query_model')
    candidate_model = tf.keras.models.load_model('candidate_model')
    ranking_model = joblib.load('weather_ranking_model.pkl')
    return users_df, events_df, interactions_df, query_model, candidate_model, ranking_model

users_df, events_df, interactions_df, query_model, candidate_model, ranking_model = load_data_and_models()

# --- Helper Functions (adapt from your pipeline) ---
def get_user_embeddings(query_model, user_ids):
    batch_users = users_df[users_df['user_id'].isin(user_ids)]
    batch_input = {
        "user_id": tf.constant(batch_users['user_id'].values),
        "user_city": tf.constant(batch_users['user_city'].values),
        "age": tf.constant(batch_users['age'].values, dtype=tf.float32),
        "user_interests": tf.constant(batch_users['user_interests'].values)
    }
    return query_model(batch_input).numpy()

def get_event_embeddings(candidate_model, events_data=None):
    if events_data is None:
        events_data = events_df.copy()
    events_dataset = tf.data.Dataset.from_tensor_slices({
        "event_id": tf.constant(events_data["event_id"].values),
        "event_city": tf.constant(events_data["event_city"].values),
        "event_type": tf.constant(events_data["event_type"].values),
        "title": tf.constant(events_data["title"].values)
    }).batch(128)
    event_ids = []
    event_embeddings = []
    for batch in events_dataset:
        batch_embeddings = candidate_model(batch)
        event_ids.extend(batch["event_id"].numpy())
        event_embeddings.append(batch_embeddings.numpy())
    return np.vstack(event_embeddings), [str(eid) for eid in event_ids]

def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

# --- Streamlit UI ---
st.title("Weather-Aware Event Recommendation System")

# User selection
user_id = st.selectbox("Select User ID", users_df['user_id'].unique().astype(str))

if st.button("Get Recommendations"):
    # 1. Retrieve user embedding
    user_embedding = get_user_embeddings(query_model, [user_id])
    faiss_index = build_faiss_index(get_event_embeddings(candidate_model)[0])
    event_ids = get_event_embeddings(candidate_model)[1]
    faiss.normalize_L2(user_embedding)
    distances, indices = faiss_index.search(user_embedding, 100)
    retrieved_event_ids = [event_ids[idx] for idx in indices[0]]
    retrieved_events_df = events_df[events_df['event_id'].isin(retrieved_event_ids)].copy()

    # 2. Prepare ranking features
    user_data = users_df[users_df['user_id'] == user_id].iloc[0]
    ranking_data = []
    for _, event in retrieved_events_df.iterrows():
        row = {
            'event_id': event['event_id'],
            'event_type': event['event_type'],
            'event_city': event['event_city'],
            'attendance_rate': event['attendance_rate'],
            'event_indoor_capability': event['event_indoor_capability'],
            'user_city': user_data['user_city'],
            'age': user_data['age'],
            'user_interests': user_data['user_interests'],
            'weather_condition': event['weather_condition'],
            'temperature': event['temperature'],
            'user_weather_preference': user_data['user_weather_preference'],
            'interaction_distance_to_event': np.nan  # Fill as needed
        }
        ranking_data.append(row)
    ranking_df = pd.DataFrame(ranking_data)
    # Fill NaNs
    for col in ranking_df.columns:
        if ranking_df[col].dtype == 'object':
            ranking_df[col] = ranking_df[col].fillna('unknown')
        else:
            ranking_df[col] = ranking_df[col].fillna(0.0)
    cat_features = ranking_df.select_dtypes(include=["object", "bool"]).columns.tolist()
    ranking_pool = Pool(data=ranking_df, cat_features=cat_features)
    ranking_scores = ranking_model.predict_proba(ranking_pool)[:, 1]
    retrieved_events_df['ranking_score'] = ranking_scores

    # 3. Show recommendations
    top_recs = retrieved_events_df.sort_values('ranking_score', ascending=False).head(10)
    st.subheader("Top Event Recommendations")
    st.dataframe(top_recs[['title', 'event_type', 'event_city', 'start_time', 'weather_condition', 'ranking_score']])

    # 4. Optionally, show score distribution
    st.subheader("Ranking Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(ranking_scores, bins=20, alpha=0.7)
    ax.set_xlabel("Ranking Score")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 5. Optionally, compare with a baseline (e.g., popularity)
    st.subheader("Compare with Popularity Baseline")
    pop_scores = retrieved_events_df['attendance_rate'].values
    pop_top = retrieved_events_df.assign(pop_score=pop_scores).sort_values('pop_score', ascending=False).head(10)
    st.dataframe(pop_top[['title', 'event_type', 'event_city', 'start_time', 'weather_condition', 'pop_score']])

st.markdown("""
---
*Powered by Streamlit and your weather-aware event recommendation engine.*
""")
