import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import faiss
import requests
import random
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from catboost import Pool

# Set page config
st.set_page_config(
    page_title="Weather-Aware Event Recommender",
    page_icon="🌦️",
    layout="wide"
)

# --- Helper Functions ---

def get_random_user_profile(users_df):
    """Select a random user profile from users_df"""
    random_user = users_df.sample(1).iloc[0]
    return {
        "user_id": random_user["user_id"],
        "interests": random_user["user_interests"],
        "gender": random.choice(["male", "female", "other"]),  # Assuming gender not in df
        "age": random_user["age"],
        "lat": random_user.get("user_lat", None),
        "lon": random_user.get("user_lon", None),
        "event_type_preference": random_user.get("user_weather_preference", "any")
    }

def fetch_weather_open_meteo(lat, lon, date=None):
    """Fetch weather data from open-meteo API given lat, lon, date"""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["weathercode", "temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "start_date": date,
        "end_date": date,
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "daily" in data:
                daily = data["daily"]
                weathercode = daily["weathercode"][0]
                weather_condition = map_weathercode_to_condition(weathercode)
                temperature_max = daily["temperature_2m_max"][0]
                temperature_min = daily["temperature_2m_min"][0]
                temperature = (temperature_max + temperature_min) / 2
                precipitation = daily["precipitation_sum"][0]
                
                return {
                    "weather_condition": weather_condition,
                    "temperature": temperature,
                    "precipitation_sum": precipitation
                }
    except Exception as e:
        st.error(f"Error fetching weather data: {e}")
    
    # Default values if API call fails
    return {
        "weather_condition": "Unknown",
        "temperature": 20.0,
        "precipitation_sum": 0.0
    }

def map_weathercode_to_condition(code):
    """Map open-meteo weather code to condition string"""
    mapping = {
        0: "Clear",
        1: "Mainly Clear",
        2: "Partly Cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing Rime Fog",
        51: "Light Drizzle",
        53: "Moderate Drizzle",
        55: "Dense Drizzle",
        56: "Light Freezing Drizzle",
        57: "Dense Freezing Drizzle",
        61: "Slight Rain",
        63: "Moderate Rain",
        65: "Heavy Rain",
        66: "Light Freezing Rain",
        67: "Heavy Freezing Rain",
        71: "Slight Snow Fall",
        73: "Moderate Snow Fall",
        75: "Heavy Snow Fall",
        77: "Snow Grains",
        80: "Slight Rain Showers",
        81: "Moderate Rain Showers",
        82: "Violent Rain Showers",
        85: "Slight Snow Showers",
        86: "Heavy Snow Showers",
        95: "Thunderstorm",
        96: "Thunderstorm with Slight Hail",
        99: "Thunderstorm with Heavy Hail"
    }
    return mapping.get(code, "Unknown")

def ensure_deployment_running(ms, deployment_name):
    """Ensure a deployment is running before using it"""
    deployment = ms.get_deployment(deployment_name)
    
    # Check if deployment exists
    if deployment is None:
        st.error(f"Deployment '{deployment_name}' does not exist!")
        return None
    
    state = deployment.get_state()
    print(f"Deployment '{deployment_name}' state: {state}")
    
    if not deployment.is_running():
        print(f"Starting deployment '{deployment_name}'...")
        deployment.start()
        
        # Wait for deployment to start (with timeout)
        max_retries = 12
        for i in range(max_retries):
            time.sleep(10)  # Wait 10 seconds between checks
            state = deployment.get_state()
            print(f"Check {i+1}/{max_retries}: {state}")
            
            if deployment.is_running():
                print(f"Deployment '{deployment_name}' is now running!")
                return deployment
        
        print(f"WARNING: Deployment '{deployment_name}' did not reach running state within timeout")
    else:
        print(f"Deployment '{deployment_name}' is already running")
    
    return deployment

# Connect to Hopsworks
@st.cache_resource
def connect_to_hopsworks():
    project = hopsworks.login()
    return project

# Load models and feature data
@st.cache_resource
def load_resources():
    project = connect_to_hopsworks()
    fs = project.get_feature_store()
    ms = project.get_model_serving()
    
    # Get feature views and feature groups
    users_fg = fs.get_feature_group("users", version=1)
    events_fg = fs.get_feature_group("events", version=1)
    event_embeddings_fg = fs.get_feature_group("event_embeddings", version=1)
    
    # Get deployed models
    try:
        query_deployment = ms.get_deployment("querymodel")
        query_predictor = ensure_deployment_running(ms, "querymodel")
        ranking_deployment = ms.get_deployment("weathermodel")
        ranking_predictor = ensure_deployment_running(ms, "weathermodel")
    except Exception as e:
        st.warning(f"Model deployment issue: {e}. Using mock predictions for demo.")
        query_predictor = None
        ranking_predictor = None
    
    # Get feature data
    users_df = users_fg.read()
    events_df = events_fg.read()
    
    # Get precomputed event embeddings
    try:
        event_embeddings_df = event_embeddings_fg.read()
    except:
        st.warning("Could not load event embeddings. Using mock embeddings for demo.")
        event_embeddings_df = pd.DataFrame({
            "event_id": events_df["event_id"],
            "embedding": [np.random.rand(64).tolist() for _ in range(len(events_df))]
        })
    
    return {
        "project": project,
        "fs": fs,
        "ms": ms,
        "users_df": users_df,
        "events_df": events_df,
        "event_embeddings_df": event_embeddings_df,
        "query_predictor": query_predictor,
        "ranking_predictor": ranking_predictor
    }

# Initialize resources
try:
    resources = load_resources()
    users_df = resources["users_df"]
    events_df = resources["events_df"]
    event_embeddings_df = resources["event_embeddings_df"]
    query_predictor = resources["query_predictor"]
    ranking_predictor = resources["ranking_predictor"]
except Exception as e:
    st.error(f"Error connecting to Hopsworks: {e}")
    st.stop()

# Build FAISS index from precomputed embeddings
@st.cache_resource
def build_faiss_index():
    # Extract embeddings from DataFrame
    event_ids = event_embeddings_df["event_id"].values
    embeddings = np.array(event_embeddings_df["embedding"].tolist())
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    return index, event_ids

# Get user embedding from query model
def get_user_embedding(user_data):
    # If we don't have a working query model, use random embeddings for demo
    if query_predictor is None:
        return np.random.rand(64), user_data
    
    # Prepare input for query model
    query_input = {
        "instances": [
            [
                user_data["user_id"],
                user_data["user_city"],
                float(user_data["age"]),
                user_data["user_interests"]
            ]
        ]
    }
    
    # Get embedding from deployed model
    try:
        response = query_predictor.predict(query_input)
        user_embedding = np.array(response["predictions"][0])
    except Exception as e:
        st.warning(f"Error getting user embedding: {e}. Using random embedding for demo.")
        user_embedding = np.random.rand(64)
    
    return user_embedding, user_data

# Retrieve top-k candidates using FAISS
def retrieve_candidates(user_embedding, k=100):
    # Normalize user embedding
    user_emb_norm = user_embedding.copy().reshape(1, -1)
    faiss.normalize_L2(user_emb_norm)
    
    # Search index
    index, event_ids = build_faiss_index()
    distances, indices = index.search(user_emb_norm, k)
    
    # Get candidate event IDs and similarity scores
    candidate_ids = [event_ids[idx] for idx in indices[0]]
    similarity_scores = distances[0]
    
    return candidate_ids, similarity_scores

# Rank candidates using the ranking model
def rank_candidates(user_data, candidate_ids, similarity_scores, use_weather=True):
    # Get candidate events data
    candidates_df = events_df[events_df["event_id"].isin(candidate_ids)].copy()
    
    if len(candidates_df) == 0:
        return pd.DataFrame()
    
    # Add retrieval similarity scores
    id_to_score = {id: score for id, score in zip(candidate_ids, similarity_scores)}
    candidates_df["retrieval_score"] = candidates_df["event_id"].map(id_to_score)
    
    # Add weather data if needed
    if use_weather:
        # Add weather data for events that don't have it
        for idx, event in candidates_df.iterrows():
            if pd.isna(event["weather_condition"]) or event["weather_condition"] == "unknown":
                # Extract event date from start_time or use current date
                event_date = datetime.now().strftime("%Y-%m-%d")
                if "start_time" in event and not pd.isna(event["start_time"]):
                    try:
                        event_date = pd.to_datetime(event["start_time"]).strftime("%Y-%m-%d")
                    except:
                        pass
                
                # Fetch weather data for event location and date
                weather_data = fetch_weather_open_meteo(
                    event["event_lat"], 
                    event["event_lon"],
                    event_date
                )
                
                # Update event with weather data
                candidates_df.loc[idx, "weather_condition"] = weather_data["weather_condition"]
                candidates_df.loc[idx, "temperature"] = weather_data["temperature"]
                candidates_df.loc[idx, "precipitation_sum"] = weather_data["precipitation_sum"]
    
    # Prepare features for ranking
    ranking_features = []
    for _, event in candidates_df.iterrows():
        # Create feature dict for this user-event pair
        features = {
            "interaction_distance_to_event": calculate_distance(
                user_data["user_lat"], user_data["user_lon"],
                event["event_lat"], event["event_lon"]
            ),
            "event_type": event["event_type"],
            "event_city": event["event_city"],
            "weather_condition": event["weather_condition"],
            "temperature": event["temperature"],
            "attendance_rate": event["attendance_rate"],
            "event_indoor_capability": event["event_indoor_capability"],
            "user_city": user_data["user_city"],
            "user_weather_preference": user_data["user_weather_preference"],
            "age": user_data["age"],
            "user_interests": user_data["user_interests"]
        }
        ranking_features.append(features)
    
    # Convert to DataFrame
    ranking_df = pd.DataFrame(ranking_features)
    
    # Get ranking scores from model
    if ranking_predictor is None:
        # If no ranking model, use retrieval scores
        ranking_scores = candidates_df["retrieval_score"].values
    else:
        try:
            # Format input for ranking model
            ranking_input = {
                "instances": ranking_df.to_dict("records")
            }
            response = ranking_predictor.predict(ranking_input)
            ranking_scores = np.array(response["predictions"])
        except Exception as e:
            st.warning(f"Error getting ranking scores: {e}. Using retrieval scores for demo.")
            ranking_scores = candidates_df["retrieval_score"].values
    
    # Add ranking scores to candidates
    candidates_df["ranking_score"] = ranking_scores
    
    # Sort by ranking score
    ranked_candidates = candidates_df.sort_values("ranking_score", ascending=False)
    
    return ranked_candidates

# Calculate distance between coordinates
def calculate_distance(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    
    # Convert decimal degrees to radians
    try:
        lon1, lat1, lon2, lat2 = map(float, [lon1, lat1, lon2, lat2])
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r
    except:
        return 10.0  # Default distance if calculation fails

# Compute weather match score
def compute_weather_match(user_weather, event_weather, user_pref, event_type):
    # Base score
    score = 0.0
    
    # Weather group match (0.4 points)
    if user_weather == event_weather:
        score += 0.4
    elif (user_weather in ['Sunny', 'Cloudy', 'Clear', 'Partly Cloudy'] and
          event_weather in ['Sunny', 'Cloudy', 'Clear', 'Partly Cloudy']):
        score += 0.2
    elif (user_weather in ['Rain', 'Light Drizzle', 'Heavy Rain', 'Thunderstorm'] and
          event_weather in ['Rain', 'Light Drizzle', 'Heavy Rain', 'Thunderstorm']):
        score += 0.2
    
    # Indoor/outdoor preference match (0.6 points)
    if user_pref == 'any':
        score += 0.6
    elif (user_pref == 'indoor' and event_type in ['Indoor', 'Hybrid']) or \
         (user_pref == 'outdoor' and event_type in ['Outdoor', 'Hybrid']):
        score += 0.6
    
    return round(score, 2)

# --- UI Components ---
st.title("🌦️ Weather-Aware Event Recommendation System")
st.write("**Get personalized event recommendations based on your preferences and local weather conditions.**")

# User information input form - now as a single tab
with st.container():
    with st.form("user_info_form"):
        st.subheader("Enter Your Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            interests = st.text_input("Interests (comma separated)", "music, sports, art")
            gender = st.selectbox("Gender", ["male", "female", "other"])
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
        
        with col2:
            location_method = st.radio("Location Input Method", ["Coordinates", "City"])
            
            if location_method == "Coordinates":
                lat = st.number_input("Latitude", value=52.52, format="%.4f")
                lon = st.number_input("Longitude", value=13.41, format="%.4f")
                city = "Unknown"
            else:
                city = st.text_input("City", "Berlin")
                # Use a default location for the city
                lat = 52.52
                lon = 13.41
        
        event_type_pref = st.selectbox("Event Type Preference", ["any", "indoor", "outdoor"])
        
        # Weather toggle
        use_weather_form = st.checkbox("Use Weather-Based Recommendations", value=True)
        
        submit_button = st.form_submit_button("Get Recommendations")

# Sidebar for user selection and filters
with st.sidebar:
    st.markdown("# 🔍 Find Events")
    
    # Add a divider and some spacing
    st.markdown("---")
    st.markdown("Select Existing User")
    # User selection with a cleaner look
    st.markdown("### 👤 User")
    
    # Create a dictionary mapping user_ids to their interests for quick lookup
    user_interests_dict = dict(zip(users_df["user_id"], users_df["user_interests"]))
    user_location_dict = dict(zip(users_df["user_id"], users_df["user_city"]))
    # Create a formatted list of users with their interests
    user_options = [f"{uid} - {user_interests_dict[uid][:20]}..." if len(user_interests_dict[uid]) > 20 
                   else f"{uid} - {user_interests_dict[uid]}" for uid in users_df["user_id"].unique()]
    
    # Display the formatted selectbox
    selected_user_option = st.selectbox("Select User", user_options, label_visibility="collapsed")
    
    # Extract just the user_id from the selection
    user_id = selected_user_option.split(" - ")[0]
    
    # Show full interests in an expandable section
    with st.expander("User Details"):
        st.write(f"**User ID:** {user_id}")
        st.write(f"**Interests:** {user_interests_dict[user_id]}")
        st.write(f"**location:** {user_location_dict[user_id]}")
    
    st.markdown("### ⚙️ Filters")
    
    # Weather filtering with toggle
    use_weather = st.toggle("Weather-Based Recommendations", value=True)
    
    # Distance filtering with slider
    st.markdown("#### Distance")
    max_distance = st.slider("", 5, 500, 100, help="Maximum distance in kilometers")
    
    # Add some space
    st.markdown("---")
    
    # Get recommendations button with custom styling
    recommend_button = st.button("Find My Events", type="primary", use_container_width=True)

# Random user button (outside tabs)
random_user_col1, random_user_col2 = st.columns([1, 3])
with random_user_col1:
    random_user_button = st.button("Try Random User",type="primary", use_container_width=True)
with random_user_col2:
    st.info("Don't want to enter information? Click to get recommendations for a random user profile.")

# Process user input and generate recommendations
if recommend_button or submit_button or random_user_button:
    with st.spinner("Generating recommendations...",type="primary", use_container_width=True):
        # Determine which input method was used
        if random_user_button:
            # Use a random user profile
            random_profile = get_random_user_profile(users_df)
            user_data = users_df[users_df["user_id"] == random_profile["user_id"]].iloc[0].to_dict()
            use_weather = True
            st.success(f"Using random user profile: {user_data['user_id']}")
            
        elif submit_button:
            # Create a user data dict from form inputs
            user_data = {
                "user_id": f"custom_{int(datetime.now().timestamp())}",
                "user_interests": interests,
                "age": age,
                "user_lat": lat,
                "user_lon": lon,
                "user_city": city,
                "user_weather_preference": event_type_pref
            }
            use_weather = use_weather_form
            
        else:  # sidebar recommend_button
            # Use selected user from sidebar
            user_data = users_df[users_df["user_id"] == user_id].iloc[0].to_dict()
        
        # Get weather for user location if using weather
        if use_weather:
            user_weather = fetch_weather_open_meteo(
                user_data["user_lat"], 
                user_data["user_lon"]
            )
            user_data["user_weather_condition"] = user_weather["weather_condition"]
            user_data["user_temperature"] = user_weather["temperature"]
            user_data["user_precipitation"] = user_weather["precipitation_sum"]
        
        # Get user embedding
        user_embedding, user_data = get_user_embedding(user_data)
        
        # Retrieve candidates
        candidate_ids, similarity_scores = retrieve_candidates(user_embedding, k=100)
        
        # Rank candidates
        ranked_events = rank_candidates(user_data, candidate_ids, similarity_scores, use_weather=use_weather)
        
        # Apply distance filter
        ranked_events = ranked_events[ranked_events.apply(
            lambda x: calculate_distance(
                user_data["user_lat"], user_data["user_lon"],
                x["event_lat"], x["event_lon"]
            ) <= max_distance, 
            axis=1
        )]
        
        # Add weather match score if using weather filtering
        if use_weather:
            ranked_events["weather_match"] = ranked_events.apply(
                lambda x: compute_weather_match(
                    user_data.get("user_weather_condition", "unknown"),
                    x["weather_condition"],
                    user_data["user_weather_preference"],
                    x["event_type"]
                ),
                axis=1
            )
            
            # Boost ranking score with weather match
            ranked_events["final_score"] = 0.7 * ranked_events["ranking_score"] + 0.3 * ranked_events["weather_match"]
            ranked_events = ranked_events.sort_values("final_score", ascending=False)
        else:
            ranked_events["final_score"] = ranked_events["ranking_score"]
    
    # Display user info
    st.subheader("User Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Interests:** {user_data['user_interests']}")
        st.write(f"**Age:** {user_data['age']}")
        st.write(f"**Location:** {user_data['user_city']}")
    
    with col2:
        st.write(f"**Event Type Preference:** {user_data['user_weather_preference']}")
        if use_weather and 'user_weather_condition' in user_data:
            st.write(f"**Current Weather:** {user_data['user_weather_condition']}")
            st.write(f"**Temperature:** {user_data.get('user_temperature', 'N/A')}°C")
    
    with col3:
        st.write(f"**Weather-Based Filtering:** {'Enabled' if use_weather else 'Disabled'}")
        st.write(f"**Max Distance:** {max_distance} km")
    
    # Display recommendations
    if len(ranked_events) > 0:
        st.subheader("Top Event Recommendations")
        
        # Display top recommendations as cards
        for i, (_, event) in enumerate(ranked_events.head(10).iterrows()):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image("https://via.placeholder.com/150", width=150)
            
            with col2:
                st.subheader(event["title"])
                st.write(f"**Type:** {event['event_type']} | **City:** {event['event_city']}")
                
                if use_weather:
                    st.write(f"**Weather:** {event['weather_condition']} | **Temperature:** {event['temperature']}°C")
                    st.write(f"**Precipitation:** {event.get('precipitation_sum', 0):.1f} mm")
                
                st.write(f"**Distance:** {calculate_distance(user_data['user_lat'], user_data['user_lon'], event['event_lat'], event['event_lon']):.1f} km")
                
                # Show scores
                score_col1, score_col2, score_col3 = st.columns(3)
                with score_col1:
                    st.metric("Ranking Score", f"{event['ranking_score']:.2f}")
                with score_col2:
                    if use_weather:
                        st.metric("Weather Match", f"{event['weather_match']:.2f}")
                with score_col3:
                    st.metric("Final Score", f"{event['final_score']:.2f}")
                
                st.divider()
        
        # Visualizations
        st.subheader("Recommendation Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Score Distribution", "Event Types", "Weather Impact"])
        
        with tab1:
            # Score distribution
            fig = px.histogram(
                ranked_events, 
                x="final_score", 
                nbins=20,
                title="Distribution of Recommendation Scores"
            )
            st.plotly_chart(fig)
        
        with tab2:
            # Event type distribution
            fig = px.pie(
                ranked_events.head(20), 
                names="event_type",
                title="Event Types in Top 20 Recommendations"
            )
            st.plotly_chart(fig)
        
        with tab3:
            if use_weather:
                # Weather impact visualization
                fig = px.scatter(
                    ranked_events.head(50),
                    x="ranking_score",
                    y="weather_match",
                    size="final_score",
                    color="event_type",
                    hover_name="title",
                    title="Impact of Weather on Recommendations"
                )
                st.plotly_chart(fig)
            else:
                st.info("Weather filtering is disabled. Enable it to see weather impact analysis.")
    else:
        st.warning("No recommendations found matching your criteria. Try adjusting your preferences or location.")

# Footer
st.divider()
st.write("*Powered by Hopsworks, TensorFlow Recommenders, and Open-Meteo Weather API*")
