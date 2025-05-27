import numpy as np
import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
from geopy.distance import geodesic
from mimesis import Generic
import random

np.random.seed(42)
random.seed(42)
# Initialize Faker
fake = Faker()
generic = Generic('en')

# City coordinates and probabilities
cities = ['New York', 'London', 'Paris', 'Tokyo', 'Sydney', 'Berlin', 'Mumbai', 'São Paulo', 'Toronto', 'Dubai']
city_probs = [0.2, 0.15, 0.1, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05]

def calculate_time_weight(interaction_time, current_time, half_life=30):
    time_diff = (current_time - interaction_time).days
    return np.exp(np.log(0.5) * time_diff / half_life)

def match_interests(user_interests, event_type):
    """Calculate interest match score based on user interests and event type."""
    interest_list = user_interests.split(',')
    event_type_lower = event_type.lower()
    interest_mapping = {
        'music': ['music & concerts'],
        'sports': ['sports & fitness'],
        'tech': ['technology', 'business & networking'],
        'food': ['food & drink'],
        'art': ['arts & culture'],
        'literature': ['education & learning'],
        'cinema': ['entertainment'],
        'travel': ['seasonal & festivals'],
        'fitness': ['sports & fitness', 'health & wellness'],
        'fashion': ['arts & culture']
    }
    score = 0
    for interest in interest_list:
        if interest in interest_mapping:
            if event_type_lower in [e.lower() for e in interest_mapping[interest]]:
                score += 1
    return score / max(1, len(interest_list))  # Normalize by number of interests

def generate_interactions(users, events, n_interactions=500000):
    interactions = []
    interaction_types = ['maybe', 'invited & maybe', 'no', 'yes', 'invited & yes', 'invited & no', 'invited']
    current_time = datetime(2025, 5, 25, 11, 42)  # Updated to current date

    # Precompute user and event indices for stratified sampling
    users_per_city = {city: users[users['user_city'] == city].index for city in cities}
    events_per_city = {city: events[events['event_city'] == city].index for city in cities}

    # Target ~10 interactions per user
    target_interactions_per_user = n_interactions // len(users)  # ~10
    user_interaction_counts = {user_id: 0 for user_id in users['user_id']}

    while len(interactions) < n_interactions:
        # Prefer same-city interactions to increase local events
        city = np.random.choice(cities, p=city_probs)
        user_idx = np.random.choice(users_per_city[city])
        user = users.loc[user_idx]

        # 80% chance of same-city event, 20% chance of any event
        if random.random() < 0.8:
            if len(events_per_city[city]) > 0:
                event_idx = np.random.choice(events_per_city[city])
            else:
                event_idx = np.random.choice(events.index)
        else:
            event_idx = np.random.choice(events.index)
        event = events.loc[event_idx]

        # Calculate distance
        distance = geodesic((user['user_lat'], user['user_lon']),
                           (event['event_lat'], event['event_lon'])).km

        # Relaxed distance constraint
        max_distance = 100 if random.random() < 0.8 else 500  # Increased to allow more interactions

        # Compute interaction probability with interest matching
        distance_score = np.exp(-distance / 50)  # Softer decay
        weather_score = 1.5 if (event['weather_condition'] == 'Clear' and
                               user['indoor_outdoor_preference'] in ['outdoor', 'any']) or \
                              (event['event_indoor_capability'] and
                               event['weather_condition'] in ['Rain', 'Snow']) else 0.8
        social_score = np.log1p(user['social_connectedness']) / 5  # Increased weight
        interest_score = match_interests(user['user_interests'], event['event_type'])  # New interest matching

        interaction_prob = 0.5 * distance_score + 0.2 * weather_score + 0.1 * social_score + 0.2 * interest_score

        # Relaxed probability threshold and user interaction limit
        if distance < max_distance and random.random() < min(interaction_prob * 1.5, 0.9) and \
           user_interaction_counts[user['user_id']] < target_interactions_per_user * 2:
            interaction_time = fake.date_time_between(
                start_date=event['start_time'] - timedelta(days=30),
                end_date=event['start_time']
            )
            time_weight = calculate_time_weight(interaction_time, current_time)
            interaction_prob *= time_weight

            # Adjusted interaction type probabilities based on distance and interest
            if distance <= 10:
                interaction_type_probs = [0.1, 0.15, 0.05, 0.35, 0.25, 0.05, 0.05]
            elif distance <= 50:
                interaction_type_probs = [0.15, 0.15, 0.1, 0.25, 0.2, 0.1, 0.05]
            elif distance <= 100:
                interaction_type_probs = [0.2, 0.1, 0.15, 0.2, 0.15, 0.15, 0.05]
            else:
                interaction_type_probs = [0.25, 0.05, 0.2, 0.15, 0.1, 0.2, 0.05]

            # Bias toward positive interactions for relevant events
            if interest_score > 0.5:
                interaction_type_probs = [p * 1.5 if t in ['yes', 'invited & yes'] else p * 0.7
                                         for p, t in zip(interaction_type_probs, interaction_types)]
                interaction_type_probs = [p / sum(interaction_type_probs) for p in interaction_type_probs]

            interactions.append({
                'interaction_id': generic.person.identifier(mask='@@###@'),
                'user_id': user['user_id'],
                'event_id': event['event_id'],
                'interaction_type': np.random.choice(interaction_types, p=interaction_type_probs),
                'interaction_time': interaction_time,
                'interaction_distance_to_event': distance
            })
            user_interaction_counts[user['user_id']] += 1

    return pd.DataFrame(interactions)

    
    
    
   