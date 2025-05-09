# import numpy as np
# import pandas as pd
# import random
# from faker import Faker
# from datetime import datetime, timedelta
# from geopy.distance import geodesic
# from scipy.stats import skewnorm, dirichlet
# import hopsworks

# # Initialize Faker and set seed for reproducibility
# fake = Faker()
# np.random.seed(42)
# random.seed(42)

# # City coordinates and probabilities
# cities = ['New York', 'London', 'Paris', 'Tokyo', 'Sydney', 'Berlin', 'Mumbai', 'São Paulo', 'Toronto', 'Dubai']
# city_probs = [0.2, 0.15, 0.1, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05]
# city_coords = {
#     'New York': (40.7128, -74.0060), 'London': (51.5074, -0.1278), 'Paris': (48.8566, 2.3522),
#     'Tokyo': (35.6762, 139.6503), 'Sydney': (-33.8688, 151.2093), 'Berlin': (52.5200, 13.4050),
#     'Mumbai': (19.0760, 72.8777), 'São Paulo': (-23.5505, -46.6333), 'Toronto': (43.6532, -79.3832),
#     'Dubai': (25.2048, 55.2708)
# }

# def generate_location(city):
#     base_lat, base_lon = city_coords[city]
#     if random.random() < 0.8:
#         lat = base_lat + np.random.uniform(-0.1, 0.1)
#         lon = base_lon + np.random.uniform(-0.1, 0.1)
#     else:
#         lat = base_lat + np.random.uniform(-2, 2)
#         lon = base_lon + np.random.uniform(-2, 2)
#     return lat, lon

# def generate_users(n_users=20000):
#     users = []
#     interests = ['music', 'sports', 'tech', 'food', 'art', 'literature', 'cinema', 'travel', 'fitness', 'fashion']
    
#     for _ in range(n_users):
#         city = np.random.choice(cities, p=city_probs)
#         lat, lon = generate_location(city)
#         age = max(18, min(100, int(skewnorm.rvs(5, loc=25, scale=15))))
#         weather_probs = dirichlet.rvs([0.3, 0.5, 0.2])[0]
        
#         users.append({
#             'user_id': fake.uuid4(),
#             'location_lat': lat,
#             'location_lon': lon,
#             'city': city,
#             'weather_preference': np.random.choice(['indoor', 'outdoor', 'any'], p=weather_probs),
#             'age': age,
#             'declared_interests': random.sample(interests, k=random.randint(0, min(4, len(interests)))) if random.random() < 0.7 else [],
#             'signup_date': fake.date_time_between(start_date='-2y', end_date='now'),
#             'social_connectedness': np.random.poisson(lam=15)
#         })
#     return pd.DataFrame(users)

# def generate_events(n_events=5000):
#     events = []
#     event_types = [
#         'Education & Learning', 'Technology', 'Seasonal & Festivals', 'Arts & Culture', 
#         'Entertainment', 'Sports & Fitness', 'Business & Networking', 'Health & Wellness', 
#         'Music & Concerts', 'Food & Drink', 'Community & Causes', 'Immersive Experiences'
#     ]
#     weather_conditions = ['Clear', 'Rain', 'Snow', 'Cloudy', 'Windy']
#     weather_probs = [0.5, 0.2, 0.05, 0.2, 0.05]
    
#     current_date = datetime(2025, 3, 26, 19, 42)  # Current date and time
    
#     for _ in range(n_events):
#         event_type = np.random.choice(event_types)
#         city = np.random.choice(cities, p=city_probs)
#         lat, lon = generate_location(city)
        
#         if event_type in ['Sports & Fitness', 'Seasonal & Festivals']:
#             weather_condition = 'Clear' if random.random() < 0.8 else np.random.choice(['Rain', 'Cloudy'])
#         elif event_type in ['Education & Learning', 'Technology', 'Business & Networking']:
#             weather_condition = np.random.choice(['Clear', 'Cloudy'])
#         else:
#             weather_condition = np.random.choice(weather_conditions, p=weather_probs)
        
#         base_temp = {
#             'New York': 15, 'London': 12, 'Paris': 16, 'Tokyo': 20, 
#             'Sydney': 22, 'Berlin': 14, 'Mumbai': 28, 'São Paulo': 24, 
#             'Toronto': 10, 'Dubai': 32
#         }[city]
        
#         temp_adjustment = {
#             'Clear': np.random.uniform(2, 5),
#             'Rain': np.random.uniform(-3, 0),
#             'Snow': np.random.uniform(-8, -3),
#             'Cloudy': np.random.uniform(-1, 2),
#             'Windy': np.random.uniform(-2, 1)
#         }[weather_condition]
        
#         temperature = round(base_temp + temp_adjustment, 1)
        
#         start_time = fake.date_time_between(start_date=current_date, end_date=current_date + timedelta(days=180))
#         is_weekend = start_time.weekday() >= 5
#         hour_choices = [10, 14, 18] if is_weekend else [9, 13, 18, 19]
#         start_time = start_time.replace(hour=np.random.choice(hour_choices))
        
#         events.append({
#             'event_id': fake.uuid4(),
#             'title': f"{fake.catch_phrase()} {event_type} in {city}",
#             'event_type': event_type,
#             'lat': lat,
#             'lon': lon,
#             'event_city': city,
#             'start_time': start_time,
#             'duration': np.random.choice([120, 180, 240, 360, 480]),
#             'weather_condition': weather_condition,
#             'temperature': temperature,
#             'historical_attendance_rate': np.random.beta(a=2, b=5) * 100,
#             'indoor_capability': event_type in ['Education & Learning', 'Technology', 'Business & Networking', 
#                                                'Arts & Culture', 'Entertainment', 'Immersive Experiences']
#         })
#     return pd.DataFrame(events)

# def calculate_time_weight(interaction_time, current_time, half_life=30):
#     time_diff = (current_time - interaction_time).days
#     return np.exp(np.log(0.5) * time_diff / half_life)

# def generate_interactions(users, events, n_interactions=100000):
#     interactions = []
#     interaction_types = ['maybe', 'invited & maybe', 'no', 'yes', 'invited & yes', 'invited & no', 'invited']
#     attempts = n_interactions * 5
#     current_time = datetime(2025, 3, 26, 19, 42)
    
#     for _ in range(attempts):
#         if len(interactions) >= n_interactions:
#             break
            
#         user = users.sample(1).iloc[0]
#         event = events.sample(1).iloc[0]
        
#         distance = geodesic((user['location_lat'], user['location_lon']), 
#                            (event['lat'], event['lon'])).km
        
#         distance_score = np.exp(-distance/10)
#         weather_score = 1.2 if (event['weather_condition'] == 'Clear' and 
#                                user['weather_preference'] in ['outdoor', 'any']) else 0.5
#         social_score = np.log1p(user['social_connectedness']) / 10
        
#         interaction_prob = 0.85*distance_score + 0.1*weather_score + 0.05*social_score

#         max_distance = 50 if random.random() < 0.7 else 300
        
#         if distance < max_distance and (random.random() < interaction_prob):
#             interaction_time = fake.date_time_between(
#                 start_date=event['start_time'] - timedelta(days=30), 
#                 end_date=event['start_time']
#             )
#             time_weight = calculate_time_weight(interaction_time, current_time)
#             interaction_prob *= time_weight
            
#             if distance <= 5:
#                 interaction_type_probs = [0.15, 0.20, 0.05, 0.25, 0.20, 0.05, 0.10]
#             elif distance <= 20:
#                 interaction_type_probs = [0.20, 0.15, 0.10, 0.20, 0.15, 0.10, 0.10]
#             elif distance <= 50:
#                 interaction_type_probs = [0.25, 0.10, 0.15, 0.15, 0.10, 0.15, 0.10]
#             elif distance <= 100:
#                 interaction_type_probs = [0.20, 0.05, 0.25, 0.10, 0.05, 0.20, 0.15]
#             else:
#                 interaction_type_probs = [0.15, 0.05, 0.30, 0.05, 0.05, 0.25, 0.15]
                
#             interactions.append({
#                 'interaction_id': fake.uuid4(),
#                 'user_id': user['user_id'],
#                 'event_id': event['event_id'],
#                 'interaction_type': np.random.choice(interaction_types, p=interaction_type_probs),
#                 'interaction_time': interaction_time,
#                 'distance_to_event': distance
#             })
    
#     return pd.DataFrame(interactions)

# def augment_cold_start(users, events, interactions):
#     cold_users = users[users['declared_interests'].apply(len) > 0]
#     trending_events = events.nlargest(100, 'historical_attendance_rate')
#     interaction_types = ['maybe', 'invited & maybe', 'no', 'yes', 'invited & yes', 'invited & no', 'invited']
    
#     cold_interactions = []
#     for _, user in cold_users.iterrows():
#         for _ in range(max(3, random.randint(3, 6))):
#             event = trending_events.sample(1).iloc[0]
            
#             distance = geodesic((user['location_lat'], user['location_lon']), 
#                               (event['lat'], event['lon'])).km
            
#             if distance <= 5:
#                 interaction_type_probs = [0.15, 0.20, 0.05, 0.25, 0.20, 0.05, 0.10]
#             elif distance <= 20:
#                 interaction_type_probs = [0.20, 0.15, 0.10, 0.20, 0.15, 0.10, 0.10]
#             elif distance <= 50:
#                 interaction_type_probs = [0.25, 0.10, 0.15, 0.15, 0.10, 0.15, 0.10]
#             elif distance <= 100:
#                 interaction_type_probs = [0.20, 0.05, 0.25, 0.10, 0.05, 0.20, 0.15]
#             else:
#                 interaction_type_probs = [0.15, 0.05, 0.30, 0.05, 0.05, 0.25, 0.15]
                
#             cold_interactions.append({
#                 'interaction_id': fake.uuid4(),
#                 'user_id': user['user_id'],
#                 'event_id': event['event_id'],
#                 'interaction_type': np.random.choice(interaction_types, p=interaction_type_probs),
#                 'interaction_time': fake.date_time_between(
#                     start_date=user['signup_date'], 
#                     end_date=user['signup_date'] + timedelta(days=7)
#                 ),
#                 'distance_to_event': round(distance)
#             })
    
#     return pd.concat([interactions, pd.DataFrame(cold_interactions)], ignore_index=True)

# # Connect to Hopsworks
# project = hopsworks.login()
# fs = project.get_feature_store()

# # Generate data
# print("Generating user data...")
# users_df = generate_users(25000)

# print("Generating event data...")
# events_df = generate_events(20000)

# print("Generating interaction data...")
# interactions_df = generate_interactions(users_df, events_df, 300000)
# interactions_df = augment_cold_start(users_df, events_df, interactions_df)

# # Create feature groups
# users_fg = fs.get_or_create_feature_group(
#     name="users",
#     version=1,
#     primary_key=["user_id"],
#     description="User features for weather-based event recommendation"
# )

# events_fg = fs.get_or_create_feature_group(
#     name="events",
#     version=1,
#     primary_key=["event_id"],
#     description="Event features for weather-based event recommendation"
# )

# interactions_fg = fs.get_or_create_feature_group(
#     name="interactions",
#     version=1,
#     primary_key=["interaction_id"],
#     description="User-event interactions for weather-based event recommendation"
# )

# # Insert data into feature groups
# print("Inserting data into feature groups...")
# users_fg.insert(users_df)
# events_fg.insert(events_df)
# interactions_fg.insert(interactions_df)


# # Save data to csv
# users_df.to_csv('users.csv')
# events_df.to_csv('events.csv')
# interactions_df.to_csv('interactions.csv')
# print("All data saved to csv")

# # Create Ranking Feature group
# # import pandas as pd
# # users_df = pd.read_csv("users_df.csv").drop(columns=(['Unnamed: 0']))
# # events_df = pd.read_csv("events_df.csv").drop(columns=(['Unnamed: 0']))
# # interactions_df = pd.read_csv("interactions_df.csv").drop(columns=(['Unnamed: 0']))
# events_interactions_df = pd.merge(
#     interactions_df, 
#     events_df, 
#     on='event_id', 
#     how='inner',
#     suffixes=('', '_event')  # Add suffix for event columns
# )

# ranking_df = pd.merge(
#     events_interactions_df, 
#     users_df, 
#     on='user_id', 
#     how='inner',
#     suffixes=('', '_user')  # Add suffix for user columns
# )

# ranking_df['label'] = ranking_df['interaction_type'].apply(
#     lambda x: 1 if x in ['maybe', 'invited & maybe', 'yes', 'invited & yes'] else 0
# )

# ranking_df_with_weather = ranking_df.drop(
#     ['interaction_id', 'interaction_type','interaction_time',\
#      'start_time', 'signup_date','social_connectedness'], 
#     axis=1
# )

# ranking_fg_weather = fs.get_or_create_feature_group(
#     name="weather_ranking",
#     description="Ranking Data with weather conditions.",
#     version=1,
#     primary_key=["user_id", "event_id"],
#     online_enabled=True,
#     #features=ranking_df_with_weather.columns.to_list()  # ← critical!
# )

# ranking_fg_weather.insert(ranking_df_with_weather)
# print('Done ✅')
# ranking_df_with_weather.to_csv("ranking_df_with_weather2.csv")


# # create ranking data without weather information
# ranking_df_without_weather = ranking_df.drop(['interaction_id', 'interaction_type',
#        'interaction_time', 'start_time','weather_condition',
#        'temperature', 'weather_preference',
#        'signup_date', 'social_connectedness'],axis=1)

# ranking_fg_without_weather = fs.get_or_create_feature_group(
#     name="no_weather_ranking",
#     description="Ranking Data without weather conditions.",
#     version=1,
#     primary_key=["user_id", "event_id"],
#     online_enabled=True,
#     #features=ranking_df_without_weather.columns.to_list()
# )

# ranking_fg_without_weather.insert(ranking_df_without_weather)
# print('Done ✅')

# ranking_df_without_weather.to_csv("ranking_df_without_weather2.csv")
# print("Feature backfill")
# # print(users_df.columns)
# # print(events_df.head()["event_type"])
# # print(interactions_df.columns)



import numpy as np
import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
from geopy.distance import geodesic
from scipy.stats import skewnorm, dirichlet
from mimesis import Generic
import hopsworks
import re

# Initialize Faker and set seed for reproducibility
fake = Faker()
generic = Generic('en')

np.random.seed(42)
random.seed(42)

# City coordinates and probabilities
cities = ['New York', 'London', 'Paris', 'Tokyo', 'Sydney', 'Berlin', 'Mumbai', 'São Paulo', 'Toronto', 'Dubai']
city_probs = [0.2, 0.15, 0.1, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05]
city_coords = {
    'New York': (40.7128, -74.0060), 'London': (51.5074, -0.1278), 'Paris': (48.8566, 2.3522),
    'Tokyo': (35.6762, 139.6503), 'Sydney': (-33.8688, 151.2093), 'Berlin': (52.5200, 13.4050),
    'Mumbai': (19.0760, 72.8777), 'São Paulo': (-23.5505, -46.6333), 'Toronto': (43.6532, -79.3832),
    'Dubai': (25.2048, 55.2708)
}

#generic = Person()

def generate_location(city):
    base_lat, base_lon = city_coords[city]
    if random.random() < 0.8:
        lat = base_lat + np.random.uniform(-0.1, 0.1)
        lon = base_lon + np.random.uniform(-0.1, 0.1)
    else:
        lat = base_lat + np.random.uniform(-2, 2)
        lon = base_lon + np.random.uniform(-2, 2)
    return lat, lon

def generate_users(n_users=20000):
    users = []
    interests = ['music', 'sports', 'tech', 'food', 'art', 'literature', 'cinema', 'travel', 'fitness', 'fashion']
    
    for _ in range(n_users):
        city = np.random.choice(cities, p=city_probs)
        lat, lon = generate_location(city)
        age = max(18, min(100, int(skewnorm.rvs(5, loc=25, scale=15))))
        weather_probs = dirichlet.rvs([0.3, 0.5, 0.2])[0]
        
        # Ensure at least one interest is selected
        user_interests = random.sample(interests, k=random.randint(1, min(4, len(interests))))
        
        users.append({
            'user_id': generic.person.identifier(mask='@@###@'),
            'user_lat': lat,
            'user_lon': lon,
            'user_city': city,
            'user_weather_preference': np.random.choice(['indoor', 'outdoor', 'any'], p=weather_probs),
            'age': age,
            'user_interests': ','.join(user_interests),  # Join interests into a comma-separated string
            'signup_date': fake.date_time_between(start_date='-2y', end_date='now'),
            'social_connectedness': np.random.poisson(lam=15)
        })
    return pd.DataFrame(users)


def generate_events(n_events=5000):
    events = []
    event_types = [
        'Education & Learning', 'Technology', 'Seasonal & Festivals', 'Arts & Culture', 
        'Entertainment', 'Sports & Fitness', 'Business & Networking', 'Health & Wellness', 
        'Music & Concerts', 'Food & Drink', 'Community & Causes', 'Immersive Experiences'
    ]
    weather_conditions = ['Clear', 'Rain', 'Snow', 'Cloudy', 'Windy']
    weather_probs = [0.5, 0.2, 0.05, 0.2, 0.05]
    
    current_date = datetime(2025, 3, 27, 11, 48)  # Current date and time
    
    for _ in range(n_events):
        event_type = np.random.choice(event_types)
        city = np.random.choice(cities, p=city_probs)
        lat, lon = generate_location(city)
        
        if event_type in ['Sports & Fitness', 'Seasonal & Festivals']:
            weather_condition = 'Clear' if random.random() < 0.8 else np.random.choice(['Rain', 'Cloudy'])
        elif event_type in ['Education & Learning', 'Technology', 'Business & Networking']:
            weather_condition = np.random.choice(['Clear', 'Cloudy'])
        else:
            weather_condition = np.random.choice(weather_conditions, p=weather_probs)
        
        base_temp = {
            'New York': 15, 'London': 12, 'Paris': 16, 'Tokyo': 20, 
            'Sydney': 22, 'Berlin': 14, 'Mumbai': 28, 'São Paulo': 24, 
            'Toronto': 10, 'Dubai': 32
        }[city]
        
        temp_adjustment = {
            'Clear': np.random.uniform(2, 5),
            'Rain': np.random.uniform(-3, 0),
            'Snow': np.random.uniform(-8, -3),
            'Cloudy': np.random.uniform(-1, 2),
            'Windy': np.random.uniform(-2, 1)
        }[weather_condition]
        
        temperature = round(base_temp + temp_adjustment, 1)
        
        start_time = fake.date_time_between(start_date=current_date, end_date=current_date + timedelta(days=180))
        is_weekend = start_time.weekday() >= 5
        hour_choices = [10, 14, 18] if is_weekend else [9, 13, 18, 19]
        start_time = start_time.replace(hour=np.random.choice(hour_choices))
        
        events.append({
            'event_id': generic.person.identifier(mask='@@###@'),
            'title': f"{fake.catch_phrase()} {event_type} in {city}",
            'event_type': event_type,
            'event_lat': lat,
            'event_lon': lon,
            'event_city': city,
            'start_time': start_time,
            'duration': np.random.choice([120, 180, 240, 360, 480]),
            'weather_condition': weather_condition,
            'temperature': temperature,
            'attendance_rate': np.random.beta(a=2, b=5) * 100,
            'event_indoor_capability': event_type in ['Education & Learning', 'Technology', 'Business & Networking', 
                                               'Arts & Culture', 'Entertainment', 'Immersive Experiences']
        })
    return pd.DataFrame(events)

def calculate_time_weight(interaction_time, current_time, half_life=30):
    time_diff = (current_time - interaction_time).days
    return np.exp(np.log(0.5) * time_diff / half_life)

def generate_interactions(users, events, n_interactions=100000):
    interactions = []
    interaction_types = ['maybe', 'invited & maybe', 'no', 'yes', 'invited & yes', 'invited & no', 'invited']
    attempts = n_interactions * 5
    current_time = datetime(2025, 3, 27, 11, 48)
    
    for _ in range(attempts):
        if len(interactions) >= n_interactions:
            break
            
        user = users.sample(1).iloc[0]
        event = events.sample(1).iloc[0]
        
        distance = geodesic((user['user_lat'], user['user_lon']), 
                           (event['event_lat'], event['event_lon'])).km
        
        distance_score = np.exp(-distance/10)
        weather_score = 1.2 if (event['weather_condition'] == 'Clear' and 
                               user['user_weather_preference'] in ['outdoor', 'any']) else 0.5
        social_score = np.log1p(user['social_connectedness']) / 10
        
        interaction_prob = 0.85*distance_score + 0.1*weather_score + 0.05*social_score

        max_distance = 50 if random.random() < 0.7 else 300
        
        if distance < max_distance and (random.random() < interaction_prob):
            interaction_time = fake.date_time_between(
                start_date=event['start_time'] - timedelta(days=30), 
                end_date=event['start_time']
            )
            time_weight = calculate_time_weight(interaction_time, current_time)
            interaction_prob *= time_weight
            
            if distance <= 5:
                interaction_type_probs = [0.15, 0.20, 0.05, 0.25, 0.20, 0.05, 0.10]
            elif distance <= 20:
                interaction_type_probs = [0.20, 0.15, 0.10, 0.20, 0.15, 0.10, 0.10]
            elif distance <= 50:
                interaction_type_probs = [0.25, 0.10, 0.15, 0.15, 0.10, 0.15, 0.10]
            elif distance <= 100:
                interaction_type_probs = [0.20, 0.05, 0.25, 0.10, 0.05, 0.20, 0.15]
            else:
                interaction_type_probs = [0.15, 0.05, 0.30, 0.05, 0.05, 0.25, 0.15]
                
            interactions.append({
                'interaction_id': generic.person.identifier(mask='@@###@'),
                'user_id': user['user_id'],
                'event_id': event['event_id'],
                'interaction_type': np.random.choice(interaction_types, p=interaction_type_probs),
                'interaction_time': interaction_time,
                'interaction_distance_to_event': distance
            })
    
    return pd.DataFrame(interactions)

def augment_cold_start(users, events, interactions):
    cold_users = users[users['user_interests'].apply(len) > 0]
    trending_events = events.nlargest(100, 'attendance_rate')
    interaction_types = ['maybe', 'invited & maybe', 'no', 'yes', 'invited & yes', 'invited & no', 'invited']
    
    cold_interactions = []
    for _, user in cold_users.iterrows():
        for _ in range(max(3, random.randint(3, 6))):
            event = trending_events.sample(1).iloc[0]
            
            distance = geodesic((user['user_lat'], user['user_lon']), 
                              (event['event_lat'], event['event_lon'])).km
            
            if distance <= 5:
                interaction_type_probs = [0.15, 0.20, 0.05, 0.25, 0.20, 0.05, 0.10]
            elif distance <= 20:
                interaction_type_probs = [0.20, 0.15, 0.10, 0.20, 0.15, 0.10, 0.10]
            elif distance <= 50:
                interaction_type_probs = [0.25, 0.10, 0.15, 0.15, 0.10, 0.15, 0.10]
            elif distance <= 100:
                interaction_type_probs = [0.20, 0.05, 0.25, 0.10, 0.05, 0.20, 0.15]
            else:
                interaction_type_probs = [0.15, 0.05, 0.30, 0.05, 0.05, 0.25, 0.15]
                
            cold_interactions.append({
                'interaction_id': fake.uuid4(),
                'user_id': user['user_id'],
                'event_id': event['event_id'],
                'interaction_type': np.random.choice(interaction_types, p=interaction_type_probs),
                'interaction_time': fake.date_time_between(
                    start_date=user['signup_date'], 
                    end_date=user['signup_date'] + timedelta(days=7)
                ),
                'interaction_distance_to_event': round(distance)
            })
    
    return pd.concat([interactions, pd.DataFrame(cold_interactions)], ignore_index=True)

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Generate data
print("Generating user data...")
users_df = generate_users(25000)

print("Generating event data...")
events_df = generate_events(20000)

print("Generating interaction data...")
interactions_df = generate_interactions(users_df, events_df, 300000)
interactions_df = augment_cold_start(users_df, events_df, interactions_df)

# Cleaning logic for user_interests and title
def clean_text(text):
    if not isinstance(text, str):
        return "unknown"
    text = text.lower().strip()
    text = text.replace(',', ' ')
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)     # collapse multiple spaces
    return text if text else "unknown"

def clean_text_columns(df):
    if 'title' in df.columns:
        df["title"] = df["title"].apply(lambda x: clean_text(x) if isinstance(x, str) and x.strip() else "unknown")
    if 'user_interests' in df.columns:
        df["user_interests"] = df["user_interests"].apply(
            lambda x: clean_text(x) if isinstance(x, str) and x.strip() else "unknown")
        
    return df

# Clean the dataframes
users_df = clean_text_columns(users_df)
events_df = clean_text_columns(events_df)
interactions_df = clean_text_columns(interactions_df)

# Create feature groups
users_fg = fs.get_or_create_feature_group(
    name="users",
    version=1,
    primary_key=["user_id"],
    online_enabled=True,
    description="User features for weather-based event recommendation"
)

events_fg = fs.get_or_create_feature_group(
    name="events",
    version=1,
    primary_key=["event_id"],
    online_enabled=True,
    description="Event features for weather-based event recommendation"
)

interactions_fg = fs.get_or_create_feature_group(
    name="interactions",
    version=1,
    primary_key=["interaction_id","user_id", "event_id"],
    online_enabled=True,
    description="User-event interactions for weather-based event recommendation"
)

# Insert data into feature groups
print("Inserting data into feature groups...")
users_fg.insert(users_df)
events_fg.insert(events_df)
interactions_fg.insert(interactions_df)

# Save data to csv
users_df.to_csv('users.csv')
events_df.to_csv('events.csv')
interactions_df.to_csv('interactions.csv')
print("All data saved to csv files.")

#create ranking feature group
events_interactions_df = pd.merge(
    interactions_df, 
    events_df, 
    on='event_id', 
    how='inner'  
)

ranking_df = pd.merge(
    events_interactions_df, 
    users_df, 
    on='user_id', 
    how='inner'
)

ranking_df['label'] = ranking_df['interaction_type'].apply(
    lambda x: 1 if x in ['maybe', 'invited & maybe', 'yes', 'invited & yes'] else 0
)

ranking_df_with_weather = ranking_df.drop(
    ['interaction_id', 'interaction_type','interaction_time',\
     'start_time', 'signup_date','social_connectedness'], 
    axis=1
)

ranking_fg_weather = fs.get_or_create_feature_group(
    name="weather_ranking",
    description="Ranking Data with weather conditions.",
    version=1,
    primary_key=["user_id", "event_id"],
    online_enabled=True,
    #features=ranking_df_with_weather.columns.to_list()  # ← critical!
)

ranking_fg_weather.insert(ranking_df_with_weather)
print('Done ✅')
ranking_df_with_weather.to_csv("ranking_df_with_weather2.csv")


# create ranking data without weather information
ranking_df_without_weather = ranking_df.drop(['interaction_id', 'interaction_type',
       'interaction_time', 'start_time','weather_condition',
       'temperature', 'user_weather_preference',
       'signup_date', 'social_connectedness'],axis=1)

ranking_fg_without_weather = fs.get_or_create_feature_group(
    name="no_weather_ranking",
    description="Ranking Data without weather conditions.",
    version=1,
    primary_key=["user_id", "event_id"],
    online_enabled=True,
    #features=ranking_df_without_weather.columns.to_list()
)

ranking_fg_without_weather.insert(ranking_df_without_weather)
print('Done ✅')

ranking_df_without_weather.to_csv("ranking_df_without_weather2.csv")
print("Feature backfill")

