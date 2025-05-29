# import numpy as np
# import pandas as pd
# import random
# from faker import Faker
# from scipy.stats import skewnorm, dirichlet
# from mimesis import Generic

# # Initialize Faker and set seed for reproducibility
# fake = Faker()
# generic = Generic('en')
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
#     lat = base_lat + np.random.uniform(-0.5, 0.5)
#     lon = base_lon + np.random.uniform(-0.5, 0.5)
#     return lat, lon

# def get_city_weather(city):
#     # Weather and temperature logic similar to events
#     weather_conditions = ['Clear', 'Rain', 'Snow', 'Cloudy', 'Windy']
#     weather_probs = [0.5, 0.2, 0.05, 0.2, 0.05]
#     weather_condition = np.random.choice(weather_conditions, p=weather_probs)
#     base_temp = {
#         'New York': 15, 'London': 12, 'Paris': 16, 'Tokyo': 20,
#         'Sydney': 22, 'Berlin': 14, 'Mumbai': 28, 'São Paulo': 24,
#         'Toronto': 10, 'Dubai': 32
#     }[city]
#     temp_adjustment = {
#         'Clear': np.random.uniform(2, 5),
#         'Rain': np.random.uniform(-3, 0),
#         'Snow': np.random.uniform(-8, -3),
#         'Cloudy': np.random.uniform(-1, 2),
#         'Windy': np.random.uniform(-2, 1)
#     }[weather_condition]
#     temperature = round(base_temp + temp_adjustment, 1)
#     # Precipitation mm logic
#     precipitation = {
#         'Clear': 0.0,
#         'Cloudy': np.random.uniform(0, 0.5),
#         'Windy': np.random.uniform(0, 0.5),
#         'Rain': np.random.uniform(2, 20),
#         'Snow': np.random.uniform(1, 10)
#     }[weather_condition]
#     precipitation = round(precipitation, 2)
#     return weather_condition, temperature, precipitation

# def generate_users(n_users=50000):
#     users = []
#     interests = ['music', 'sports', 'tech', 'food', 'art', 'literature', 'cinema', 'travel', 'fitness', 'fashion']

#     for _ in range(n_users):
#         city = np.random.choice(cities, p=city_probs)
#         lat, lon = generate_location(city)
#         age = max(18, min(100, int(skewnorm.rvs(5, loc=25, scale=15))))
#         weather_probs = dirichlet.rvs([0.3, 0.5, 0.2])[0]
#         user_interests = random.sample(interests, k=random.randint(2, 4))
#         weather_condition, temperature, precipitation = get_city_weather(city)

#         users.append({
#             'user_id': generic.person.identifier(mask='@@###@'),
#             'user_lat': lat,
#             'user_lon': lon,
#             'user_city': city,
#             'indoor_outdoor_preference': np.random.choice(['indoor', 'outdoor', 'any'], p=weather_probs),
#             'age': age,
#             'user_interests': ','.join(user_interests),
#             'signup_date': fake.date_time_between(start_date='-2y', end_date='now'),
#             'social_connectedness': np.random.poisson(lam=15),
#             'weather_condition': weather_condition,
#             'temperature': temperature,
#             'precipitation': precipitation
#         })
#     return pd.DataFrame(users)




# import numpy as np
# import pandas as pd
# from faker import Faker
# from datetime import datetime, timedelta
# from mimesis import Generic
# import random

# # Initialize Faker
# fake = Faker()
# generic = Generic('en')
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
#     lat = base_lat + np.random.uniform(-0.5, 0.5)
#     lon = base_lon + np.random.uniform(-0.5, 0.5)
#     return lat, lon

# def get_precipitation(weather_condition):
#     # Precipitation mm logic
#     return round({
#         'Clear': 0.0,
#         'Cloudy': np.random.uniform(0, 0.5),
#         'Windy': np.random.uniform(0, 0.5),
#         'Rain': np.random.uniform(2, 20),
#         'Snow': np.random.uniform(1, 10)
#     }[weather_condition], 2)

# def generate_events(n_events=10000):
#     events = []
#     event_types = [
#         'Education & Learning', 'Technology', 'Seasonal & Festivals', 'Arts & Culture',
#         'Entertainment', 'Sports & Fitness', 'Business & Networking', 'Health & Wellness',
#         'Music & Concerts', 'Food & Drink', 'Community & Causes', 'Immersive Experiences'
#     ]
#     weather_conditions = ['Clear', 'Rain', 'Snow', 'Cloudy', 'Windy']
#     weather_probs = [0.5, 0.2, 0.05, 0.2, 0.05]
#     current_date = datetime(2025, 5, 25, 11, 42)

#     for _ in range(n_events):
#         event_type = np.random.choice(event_types)
#         city = np.random.choice(cities, p=city_probs)
#         lat, lon = generate_location(city)
#         weather_condition = np.random.choice(weather_conditions, p=weather_probs)
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
#         precipitation = get_precipitation(weather_condition)
#         start_time = fake.date_time_between(start_date=current_date, end_date=current_date + timedelta(days=180))
#         is_weekend = start_time.weekday() >= 5
#         hour_choices = [10, 14, 18] if is_weekend else [9, 13, 18, 19]
#         start_time = start_time.replace(hour=np.random.choice(hour_choices))

#         events.append({
#             'event_id': generic.person.identifier(mask='@@###@'),
#             'title': f"{fake.catch_phrase()} {event_type} in {city}",
#             'category': event_type,
#             'event_lat': lat,
#             'event_lon': lon,
#             'event_city': city,
#             'start_time': start_time,
#             'duration': np.random.choice([120, 180, 240, 360, 480]),
#             'weather_condition': weather_condition,
#             'temperature': temperature,
#             'precipitation': precipitation,
#             'attendance_rate': np.random.beta(a=2, b=5) * 100,
#             'event_indoor_capability': event_type in [
#                 'Education & Learning', 'Technology', 'Business & Networking',
#                 'Arts & Culture', 'Entertainment', 'Immersive Experiences'
#             ]
#         })
#     return pd.DataFrame(events)


# import numpy as np
# import pandas as pd
# import random
# from faker import Faker
# from datetime import datetime, timedelta
# from geopy.distance import geodesic
# from mimesis import Generic

# np.random.seed(42)
# random.seed(42)
# # Initialize Faker
# fake = Faker()
# generic = Generic('en')

# # City coordinates and probabilities
# cities = ['New York', 'London', 'Paris', 'Tokyo', 'Sydney', 'Berlin', 'Mumbai', 'São Paulo', 'Toronto', 'Dubai']
# city_probs = [0.2, 0.15, 0.1, 0.1, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05]

# def calculate_time_weight(interaction_time, current_time, half_life=30):
#     time_diff = (current_time - interaction_time).days
#     return np.exp(np.log(0.5) * time_diff / half_life)

# def match_interests(user_interests, event_type):
#     interest_list = user_interests.split(',')
#     event_type_lower = event_type.lower()
#     interest_mapping = {
#         'music': ['music & concerts'],
#         'sports': ['sports & fitness'],
#         'tech': ['technology', 'business & networking'],
#         'food': ['food & drink'],
#         'art': ['arts & culture'],
#         'literature': ['education & learning'],
#         'cinema': ['entertainment'],
#         'travel': ['seasonal & festivals'],
#         'fitness': ['sports & fitness', 'health & wellness'],
#         'fashion': ['arts & culture']
#     }
#     score = 0
#     for interest in interest_list:
#         if interest in interest_mapping:
#             if event_type_lower in [e.lower() for e in interest_mapping[interest]]:
#                 score += 1
#     return score / max(1, len(interest_list))

# def generate_interactions(users, events, n_interactions=500000):
#     interactions = []
#     interaction_types = ['maybe', 'invited & maybe', 'no', 'yes', 'invited & yes', 'invited & no', 'invited']
#     current_time = datetime(2025, 5, 25, 11, 42)

#     users_per_city = {city: users[users['user_city'] == city].index for city in cities}
#     events_per_city = {city: events[events['event_city'] == city].index for city in cities}

#     target_interactions_per_user = n_interactions // len(users)
#     user_interaction_counts = {user_id: 0 for user_id in users['user_id']}

#     while len(interactions) < n_interactions:
#         city = np.random.choice(cities, p=city_probs)
#         user_idx = np.random.choice(users_per_city[city])
#         user = users.loc[user_idx]

#         if random.random() < 0.8:
#             if len(events_per_city[city]) > 0:
#                 event_idx = np.random.choice(events_per_city[city])
#             else:
#                 event_idx = np.random.choice(events.index)
#         else:
#             event_idx = np.random.choice(events.index)
#         event = events.loc[event_idx]

#         distance = geodesic((user['user_lat'], user['user_lon']),
#                            (event['event_lat'], event['event_lon'])).km

#         max_distance = 100 if random.random() < 0.8 else 500

#         distance_score = np.exp(-distance / 50)
#         weather_score = 1.5 if (event['weather_condition'] == 'Clear' and
#                                user['indoor_outdoor_preference'] in ['outdoor', 'any']) or \
#                               (event['event_indoor_capability'] and
#                                event['weather_condition'] in ['Rain', 'Snow']) else 0.8
#         social_score = np.log1p(user['social_connectedness']) / 5
#         interest_score = match_interests(user['user_interests'], event['category'])

#         interaction_prob = 0.5 * distance_score + 0.2 * weather_score + 0.1 * social_score + 0.2 * interest_score

#         if distance < max_distance and random.random() < min(interaction_prob * 1.5, 0.9) and \
#            user_interaction_counts[user['user_id']] < target_interactions_per_user * 2:
#             interaction_time = fake.date_time_between(
#                 start_date=event['start_time'] - timedelta(days=30),
#                 end_date=event['start_time']
#             )
#             time_weight = calculate_time_weight(interaction_time, current_time)
#             interaction_prob *= time_weight

#             if distance <= 10:
#                 interaction_type_probs = [0.1, 0.15, 0.05, 0.35, 0.25, 0.05, 0.05]
#             elif distance <= 50:
#                 interaction_type_probs = [0.15, 0.15, 0.1, 0.25, 0.2, 0.1, 0.05]
#             elif distance <= 100:
#                 interaction_type_probs = [0.2, 0.1, 0.15, 0.2, 0.15, 0.15, 0.05]
#             else:
#                 interaction_type_probs = [0.25, 0.05, 0.2, 0.15, 0.1, 0.2, 0.05]

#             if interest_score > 0.5:
#                 interaction_type_probs = [p * 1.5 if t in ['yes', 'invited & yes'] else p * 0.7
#                                          for p, t in zip(interaction_type_probs, interaction_types)]
#                 interaction_type_probs = [p / sum(interaction_type_probs) for p in interaction_type_probs]

#             interactions.append({
#                 'interaction_id': generic.person.identifier(mask='@@###@'),
#                 'user_id': user['user_id'],
#                 'event_id': event['event_id'],
#                 'interaction_type': np.random.choice(interaction_types, p=interaction_type_probs),
#                 'interaction_time': interaction_time,
#                 'interaction_distance_to_event': distance
#             })
#             user_interaction_counts[user['user_id']] += 1

#     return pd.DataFrame(interactions)


# Generate data
print("Generating user data...")
num_users = 50000
users = generate_users(num_users)
users_df = pd.DataFrame(users)

print("Generating event data...")
num_events = 10000
events = generate_events(num_events)
events_df = pd.DataFrame(events)
#"/home/nkama/masters_thesis_project/thesis"

print("Done!")
print("Generating interaction data...")
num_interactions = 500000
interactions = generate_interactions(users_df, events_df, num_interactions)
interactions_df = pd.DataFrame(interactions)

# Add interaction_label for model training
interactions_df['interaction_label'] = interactions_df['interaction_type'].apply(
    lambda x: 1 if x in ['yes', 'invited & yes', 'maybe', 'invited & maybe'] else 0
)

print("Done!")

events_df.to_csv("/home/nkama/masters_thesis_project/thesis/events.csv",index=False )
# users_df.to_csv("/home/nkama/masters_thesis_project/thesis/users.csv",index=False)

# interactions_df.to_csv("/home/nkama/masters_thesis_project/thesis/interactions.csv",index=False)