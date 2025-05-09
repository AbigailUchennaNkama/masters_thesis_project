# import numpy as np
# import pandas as pd
# import random
# from faker import Faker
# from datetime import datetime, timedelta
# from geopy.distance import geodesic

# # Initialize Faker and set seed for reproducibility
# fake = Faker()
# np.random.seed(42)
# random.seed(42)

# def calculate_time_weight(interaction_time, current_time, half_life=30):
#     time_diff = (current_time - interaction_time).days
#     return np.exp(np.log(0.5) * time_diff / half_life)

# def generate_interactions(users, events, n_interactions=100000):
#     interactions = []
#     interaction_types = ['maybe', 'invited & maybe', 'no', 'yes', 'invited & yes', 'invited & no', 'invited']
#     all_distances = []
#     attempts = n_interactions * 5
    
#     for _ in range(attempts):
#         if len(interactions) >= n_interactions:
#             break
            
#         user = random.choice(users)
#         event = random.choice(events)
        
#         distance = geodesic((user['lat'], user['lon']), 
#                            (event['lat'], event['lon'])).km
#         all_distances.append(distance)
        
#         distance_score = np.exp(-distance/10)
#         weather_score = 1.2 if (event['weather_condition'] == 'Clear' and 
#                                user['event_type_preference'] in ['outdoor', 'any']) else 0.5
        
        
#         interaction_prob = 0.9*distance_score + 0.1*weather_score 

#         max_distance = 50 if random.random() < 0.7 else 300
        
#         if distance < max_distance and (random.random() < interaction_prob):
#             interaction_time = fake.date_time_between(
#                 start_date=event['start_time'] - timedelta(days=30), 
#                 end_date=event['start_time']
#             )
#             current_time = datetime(2025, 3, 26, 17, 57, 0)  # Updated to match the given date
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
#                 'distance_to_event': distance
#             })
    
#     return interactions

# def augment_cold_start(users, events, interactions):
#     cold_users = [user for user in users if user['declared_interests']]
#     trending_events = sorted(events, key=lambda x: x['historical_attendance_rate'], reverse=True)[:100]
#     interaction_types = ['maybe', 'invited & maybe', 'no', 'yes', 'invited & yes', 'invited & no', 'invited']
    
#     cold_interactions = []
#     for user in cold_users:
#         for _ in range(max(3, random.randint(3, 6))):
#             event = random.choice(trending_events)
            
#             distance = geodesic((user['lat'], user['lon']), 
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
#                 'distance_to_event': round(distance)
#             })
    
    # return interactions + cold_interactions
import numpy as np
import pandas as pd
import random
from faker import Faker
from datetime import datetime
from geopy.distance import geodesic

Faker.seed(42)
np.random.seed(42)
random.seed(42)

def generate_interactions(users, events, n_interactions=100000):
    interactions = []
    interaction_types = ['maybe', 'invited & maybe', 'no', 'yes', 'invited & yes', 'invited & no', 'invited']
    attempts = n_interactions * 5
    current_time = datetime(2025, 3, 26, 19, 42)
    
    for _ in range(attempts):
        if len(interactions) >= n_interactions:
            break
            
        user = users.sample(1).iloc[0]
        event = events.sample(1).iloc[0]
        
        distance = geodesic((user['location_lat'], user['location_lon']), 
                           (event['location_lat'], event['location_lon'])).km
        
        distance_score = np.exp(-distance/10)
        weather_score = 1.2 if (event['weather_condition'] == 'Clear' and 
                               user['event_type_preference'] in ['outdoor', 'any']) else 0.5

        interaction_prob = 0.90*distance_score + 0.1*weather_score 

        max_distance = 50 if random.random() < 0.7 else 300
        
        if distance < max_distance and (random.random() < interaction_prob):
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
                'interaction_id': Faker().uuid4(),
                'user_id': user['user_id'],
                'event_id': event['event_id'],
                'interaction_type': np.random.choice(interaction_types, p=interaction_type_probs),
                'distance_to_event': distance
            })
    
    return pd.DataFrame(interactions)

def augment_cold_start(users, events, interactions):
    cold_users = users[users['declared_interests'].apply(len) > 0]
    trending_events = events.nlargest(100, 'historical_attendance_rate')
    interaction_types = ['maybe', 'invited & maybe', 'no', 'yes', 'invited & yes', 'invited & no', 'invited']
    
    cold_interactions = []
    for _, user in cold_users.iterrows():
        for _ in range(max(3, random.randint(3, 6))):
            event = trending_events.sample(1).iloc[0]
            
            distance = geodesic((user['location_lat'], user['location_lon']), 
                              (event['location_lat'], event['location_lon'])).km
            
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
                'interaction_id': Faker().uuid4(),
                'user_id': user['user_id'],
                'event_id': event['event_id'],
                'interaction_type': np.random.choice(interaction_types, p=interaction_type_probs),
                'distance_to_event': round(distance)
            })
    
    return pd.concat([interactions, pd.DataFrame(cold_interactions)], ignore_index=True)


