import numpy as np
import random
from faker import Faker
from scipy.stats import skewnorm, dirichlet
from datetime import datetime, timedelta

# Initialize Faker and set seed for reproducibility
fake = Faker()
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
        
        users.append({
            'user_id': fake.uuid4(),
            'location_lat': lat,
            'location_lon': lon,
            'city': city,
            'event_type_preference': np.random.choice(['indoor', 'outdoor', 'any'], p=weather_probs),
            'age': age,
            'declared_interests': random.sample(interests, k=random.randint(0, min(4, len(interests)))) if random.random() < 0.7 else [],
            #'signup_date': fake.date_time_between(start_date='-2y', end_date='now'),
            #'social_connectedness': np.random.poisson(lam=15)
        })
    return users


