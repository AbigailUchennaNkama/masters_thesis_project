import numpy as np
import pandas as pd
import random
from faker import Faker
from scipy.stats import skewnorm, dirichlet
from mimesis import Generic

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

def generate_location(city):
    base_lat, base_lon = city_coords[city]
    # Relaxed location spread to increase nearby interactions
    lat = base_lat + np.random.uniform(-0.5, 0.5)  # Wider range for more local interactions
    lon = base_lon + np.random.uniform(-0.5, 0.5)
    return lat, lon

def generate_users(n_users=50000):
    users = []
    interests = ['music', 'sports', 'tech', 'food', 'art', 'literature', 'cinema', 'travel', 'fitness', 'fashion']

    for _ in range(n_users):
        city = np.random.choice(cities, p=city_probs)
        lat, lon = generate_location(city)
        age = max(18, min(100, int(skewnorm.rvs(5, loc=25, scale=15))))
        weather_probs = dirichlet.rvs([0.3, 0.5, 0.2])[0]
        user_interests = random.sample(interests, k=random.randint(2, 4))  # Ensure 2–4 interests for diversity

        users.append({
            'user_id': generic.person.identifier(mask='@@###@'),
            'user_lat': lat,
            'user_lon': lon,
            'user_city': city,
            'indoor_outdoor_preference': np.random.choice(['indoor', 'outdoor', 'any'], p=weather_probs),
            'age': age,
            'user_interests': ','.join(user_interests),
            'signup_date': fake.date_time_between(start_date='-2y', end_date='now'),
            'social_connectedness': np.random.poisson(lam=15)
        })
    return pd.DataFrame(users)




