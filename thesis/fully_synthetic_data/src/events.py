import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
from mimesis import Generic
import random
# Initialize Faker
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

def generate_events(n_events=10000):
    events = []
    event_types = [
        'Education & Learning', 'Technology', 'Seasonal & Festivals', 'Arts & Culture',
        'Entertainment', 'Sports & Fitness', 'Business & Networking', 'Health & Wellness',
        'Music & Concerts', 'Food & Drink', 'Community & Causes', 'Immersive Experiences'
    ]
    weather_conditions = ['Clear', 'Rain', 'Snow', 'Cloudy', 'Windy']
    weather_probs = [0.5, 0.2, 0.05, 0.2, 0.05]
    current_date = datetime(2025, 5, 25, 11, 42)  # Updated to current date

    for _ in range(n_events):
        event_type = np.random.choice(event_types)
        city = np.random.choice(cities, p=city_probs)
        lat, lon = generate_location(city)
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
            'category': event_type,
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


