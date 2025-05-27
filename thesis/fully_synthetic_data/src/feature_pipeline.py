import pandas as pd
import hopsworks
from thesis.fully_synthetic_data.src.users import generate_users
from thesis.fully_synthetic_data.src.events import generate_events
from thesis.fully_synthetic_data.src.interactions import generate_interactions

# Login to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Get feature groups
users_fg = fs.get_feature_group(
    name="users",
    version=1,
)

events_fg = fs.get_feature_group(
    name="events",
    version=1,
)

interactions_fg = fs.get_feature_group(
    name="interactions",
    version=1,
)

# Generate data for users
user_data = generate_users(1_000)
data_users_df = pd.DataFrame(user_data)

# Generate data for events
event_data = generate_events(1_000)
data_events_df = pd.DataFrame(event_data)

# Generate interactions
num_interactions = 10_000
interactions = generate_interactions(num_interactions, user_data, event_data)
data_interactions_df = pd.DataFrame(interactions)

# Insert data into feature groups
users_fg.insert(data_users_df)
events_fg.insert(data_events_df)
interactions_fg.insert(data_interactions_df)
