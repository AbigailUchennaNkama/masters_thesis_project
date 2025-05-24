
import os
import pandas as pd
from datetime import datetime

import hopsworks
from opensearchpy import OpenSearch

import logging


class Transformer(object):
    
    def __init__(self):
        # Connect to Hopsworks
        project = hopsworks.connection().get_project()
        self.fs = project.get_feature_store()
        
        # Retrieve the 'events' feature view
        self.events_fv = self.fs.get_feature_view(
            name="events", 
            version=1,
        )
        
        # Get list of feature names for events
        self.event_features = [feat.name for feat in self.events_fv.schema]
        
        # Retrieve the 'users' feature view
        self.users_fv = self.fs.get_feature_view(
            name="users", 
            version=1,
        )

        # Retrieve the 'candidate_embeddings' feature view
        self.candidate_index = self.fs.get_feature_view(
            name="candidate_embeddings", 
            version=1,
        )

        # Retrieve ranking model
        mr = project.get_model_registry()
        model = mr.get_model(
            name="no_weather_ranking_model", 
            version=1,
        )
        
        # Extract input schema from the model
        input_schema = model.model_schema["input_schema"]["columnar_schema"]
        
        # Get the names of features expected by the ranking model
        self.ranking_model_feature_names = [feat["name"] for feat in input_schema]
            
    def preprocess(self, inputs):
        # Extract the input instance
        inputs = inputs["instances"][0]

        # Extract user_id from inputs
        user_id = inputs["user_id"]
        
        # Search for candidate items
        neighbors = self.candidate_index.find_neighbors(
            inputs["query_emb"], 
            k=100,
        )
        neighbors = [neighbor[0] for neighbor in neighbors]
        
        # Get user interests
        user_features = self.users_fv.get_feature_vector(
            {"user_id": user_id}, 
            return_type="pandas",
        )
        
        # Get user interests (assuming it's stored in the user feature view)
        user_interests = user_features["user_interests"].values[0].split(",") if "user_interests" in user_features else []
        
        # Get current date
        current_date = datetime.now()
        
        # Retrieve event data for candidate events
        events_data = [
            self.events_fv.get_feature_vector({"event_id": event_id}) 
            for event_id 
            in neighbors
        ]

        events_df = pd.DataFrame(
            data=events_data, 
            columns=self.event_features,
        )
        
        # Filter logic implementation:
        # 1. Filter out events with dates earlier than current date
        # 2. Filter out events with categories not in user interests

        filtered_events = []
        filtered_event_ids = []
        
        for index, row in events_df.iterrows():
            event_id = row["event_id"]
            #event_date = datetime.strptime(row["start_time"], "%Y-%m-%d") if "start_time" in row else None
            event_category = row["event_type"] if "event_type" in row else None
            
            # # Skip if event is in the past
            # if event_date and event_date < current_date:
            #     continue
                
            # Skip if event category doesn't match user interests and user has interests
            if event_category and user_interests and event_category not in user_interests:
                continue
                
                
            # If passed all filters, add to filtered list
            filtered_events.append(row)
            filtered_event_ids.append(event_id)
        
        # Create DataFrame from filtered events
        filtered_events_df = pd.DataFrame(filtered_events) if filtered_events else pd.DataFrame(columns=self.event_features)
        event_id_df = pd.DataFrame({"event_id": filtered_event_ids})
        
        # If no events passed the filters, return empty result
        if filtered_events_df.empty:
            return {
                "inputs": [{"ranking_features": [], "event_ids": []}]
            }
        
        # Join candidate items with their features
        weather_ranking_model_inputs = event_id_df.merge(
            filtered_events_df, 
            on="event_id", 
            how="inner",
        )        
        
        # Add user features
        weather_ranking_model_inputs["user_id"] = user_features['age'].values[0]   
        weather_ranking_model_inputs["gender"] = user_features["gender"].values[0] 
        weather_ranking_model_inputs["age"] = user_features["age"].values[0] 
        weather_ranking_model_inputs["country"] = user_features["country"].values[0] 
        
        # Select only the features required by the weather ranking model
        weather_ranking_model_inputs = weather_ranking_model_inputs[self.weather_ranking_model_feature_names]
        # Select only the features required by the no weather ranking model
        no_weather_ranking_model_inputs = no_weather_ranking_model_inputs[self.no_weather_ranking_model_feature_names]
                
        return { 
            "inputs" : [{"ranking_features": ranking_model_inputs.values.tolist(), "event_ids": filtered_event_ids}]
        }

    def postprocess(self, outputs):
        # Extract predictions from the outputs
        preds = outputs["predictions"]
        
        # Merge prediction scores and corresponding event IDs into a list of tuples
        ranking = list(zip(preds["scores"], preds["event_ids"]))
        
        # Sort the ranking list by score in descending order
        ranking.sort(reverse=True)
        
        # Return the sorted ranking list
        return { 
            "ranking": ranking,
        }
