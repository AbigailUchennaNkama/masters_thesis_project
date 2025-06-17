
import os
import pandas as pd
import datetime
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
            name="ranking_model", 
            version=1,
        )
        
        # Extract input schema from the model
        input_schema = model.model_schema["input_schema"]["columnar_schema"]
        
        # Get the names of features expected by the ranking model
        self.ranking_model_feature_names = [feat["name"] for feat in input_schema]
            
    def preprocess(self, inputs):
        # Extract the input instance
        inputs = inputs["instances"][0]

        # Extract customer_id from inputs
        user_id = inputs["user_id"]
        
        # Search for candidate items
        neighbors = self.candidate_index.find_neighbors(
            inputs["query_emb"], 
            k=100,
        )
        neighbors = [neighbor[0] for neighbor in neighbors]
        
        # #Get IDs of events happening before the currrent date - should filter to only furture events
        

        # Get today's date as a datetime.date object
        current_date = datetime.datetime.now().date()

        # Create a new column with the 'start_time' column converted to datetime.date
        neighbors["start_date"] = pd.to_datetime(neighbors["start_time"]).dt.date

        # Filter events in the past
        past_events = neighbors[neighbors["start_date"] < current_date] 
        past_events_ids = past_events["event_id"].tolist()
        
        # Filter candidate events to exclude those already seen by the user
        event_id_list = [
            event_id
            for event_id 
            in neighbors 
            if event_id
            not in past_events_ids
        ]
        event_id_df = pd.DataFrame({"event_id" : event_id_list})
        
        # Retrieve event data for candidate events
        events_data = [
            self.events_fv.get_feature_vector({"event_id": event_id}) 
            for event_id 
            in event_id_list
        ]

        events_df = pd.DataFrame(
            data=events_data, 
            columns=self.event_features,
        )
        
        # Join candidate items with their features
        ranking_model_inputs = event_id_df.merge(
            events_df, 
            on="event_id", 
            how="inner",
        )        
        
        # Add customer features
        user_features = self.users_fv.get_feature_vector(
            {"user_id": user_id}, 
            return_type="pandas",
        )
        # ['interaction_distance_to_event', 'event_type', 'event_city','title', 
       #'weather_condition', 'temperature','precipitation', 'attendance_rate',
        # 'event_indoor_capability', 'user_city', 'indoor_outdoor_preference',
        # 'age', 'user_interests','user_weather_condition', 'user_temperature','user_precipitation']
        ranking_model_inputs["user_id"] = user_features['user_id'].values[0]   
        ranking_model_inputs["user_city"] = user_features["user_city"].values[0]
        ranking_model_inputs["indoor_outdoor_preference"] = user_features["indoor_outdoor_preference"].values[0] 
        ranking_model_inputs["age"] = user_features["age"].values[0] 
        ranking_model_inputs["user_interests"] = user_features["user_interests"].values[0] 
        ranking_model_inputs["user_weather_condition"] = user_features["user_weather_condition"].values[0]
        ranking_model_inputs["user_temperature"] = user_features["user_temperature"].values[0]
        ranking_model_inputs["user_precipitation"] = user_features["user_precipitation"].values[0]
        
        
        # Select only the features required by the ranking model
        ranking_model_inputs = ranking_model_inputs[self.ranking_model_feature_names]
                
        return { 
            "inputs" : [{"ranking_features": ranking_model_inputs.values.tolist(), "event_ids": event_id_list}]
        }

    def postprocess(self, outputs):
        # Extract predictions from the outputs
        preds = outputs["predictions"]
        
        # Merge prediction scores and corresponding article IDs into a list of tuples
        ranking = list(zip(preds["scores"], preds["event_ids"]))
        
        # Sort the ranking list by score in descending order
        ranking.sort(reverse=True)
        
        # Return the sorted ranking list
        return { 
            "ranking": ranking,
        }
