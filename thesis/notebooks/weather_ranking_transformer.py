
import os
import pandas as pd
import logging
from datetime import datetime

import hopsworks
from opensearchpy import OpenSearch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Transformer(object):
    
    def __init__(self):
        try:
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
            logger.info(f"Event features: {self.event_features}")
            
            # Retrieve the 'users' feature view
            self.users_fv = self.fs.get_feature_view(
                name="users", 
                version=1,
            )
            
            # Get list of user features
            self.user_features_list = [feat.name for feat in self.users_fv.schema]
            logger.info(f"User features: {self.user_features_list}")

            # Retrieve the 'candidate_embeddings' feature view
            self.candidate_index = self.fs.get_feature_view(
                name="candidate_embeddings", 
                version=1,
            )

            # Retrieve ranking model
            mr = project.get_model_registry()
            model = mr.get_model(
                name="weather_ranking_model", 
                version=1,
            )
            
            # Extract input schema from the model
            input_schema = model.model_schema["input_schema"]["columnar_schema"]
            
            # Get the names of features expected by the ranking model
            self.ranking_model_feature_names = [feat["name"] for feat in input_schema]
            logger.info(f"Ranking model features: {self.ranking_model_feature_names}")
            
            # Define query and candidate features
            self.query_features = ["user_id", "user_city", "age", "user_interests"]
            self.candidate_features = ["event_id", "title", "event_type", "event_city"]
            
            logger.info("Transformer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing transformer: {str(e)}")
            raise
            
    def preprocess(self, inputs):
        try:
            logger.info(f"Input structure: {inputs}")
            
            # Handle different input formats
            if isinstance(inputs["instances"], list):
                if len(inputs["instances"]) > 0:
                    if isinstance(inputs["instances"][0], list):
                        # Handle double nested list format: {"instances": [[{...}]]}
                        instance = inputs["instances"][0][0]
                    else:
                        # Handle single nested list format: {"instances": [{...}]}
                        instance = inputs["instances"][0]
                else:
                    logger.warning("Empty instances list")
                    return {"inputs": [{"ranking_features": [], "event_ids": []}]}
            else:
                # Handle direct format: {"instances": {...}}
                instance = inputs["instances"]

            # Extract user_id from inputs
            user_id = instance["user_id"]
            logger.info(f"Processing for user_id: {user_id}")
            
            # Check if query_emb exists
            if "query_emb" not in instance:
                logger.error("No query_emb found in input")
                return {"inputs": [{"ranking_features": [], "event_ids": []}]}
            
            # Log query embedding size
            logger.info(f"Query embedding size: {len(instance['query_emb'])}")
            
            # Search for candidate items
            try:
                neighbors = self.candidate_index.find_neighbors(
                    instance["query_emb"], 
                    k=100,
                )
                neighbors = [neighbor[0] for neighbor in neighbors]
                logger.info(f"Found {len(neighbors)} neighbors")
            except Exception as e:
                logger.error(f"Error finding neighbors: {str(e)}")
                return {"inputs": [{"ranking_features": [], "event_ids": []}]}
            
            # If no neighbors found, return empty result
            if not neighbors:
                logger.warning("No neighbors found")
                return {"inputs": [{"ranking_features": [], "event_ids": []}]}
            
            # Filter candidate events 
            event_id_list = [event_id for event_id in neighbors]
            event_id_df = pd.DataFrame({"event_id": event_id_list})
            
            # Retrieve event data for candidate events
            events_data = []
            for event_id in event_id_list:
                try:
                    event = self.events_fv.get_feature_vector({"event_id": event_id})
                    events_data.append(event)
                except Exception as e:
                    logger.warning(f"Could not get features for event {event_id}: {str(e)}")
            
            # If no events data, return empty result
            if not events_data:
                logger.warning("No valid events found")
                return {"inputs": [{"ranking_features": [], "event_ids": []}]}
            
            events_df = pd.DataFrame(
                data=events_data, 
                columns=self.event_features,
            )
            logger.info(f"Retrieved {len(events_df)} events")
            
            # Join candidate items with their features
            ranking_model_inputs = event_id_df.merge(
                events_df, 
                on="event_id", 
                how="inner",
            )
            logger.info(f"After merge: {len(ranking_model_inputs)} events")
            
            # Add user features
            try:
                user_features = self.users_fv.get_feature_vector(
                    {"user_id": user_id}, 
                    return_type="pandas",
                )
                logger.info(f"User features columns: {user_features.columns.tolist()}")
            except Exception as e:
                logger.error(f"Error getting user features: {str(e)}")
                # Continue with empty user features
                user_features = pd.DataFrame()
            
            # Add user features from query features list
            for feature in self.query_features:
                if feature in user_features.columns and len(user_features[feature].values) > 0:
                    ranking_model_inputs[feature] = user_features[feature].values[0]
                else:
                    # Add default value if feature not found
                    ranking_model_inputs[feature] = None
                    logger.warning(f"User feature {feature} not found, using None")
            
            # Check if we have all required features
            missing_features = [f for f in self.ranking_model_feature_names if f not in ranking_model_inputs.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Add missing features with None values
                for feature in missing_features:
                    ranking_model_inputs[feature] = None
            
            # Select only the features required by the ranking model
            available_features = [f for f in self.ranking_model_feature_names if f in ranking_model_inputs.columns]
            
            # If no available features, return empty result
            if not available_features:
                logger.error("No available features for ranking model")
                return {"inputs": [{"ranking_features": [], "event_ids": []}]}
            
            ranking_model_inputs = ranking_model_inputs[available_features]
            
            # Check for NaN values and replace with default values
            ranking_model_inputs = ranking_model_inputs.fillna(0)
