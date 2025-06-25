
import os
import pandas as pd
import logging
from datetime import datetime

import hopsworks

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Transformer(object):
    
    def __init__(self):
        try:
            # Connect to Hopsworks
            project = hopsworks.connection().get_project()
            self.fs = project.get_feature_store()
            
            # Get feature views
            self.events_fv = self.fs.get_feature_view(name="events", version=1)
            self.users_fv = self.fs.get_feature_view(name="users", version=1)
            self.candidate_index = self.fs.get_feature_view(name="candidate_embeddings", version=1)
            
            # Get event features
            self.event_features = [feat.name for feat in self.events_fv.schema]
            
            # Get model
            mr = project.get_model_registry()
            model = mr.get_model(name="weather_ranking_model", version=1)
            
            # Get feature names
            input_schema = model.model_schema["input_schema"]["columnar_schema"]
            self.ranking_features = [feat["name"] for feat in input_schema]
            
            logger.info("Transformer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing transformer: {str(e)}")
            raise
    
    def preprocess(self, inputs):
        try:
            logger.info(f"Input structure: {inputs}")
            
            # Extract user_id from inputs - handle both formats
            if isinstance(inputs["instances"], list):
                user_id = inputs["instances"][0]["user_id"]
                query_emb = inputs["instances"][0]["query_emb"]
            else:
                user_id = inputs["instances"]["user_id"]
                query_emb = inputs["instances"]["query_emb"]
            
            logger.info(f"Processing for user_id: {user_id}")
            
            # Get candidate events
            neighbors = self.candidate_index.find_neighbors(query_emb, k=20)
            event_ids = [neighbor[0] for neighbor in neighbors]
            
            # Get user features
            user_features = self.users_fv.get_feature_vector(
                {"user_id": user_id}, 
                return_type="pandas"
            )
            
            # Get event features
            events_data = []
            for event_id in event_ids:
                try:
                    event = self.events_fv.get_feature_vector({"event_id": event_id})
                    events_data.append(event)
                except Exception as e:
                    logger.warning(f"Could not get features for event {event_id}: {str(e)}")
            
            # Create DataFrame
            if not events_data:
                logger.warning("No valid events found")
                return {"inputs": [{"ranking_features": [], "event_ids": []}]}
            
            events_df = pd.DataFrame(data=events_data, columns=self.event_features)
            
            # Add user features to each event
            for col in ["age", "gender", "country"]:
                if col in user_features.columns:
                    events_df[col] = user_features[col].values[0]
            
            # Select only required features
            available_features = [f for f in self.ranking_features if f in events_df.columns]
            ranking_inputs = events_df[available_features]
            
            logger.info(f"Prepared {len(ranking_inputs)} events for ranking")
            
            return {
                "inputs": [{
                    "ranking_features": ranking_inputs.values.tolist(),
                    "event_ids": event_ids
                }]
            }
        except Exception as e:
            logger.error(f"Error in preprocess: {str(e)}")
            # Return empty result on error
            return {"inputs": [{"ranking_features": [], "event_ids": []}]}
    
    def postprocess(self, outputs):
        try:
            logger.info(f"Output structure: {outputs}")
            
            if "predictions" not in outputs:
                logger.warning("No predictions in output")
                return {"ranking": []}
            
            preds = outputs["predictions"]
            
            if "scores" not in preds or "event_ids" not in preds:
                logger.warning("Missing scores or event_ids in predictions")
                return {"ranking": []}
            
            # Create ranking tuples
            ranking = list(zip(preds["scores"], preds["event_ids"]))
            
            # Sort by score
            ranking.sort(reverse=True)
            
            logger.info(f"Returning {len(ranking)} ranked events")
            
            return {"ranking": ranking}
        except Exception as e:
            logger.error(f"Error in postprocess: {str(e)}")
            return {"ranking": []}
