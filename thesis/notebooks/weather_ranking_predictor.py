
import os
import joblib
import numpy as np
import logging

class Predict(object):
    
    def __init__(self):
        # List the directory contents to debug
        artifact_path = os.environ["ARTIFACT_FILES_PATH"]
        model_path = os.environ["MODEL_FILES_PATH"]
        logging.info(f"Artifact path contents: {os.listdir(artifact_path)}")
        logging.info(f"Model path contents: {os.listdir(model_path)}")
        
        # Try loading from MODEL_FILES_PATH instead
        self.model = joblib.load(os.path.join(model_path, "weather_ranking_model.pkl"))

    def predict(self, inputs):
        try:
            # Add detailed logging to help diagnose the issue
            logging.info(f"Input structure: {inputs}")
            
            # Handle different input formats
            if isinstance(inputs, list) and len(inputs) > 0:
                # If inputs is a list, get the first element
                input_data = inputs[0]
            else:
                # If inputs is not a list, use it directly
                input_data = inputs
            
            # Check if input_data has the expected keys
            if not isinstance(input_data, dict) or "ranking_features" not in input_data or "event_ids" not in input_data:
                logging.error(f"Invalid input format: {input_data}")
                return {"scores": [], "event_ids": []}
            
            # Extract ranking features and event IDs
            features = input_data["ranking_features"]
            event_ids = input_data["event_ids"]
            
            # Log the extracted features
            logging.info(f"Features: {features}")
            logging.info(f"Event IDs: {event_ids}")
            
            # If features is empty, return empty results
            if not features:
                return {"scores": [], "event_ids": []}
            
            # Predict probabilities for the positive class
            scores = self.model.predict_proba(features).tolist()
            
            # Get scores of positive class
            scores = np.asarray(scores)[:,1].tolist() 
            
            # Return the predicted scores along with the corresponding event IDs
            return {
                "scores": scores, 
                "event_ids": event_ids,
            }
        except Exception as e:
            # Add detailed logging to help diagnose the issue
            logging.error(f"Error in predict: {str(e)}")
            logging.error(f"Input structure: {inputs}")
            # Return empty result on error
            return {"scores": [], "event_ids": []}
