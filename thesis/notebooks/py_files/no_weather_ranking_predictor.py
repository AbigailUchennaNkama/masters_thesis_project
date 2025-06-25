
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
        self.model = joblib.load(os.path.join(model_path, "no_weather_ranking_model.pkl"))

    def predict(self, inputs):
        # Extract ranking features and event IDs from the inputs
        features = inputs[0]["ranking_features"]
        event_ids = inputs[0]["event_ids"]
        
        # Log the extracted features
        logging.info("predict -> " + str(features))

        # Predict probabilities for the positive class
        scores = self.model.predict_proba(features).tolist()
        
        # Get scores of positive class
        scores = np.asarray(scores)[:,1].tolist() 

        # Return the predicted scores along with the corresponding event IDs
        return {
            "scores": scores, 
            "event_ids": event_ids,
        }
