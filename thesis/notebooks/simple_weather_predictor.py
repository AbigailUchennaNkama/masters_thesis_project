
import os
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predict(object):
    
    def __init__(self):
        try:
            # Get model path
            model_path = os.environ["MODEL_FILES_PATH"]
            logger.info(f"Model path: {model_path}")
            logger.info(f"Files in model path: {os.listdir(model_path)}")
            
            # Load model
            model_file = os.path.join(model_path, "weather_ranking_model.pkl")
            self.model = joblib.load(model_file)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, inputs):
        try:
            logger.info(f"Input structure: {inputs}")
            
            # Handle empty inputs
            if not inputs or len(inputs) == 0:
                logger.warning("Empty inputs received")
                return {"scores": [], "event_ids": []}
            
            # Extract features and event IDs
            if "ranking_features" not in inputs[0] or "event_ids" not in inputs[0]:
                logger.warning("Missing required fields in inputs")
                return {"scores": [], "event_ids": []}
            
            features = inputs[0]["ranking_features"]
            event_ids = inputs[0]["event_ids"]
            
            # Handle empty features
            if not features:
                logger.warning("Empty features received")
                return {"scores": [], "event_ids": []}
            
            logger.info(f"Predicting for {len(features)} events")
            
            # Make predictions
            scores = self.model.predict_proba(features).tolist()
            scores = np.asarray(scores)[:,1].tolist()
            
            return {
                "scores": scores,
                "event_ids": event_ids
            }
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            return {"scores": [], "event_ids": []}
