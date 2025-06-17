
import os
import joblib
import numpy as np
import logging

class Predict(object):
    def __init__(self):
        # Load the model from the environment variable
        model_path = os.environ["MODEL_FILES_PATH"]
        self.model = joblib.load(os.path.join(model_path, "weather_ranking_model.pkl"))
        logging.info("Model loaded successfully")

    def predict(self, inputs):
        # The inputs will be a dict with a list of lists under "instances"
        # Example: {"instances": [[feature1, feature2, ...], ...]}
        features = inputs["instances"]
        logging.info(f"Predict received {len(features)} instances")
        logging.info(f"Feature shape (if available): {np.array(features).shape if features else 'empty'}")

        # Predict probabilities for the positive class
        # (Assuming your model is a binary classifier with predict_proba)
        scores = self.model.predict_proba(features)[:, 1].tolist()

        # Return the scores (event_ids are not passed here, handle in postprocessing if needed)
        return {"predictions": scores}
