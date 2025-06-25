
# weather_ranking_predictor.py
import os
import joblib
import pandas as pd
from catboost import Pool

class Predict:

    def __init__(self):
        model_dir = os.environ["MODEL_FILES_PATH"]
        self.model = joblib.load(os.path.join(model_dir, "weather_ranking_model.pkl"))
        self.feature_names = self.model.feature_names_

    def predict(self, inputs):
        # --- Fix is here: unpack from inputs["inputs"][0], not inputs[0] ---
        batch     = inputs["inputs"][0]
        features  = batch["ranking_features"]
        event_ids = batch["event_ids"]

        # Construct DataFrame with correct columns
        df = pd.DataFrame(features, columns=self.feature_names)

        # Cast categorical columns to string
        cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
        for c in cat_cols:
            df[c] = df[c].astype(str)

        # Create Pool & score
        pool  = Pool(data=df, cat_features=cat_cols)
        probs = self.model.predict_proba(pool)
        pos_idx = list(self.model.classes_).index(1)
        scores  = probs[:, pos_idx].tolist()

        # Return final outputs
        return {
            "scores":    scores,
            "event_ids": event_ids
        }



