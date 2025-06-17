
import os
import pandas as pd
import datetime
import hopsworks
import logging

class Transformer(object):
    def __init__(self):
        # Connect to Hopsworks
        project = hopsworks.connection().get_project()
        self.fs = project.get_feature_store()
        self.events_fv = self.fs.get_feature_view(name="events", version=1)
        self.event_features = [feat.name for feat in self.events_fv.schema]
        self.users_fv = self.fs.get_feature_view(name="users", version=1)
        self.candidate_index = self.fs.get_feature_view(name="candidate_embeddings", version=1)
        mr = project.get_model_registry()
        model = mr.get_model(name="weather_ranking_model", version=1)
        input_schema = model.model_schema["input_schema"]["columnar_schema"]
        self.ranking_model_feature_names = [feat["name"] for feat in input_schema]
        self.current_event_ids = []

    def preprocess(self, inputs):
        print("Transformer preprocess input:", inputs)
        if isinstance(inputs, dict) and "instances" in inputs:
            instance = inputs["instances"][0]
        else:
            instance = inputs[0] if isinstance(inputs, list) else inputs

        user_id = instance.get("user_id")
        query_emb = instance.get("query_emb")
        if user_id is None or query_emb is None:
            raise ValueError("user_id or query_emb missing in input")

        # Find candidate event IDs
        neighbors = self.candidate_index.find_neighbors(query_emb, k=100)
        candidate_ids = [n[0] for n in neighbors]

        # Retrieve event data
        events_data = [self.events_fv.get_feature_vector({"event_id": eid}) for eid in candidate_ids]
        events_df = pd.DataFrame(events_data, columns=self.event_features)

        # Filter to future events only
        current_date = datetime.datetime.now().date()
        events_df["start_date"] = pd.to_datetime(events_df["start_time"]).dt.date
        valid_events = events_df[events_df["start_date"] >= current_date]

        # If no valid events, return empty instances and store empty event_ids
        if valid_events.empty:
            self.current_event_ids = []
            print("No valid events found for user", user_id)
            return {"instances": []}

        # Merge user features
        user_features = self.users_fv.get_feature_vector({"user_id": user_id}, return_type="pandas")
        required_user_cols = [
            "user_id", "user_city", "age", "user_interests", "indoor_outdoor_preference",
            "user_weather_condition", "user_temperature", "user_precipitation"
        ]
        for col in required_user_cols:
            if col not in user_features.columns:
                raise ValueError(f"Missing user feature: {col}")
        valid_events[required_user_cols] = user_features[required_user_cols].values[0]

        # Select only the features required by the ranking model
        ranking_features = valid_events[self.ranking_model_feature_names]

        # Store event IDs for postprocess
        self.current_event_ids = valid_events["event_id"].tolist()
        print("Number of valid events:", len(self.current_event_ids))
        print("Output from preprocess:", {"instances": ranking_features.values.tolist()})

        return {"instances": ranking_features.values.tolist()}

    def postprocess(self, outputs):
        print("Transformer postprocess input:", outputs)
        predictions = outputs.get("predictions", [])
        if len(predictions) != len(self.current_event_ids):
            print("Mismatch between predictions and event IDs")
            raise ValueError("Mismatch between predictions and event IDs")
        ranking = list(zip(predictions, self.current_event_ids))
        ranking.sort(reverse=True)
        print("Postprocess returning", len(ranking), "ranked results")
        # Return both ranking and debug info
        return {
            "ranking": ranking,
            "debug": f"Number of valid events: {len(self.current_event_ids)}"
        }

