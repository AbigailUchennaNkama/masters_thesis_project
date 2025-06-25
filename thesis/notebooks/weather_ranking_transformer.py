# weather_ranking_transformer.py
import os
import pandas as pd
import hopsworks
from datetime import datetime

class Transformer:

    def __init__(self):
        project = hopsworks.connection().get_project()
        self.fs = project.get_feature_store()

        # Feature Views
        self.events_fv       = self.fs.get_feature_view(name="events",               version=1)
        self.users_fv        = self.fs.get_feature_view(name="users",                version=1)
        self.interactions_fv = self.fs.get_feature_view(name="interactions",         version=1)
        self.candidate_index = self.fs.get_feature_view(name="candidate_embeddings", version=1)

        # Model schema
        mr = project.get_model_registry()
        model = mr.get_best_model(
            name="weather_ranking_model",
            metric="fscore",
            direction="max"
        )
        schema = model.model_schema["input_schema"]["columnar_schema"]
        self.feature_names = [f["name"] for f in schema]

    def preprocess(self, inputs):
        # Unpack two nested lists to get the payload dict
        inst      = inputs["instances"][0][0]
        query_emb = inst["query_emb"]

        # 1) k-NN retrieval of candidate event IDs
        neighbors = self.candidate_index.find_neighbors(query_emb, k=100)
        event_ids = [n[0] for n in neighbors]

        # 2) Batch fetch, parse times, and filter out past events
        events_df = self.events_fv.get_batch_data(
            primary_key=True,
            event_time=True,
            read_options={"use_arrow_flight": True}
        )
        events_df["start_time"] = pd.to_datetime(events_df["start_time"])
        now       = pd.Timestamp(datetime.utcnow())
        future    = events_df[events_df["start_time"] >= now]
        candidates_df = future[future["event_id"].isin(event_ids)]

        # 3) Batch fetch user & interaction features
        user_df = self.users_fv.get_batch_data(
            primary_key=True,
            read_options={"use_arrow_flight": True}
        )
        int_df  = self.interactions_fv.get_batch_data(
            primary_key=True,
            read_options={"use_arrow_flight": True}
        )

        # 4) Assemble DataFrame
        rank_df = (
            candidates_df
            .merge(int_df, on="event_id")
            .merge(user_df, on="user_id")
        )[self.feature_names]

        # 5) Return under "inputs" for the Predictor
        return {
            "inputs": [{
                "ranking_features": rank_df.values.tolist(),
                "event_ids":        candidates_df["event_id"].tolist()
            }]
        }

    def postprocess(self, outputs):
        # outputs is the Predictor’s return value
        return outputs


