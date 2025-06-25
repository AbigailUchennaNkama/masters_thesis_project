
import os
import logging
from datetime import datetime
import pandas as pd
import hopsworks

class Transformer(object):

    def __init__(self):
        # 1) Connect to Hopsworks
        project = hopsworks.login()
        fs      = project.get_feature_store()
        mr      = project.get_model_registry()

        # 2) Feature views & groups
        # Events feature view (with event metadata & start_time)
        self.events_fv = fs.get_feature_view(name="events", version=1)
        # Candidate embeddings index for retrieval
        self.candidate_index = fs.get_feature_view(
            name="candidate_embeddings", version=1
        )
        # User feature view
        self.users_fv = fs.get_feature_view(name="users", version=1)

        # 3) Load ranking model from registry
        model = mr.get_model(name="weather_ranking_model", version=1)
        model_dir = model.download()
        # Assuming you saved your CatBoost as weather_ranking_model.pkl
        import joblib
        self.ranking_model = joblib.load(os.path.join(
            model_dir, "weather_ranking_model.pkl"
        ))

        # 4) Retrieve feature-view schema for ranking input names
        # (if you have a batch scoring feature view—otherwise just list them manually)
        self.ranking_fv = model.get_feature_view(init=False)
        self.ranking_fv.init_batch_scoring(1)
        self.ranking_features = [
            f.name for f in self.ranking_fv.schema
            if f.name != "interaction_label"
        ]

        logging.getLogger().setLevel(logging.INFO)


    def preprocess(self, inputs):
        # Unwrap KServe TF-Serving style input
        inst = inputs["instances"][0][0]
        user_id   = inst["user_id"]
        query_emb = inst["query_emb"]

        # --- Retrieval ---
        neighbors = self.candidate_index.find_neighbors(query_emb, k=100)
        candidate_ids = [n[0] for n in neighbors]

        # --- Fetch event metadata ---
        rows = [
            self.events_fv.get_feature_vector({"event_id": eid})
            for eid in candidate_ids
        ]
        df = pd.DataFrame(rows)

        # --- Filter past events if start_time exists ---
        if "start_time" in df.columns:
            df["start_date"] = pd.to_datetime(
                df["start_time"], errors="coerce"
            ).dt.date
            today = datetime.now().date()
            mask  = df["start_date"] >= today
            df    = df[mask]
            candidate_ids = [
                eid for eid, keep in zip(candidate_ids, mask.tolist())
                if keep
            ]

        if not candidate_ids:
            # no valid candidates → emit empty
            return {"instances":[{"ranking_features": [], "event_ids": []}]}

        # --- Enrich with user features ---
        uf = self.users_fv.get_feature_vector(
            {"user_id": user_id}, return_type="pandas"
        )
        for col in ["user_city", "age", "user_interests",
                    "indoor_outdoor_preference",
                    "user_weather_condition", "user_temperature", "user_precipitation"]:
            df[col] = uf.iloc[0][col]

        # --- Build ranking feature matrix ---
        avail = [c for c in self.ranking_features if c in df.columns]
        X = df[avail].fillna({
            **{n:0.0 for n in [
                "interaction_distance_to_event","temperature",
                "precipitation","attendance_rate",
                "user_temperature","user_precipitation"
            ]},
            **{s:"unknown" for s in [
                "event_type","event_city","title","weather_condition",
                "user_city","indoor_outdoor_preference",
                "user_interests","user_weather_condition"
            ]},
            **{"event_indoor_capability": False, "age": 30}
        }).values.tolist()

        #  Return the features + IDs for the predictor
        return {
            "instances": [{
                "ranking_features": X,
                "event_ids":        candidate_ids
            }]
        }


    def postprocess(self, query_outputs):
        # query_outputs["predictions"] → list of dicts
        return query_outputs["predictions"][0]
