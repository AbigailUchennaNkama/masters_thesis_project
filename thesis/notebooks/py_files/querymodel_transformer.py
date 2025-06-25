
import os
import numpy as np
import pandas as pd
from datetime import datetime

import hopsworks

import logging


class Transformer(object):
    
    def __init__(self): 
        # Connect to the Hopsworks
        project = hopsworks.connection().get_project()
        ms = project.get_model_serving()
        
        # Retrieve the 'users' feature view
        fs = project.get_feature_store()
        self.users_fv = fs.get_feature_view(
            name="users", 
            version=1,
        )
        # Retrieve the ranking deployment 
        self.ranking_server = ms.get_deployment("weatherrankingdeployment")
        
    
    def preprocess(self, inputs):
        # Check if the input data contains a key named "instances"
        # and extract the actual data if present
        inputs = inputs["instances"] if "instances" in inputs else inputs

        # Extract user_id from the inputs
        user_id = inputs["user_id"]

        # Get user features
        user_features = self.users_fv.get_feature_vector(
            {"user_id": user_id}, 
            return_type="pandas",
        )

        # Enrich inputs with user features
        inputs["user_city"] = user_features['user_city'].values[0]
        inputs["age"] = user_features['age'].values[0] 
        inputs["user_interests"] = user_features['user_interests'].values[0]
        
        return {
            "instances": [inputs]
        }
    
    def postprocess(self, outputs):
        # Return ordered ranking predictions
        return {
            "predictions": self.ranking_server.predict({"instances": outputs["predictions"]}),
        }
