from src.DiamondPricePrediction.components.data_ingestion import data_ingestion
from src.DiamondPricePrediction.components.data_transformation import DataTransformation
from src.DiamondPricePrediction.components.model_trainer import ModelTrainer
#from src.DiamondPricePrediction.components.model_evaluation import ModelEvaluation

import os
import sys
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import custom_exception
import pandas as pd 


obj=data_ingestion()
train_data_path,test_data_path=obj.intiate_data_ingestion()


data_transform_obj=DataTransformation()
train_arr,test_arr=data_transform_obj.initialize_data_transformation(train_data_path,test_data_path)


model_trainer_obj=ModelTrainer()
model_trainer_obj.initate_model_trainig(train_arr,test_arr)

    
#model_eval_obj = ModelEvaluation()
#model_eval_obj.initiate_model_evaluation(train_arr,test_arr)