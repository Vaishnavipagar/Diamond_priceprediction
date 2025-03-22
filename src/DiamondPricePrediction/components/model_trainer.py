import pandas as pd 
import numpy as np
import os
import sys
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import custom_exception
from dataclasses import dataclass
from src.DiamondPricePrediction.utils.utils import save_object
from src.DiamondPricePrediction.utils.utils import evaluate_model

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

@dataclass
class ModelTrainerConfig:
    trained_model_path:str=os.path.join("artifacts","model.pkl")
    

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initate_model_trainig(self,Train_array,Test_array):
        try:
            logging.info("spltitng the dependant and independent feature by train and test data")
            X_train, y_train, X_test, y_test =(
                Train_array[:,:-1],
                Train_array[:,-1],
                Test_array[:,:-1],
                Test_array[:,-1]
            )
             
            model={
                "Linear Regression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "ElasticNet":ElasticNet()
            }
            
            models_report:dict=evaluate_model(X_train,y_train,X_test,y_test,model)
            print(models_report)
            print("\n======================================================================================\n")
            logging.info(f'report : {models_report}')
            
            #to get best model
            best_model_score = max(models_report.values())
            
            best_model_name=list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]
            
            best_model=model[best_model_name]
            print(f'Best Model Found :{best_model_name} , r2_score : {best_model_score}')
            
            logging.info(f'best model found :{best_model_name} , r2_score;{best_model_score}')
            
            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )
            
            
        except Exception as e:
            raise custom_exception(e,sys)