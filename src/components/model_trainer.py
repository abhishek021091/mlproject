import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Forest": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),
                "XGBClassifier": XGBRegressor(),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }

            params={
                "Decision Forest":{
                    'criterion':['Squared Error','Friedman MSE','abosulte error','poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features:':['sqrt','log2'],
                },

                "Random Forest":{
                    # 'criterion':['Squared Error','Friedman MSE','abosulte error','poisson'],
                    # 'max_features:':['sqrt','log2',None],
                    'n_estimators':[8,16,32,64,128,256]
                },

                "Gradient Boosting":{
                    # 'loss':['squared_error','huber','absolute_error','quantile'],
                    'learning_rate':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['friedman_mse','squared_error'],
                    # 'max_features':['sqrt','log2','auto'],
                    'n_estimators':[8,16,32,64,128,256]
                    },

                "Linear Regression":{},

                "K-Neighbors Classifier":{
                    'n_neighbors':[5,7,9,11],
                    #'weights':['uniform','distance'],
                    #'algorithm':['auto','ball_tree','kd_tree','brute']
                },

                "XGBClassifier":{
                    'learning_rate':[0.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256]
                },

                "AdaBoost Classifier":{
                    'n_estimators':[8,16,32,64,128,256],
                    'learning_rate':[0.1,0.01,0.5,0.001]
                }

            }


            model_report:dict=evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                             models=models, param=params)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(best_model)

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )


            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test, predicted)
            return r2_square
        

        except Exception as e:
            raise CustomException(e,sys)