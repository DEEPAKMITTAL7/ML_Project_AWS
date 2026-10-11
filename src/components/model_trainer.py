import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,r2_score)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LogisticRegression(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(f"Best found model on both training and testing dataset: {best_model}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
