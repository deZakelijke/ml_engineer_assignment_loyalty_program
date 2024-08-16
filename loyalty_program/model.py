import uuid

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB

from loyalty_program.data_loader import DataLoader


class Predictor:
    def __init__(self, data_file_name: str, test_size_ratio=0.2):
        self.data_loader = DataLoader(data_file_name)
        self.data_loader.create_train_val_split(test_size_ratio)

        self.models_to_test = {
            "logistic_regression": LogisticRegression,
            "naive_bayes": GaussianNB,
            "random_forest": RandomForestClassifier,
        }

    def test_all_models(self, n_folds: int = 3) -> str:
        best_model = ""
        best_model_score = -1.0
        for name, model in self.models_to_test.items():
            print(f"Running k-fold validation on: '{name}'")
            kfold = KFold(n_splits=n_folds)
            cross_validation_scores = cross_val_score(
                model(),
                *self.data_loader.train_data_tuple,
                cv=kfold,
                scoring="accuracy",
            )
            average_score = np.mean(cross_validation_scores)
            print(
                f"Score of model '{name}': {cross_validation_scores}, average: {average_score}"
            )
            if average_score > best_model_score:
                best_model_score = average_score
                best_model = name
        return best_model

    def train_model(self, model_name: str) -> None:
        print(f"Fitting model '{model_name}'")
        self.trained_model = self.models_to_test[model_name]().fit(
            *self.data_loader.train_data_tuple
        )

    def validate_trained_model(self) -> float:
        validation_score = self.trained_model.score(
            *self.data_loader.validation_data_tuple
        )
        print(f"Validation score is: {validation_score}")
        return validation_score

    def predict_on_test_data_and_save(self, save_path: str) -> None:
        test_features = self.data_loader.construct_features_to_predict_on()
        test_predictions = self.trained_model.predict(test_features)
        predictions_dataframe = pd.DataFrame(
            test_features["contact_key"].apply(lambda id: uuid.UUID(int=id))
        )
        predictions_dataframe["prediction"] = test_predictions
        predictions_dataframe.columns = ["contact_key", "prediction"]
        predictions_dataframe.to_csv(save_path, sep=";", header=True, index=False)


if __name__ == "__main__":
    src = "/home/micha/Downloads/ML Engineer/dataset.csv"
    save_path = "/home/micha/repos/ml_engineer_assignment_loyalty_program/results.csv"
    predictor = Predictor(src)
    # NOTE: tested different models and RF is the best
    # best_model = predictor.test_all_models()
    best_model = "random_forest"
    predictor.train_model(best_model)
    predictor.validate_trained_model()
    predictor.predict_on_test_data_and_save(save_path)
