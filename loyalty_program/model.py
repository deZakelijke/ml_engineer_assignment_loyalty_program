from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

from loyalty_program.data_loader import DataLoader


class Predictor:
    def __init__(self, data_file_name: str, test_size_ratio=0.2):
        self.data_loader = DataLoader(data_file_name)
        self.data_loader.create_train_val_split(test_size_ratio)

        self.models_to_test = {"logistic_regression": LogisticRegression()}

    def test_all_models(self, n_folds: int = 3) -> None:
        for name, model in self.models_to_test.items():
            print(f"Running k-fold validation on: '{name}'")
            kfold = KFold(n_splits=n_folds)
            cross_validation_result = cross_val_score(
                model, *self.data_loader.train_data, cv=kfold, scoring="accuracy"
            )
            print(f"Score of model '{name}': {cross_validation_result}")


if __name__ == "__main__":
    src = "/home/micha/Downloads/ML Engineer/dataset.csv"
    predictor = Predictor(src)
    predictor.test_all_models()
