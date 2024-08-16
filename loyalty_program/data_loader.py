import uuid
from datetime import datetime
from typing import Final

import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

user_id_key: Final[str] = "contact_key"


class DataLoader:
    def __init__(self, data_file_name: str):
        print("Loading data...")
        self.raw_data = pd.read_csv(data_file_name, delimiter=";")
        self.raw_data[user_id_key] = (
            self.raw_data[user_id_key].apply(uuid.UUID).apply(int)
        )
        # NOTE: do we need the "transaction" data?
        # self.data["transaction"] = self.data["transaction"].apply(uuid.UUID)
        self.raw_data["date_key"] = pd.to_datetime(self.raw_data["date_key"])
        self.raw_data["is_online"] = (
            self.raw_data["is_online"]
            .apply(lambda val: True if val == "Y" else False)
            .astype(bool)
        )

        self.extend_features()
        self.split_data_at_cutoff()
        self.user_dataframe = pd.DataFrame(
            self.past_purchase_data[user_id_key].unique()  # type: ignore
        )
        self.user_dataframe.columns = [user_id_key]
        self.construct_features_to_train_with()
        self.construct_classification_targets()

    def group_transactions_per_week(self) -> None:
        self.raw_data.groupby([user_id_key, "date_key"])

    def split_data_at_cutoff(self) -> None:
        # Hardcoded last week of the dataset
        cutoff_date = datetime(year=2021, month=12, day=27)
        self.raw_data.sort_values(by="date_key", inplace=True, axis="index")
        self.past_purchase_data = self.raw_data[
            self.raw_data["date_key"] < cutoff_date
        ].reset_index(drop=True)
        self.future_purchase_data = self.raw_data[
            self.raw_data["date_key"] >= cutoff_date
        ].reset_index(drop=True)

    def construct_features_to_train_with(self) -> None:
        # Add next day that user would make a purchase
        # Then get a shifted series of days between purchases, the mean and the std
        future_first_purchase = (
            self.future_purchase_data.groupby(user_id_key)["date_key"]
            .min()
            .reset_index()
        )
        future_first_purchase.columns = [user_id_key, "first_purchase_date"]
        past_last_purchase = (
            self.past_purchase_data.groupby(user_id_key)["date_key"].max().reset_index()
        )
        past_last_purchase.columns = [user_id_key, "last_purchase_date"]

        purchase_dates = pd.merge(
            past_last_purchase, future_first_purchase, on=user_id_key, how="left"
        )
        purchase_dates["next_purchase_day"] = (
            purchase_dates["first_purchase_date"] - purchase_dates["last_purchase_date"]
        ).dt.days

        self.user_dataframe = pd.merge(
            self.user_dataframe,
            purchase_dates[[user_id_key, "next_purchase_day"]],
            on=user_id_key,
            how="left",
        )
        self.user_dataframe = self.user_dataframe.fillna(999)
        day_order = self.generate_previous_purchase_day_shift_window(
            self.past_purchase_data  # type: ignore
        )
        day_diff = self.generate_day_difference_mean_and_std(day_order)
        day_order = pd.merge(day_order, day_diff, on=user_id_key)
        self.user_dataframe = pd.merge(
            self.user_dataframe,
            day_order[
                [
                    user_id_key,
                    "day_difference",
                    "day_difference_2",
                    "day_difference_3",
                    "day_difference_mean",
                    "day_difference_std",
                ]
            ],
            on=user_id_key,
        )

    def construct_features_to_predict_on(self) -> pd.DataFrame:
        cutoff_date = datetime(year=2022, month=1, day=3)
        prediction_data = self.raw_data[
            self.raw_data["date_key"] < cutoff_date
        ].reset_index(drop=True)
        prediction_user_dataframe = pd.DataFrame(prediction_data[user_id_key].unique())  # type: ignore
        prediction_user_dataframe.columns = [user_id_key]

        day_order = self.generate_previous_purchase_day_shift_window(
            prediction_data  # type: ignore
        )
        day_diff = self.generate_day_difference_mean_and_std(day_order)
        day_order = pd.merge(day_order, day_diff, on=user_id_key)
        prediction_user_dataframe = pd.merge(
            prediction_user_dataframe,  # type: ignore
            day_order[
                [
                    user_id_key,
                    "day_difference",
                    "day_difference_2",
                    "day_difference_3",
                    "day_difference_mean",
                    "day_difference_std",
                ]
            ],
            on=user_id_key,
        )
        return prediction_user_dataframe

    def generate_previous_purchase_day_shift_window(
        self, past_purchase_data: pd.DataFrame
    ) -> pd.DataFrame:
        # Add number of days between the last three purchases
        day_order = past_purchase_data[[user_id_key, "date_key"]]
        day_order["purchase_day"] = pd.to_datetime(
            past_purchase_data["date_key"].dt.date
        )
        day_order = day_order.sort_values([user_id_key, "date_key"])  # type: ignore
        day_order = day_order.drop_duplicates(
            subset=[user_id_key, "purchase_day"], keep="first"
        )  # Drop duplicates if users made multiple purchases on the same day

        day_order["previous_purchase_day"] = day_order.groupby(user_id_key)[
            "purchase_day"
        ].shift(1)
        day_order["previous_2_purchase_day"] = day_order.groupby(user_id_key)[
            "purchase_day"
        ].shift(2)
        day_order["previous_3_purchase_day"] = day_order.groupby(user_id_key)[
            "purchase_day"
        ].shift(3)
        day_order["day_difference"] = (
            day_order["purchase_day"] - day_order["previous_purchase_day"]
        ).dt.days
        day_order["day_difference_2"] = (
            day_order["purchase_day"] - day_order["previous_2_purchase_day"]
        ).dt.days
        day_order["day_difference_3"] = (
            day_order["purchase_day"] - day_order["previous_2_purchase_day"]
        ).dt.days

        day_order = day_order.drop_duplicates(subset=[user_id_key], keep="last")
        day_order = day_order.fillna(99)
        return day_order

    @staticmethod
    def generate_day_difference_mean_and_std(day_order: pd.DataFrame) -> pd.DataFrame:
        day_diff = (
            day_order.groupby(user_id_key)
            .agg({"day_difference": ["mean", "std"]})
            .reset_index()
        )
        day_diff.columns = [user_id_key, "day_difference_mean", "day_difference_std"]
        day_diff = day_diff.fillna(99)
        return day_diff

    def construct_classification_targets(self) -> None:
        # Set target of 0 or 1 depending on whether or not a user makes
        # a puchase within seven days
        self.class_targets = self.user_dataframe.copy()
        self.class_targets["purchase_next_week"] = 0
        self.class_targets.loc[
            self.class_targets["next_purchase_day"] <= 7, "purchase_next_week"
        ] = 1

    def extend_features(self) -> None:
        # Only used for the plots. For actual predictions we don't care about
        # the purchase values
        self.raw_data["total_quantity"] = (
            self.raw_data["core_quantity"] + self.raw_data["other_quantity"]
        )
        self.raw_data["total_revenue"] = (
            self.raw_data["core_revenue"] + self.raw_data["other_revenue"]
        )

    def create_train_val_split(self, test_size_ratio=0.2) -> None:
        self.class_targets = self.class_targets.drop("next_purchase_day", axis=1)
        features, targets = (
            self.class_targets.drop("purchase_next_week", axis=1),
            self.class_targets["purchase_next_week"],
        )
        (
            self.features_train,
            self.features_val,
            self.targets_train,
            self.targets_val,
        ) = train_test_split(features, targets, test_size=test_size_ratio)

    @property
    def train_data_tuple(self) -> tuple:
        if not hasattr(self, "features_train") or not hasattr(self, "targets_train"):
            raise ValueError("Create the train-test split first")
        return self.features_train, self.targets_train

    @property
    def validation_data_tuple(self) -> tuple:
        if not hasattr(self, "features_val") or not hasattr(self, "targets_val"):
            raise ValueError("Create the train-test split first")
        return self.features_val, self.targets_val


class DataVisualizer:
    @staticmethod
    def plot_number_of_sales_per_customer(data_loader: DataLoader) -> None:
        counts = data_loader.raw_data[user_id_key].value_counts()
        counts.plot()
        plt.show()

    @staticmethod
    def plot_total_sales_volume_per_week(data_loader: DataLoader) -> None:
        weekly_quantity = (
            data_loader.raw_data["total_quantity"]
            .groupby(data_loader.raw_data["date_key"].dt.to_period("W"))
            .sum()
        )
        weekly_revenue = (
            data_loader.raw_data["total_revenue"]
            .groupby(data_loader.raw_data["date_key"].dt.to_period("W"))
            .sum()
        )
        total_min = min(min(weekly_quantity), min(weekly_revenue))
        total_max = max(max(weekly_quantity), max(weekly_revenue))
        plt.ylim(total_min, total_max)
        plt.plot(weekly_quantity.to_numpy())
        plt.plot(weekly_revenue.to_numpy())
        plt.show()

    @staticmethod
    def plot_average_sales_per_customer(data_loader: DataLoader) -> None:
        average_quantity_per_customer = (
            data_loader.raw_data["total_quantity"]
            .groupby(data_loader.raw_data[user_id_key])
            .mean()
        ).to_numpy()
        average_quantity_per_customer = np.sort(average_quantity_per_customer)

        plt.plot(average_quantity_per_customer)
        plt.show()

    @staticmethod
    def plot_spread_between_first_and_last_purchase(data_loader: DataLoader) -> None:
        grouped_contacts = data_loader.raw_data.groupby(
            data_loader.raw_data[user_id_key]
        )
        date_ranges = (
            grouped_contacts["date_key"].last() - grouped_contacts["date_key"].first()
        ).dt.days / 7
        date_ranges.sort_values(inplace=True)
        date_ranges.plot()
        plt.show()


if __name__ == "__main__":
    src = "/home/micha/Downloads/ML Engineer/dataset.csv"
    dl = DataLoader(src)
    print("Plotting data...")
    # DataVisualizer.plot_total_sales_volume_per_week(dl)
    # DataVisualizer.plot_number_of_sales_per_customer(dl)
    # DataVisualizer.plot_average_sales_per_customer(dl)
    # DataVisualizer.plot_spread_between_first_and_last_purchase(dl)
