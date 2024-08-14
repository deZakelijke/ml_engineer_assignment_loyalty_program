import uuid
from datetime import timedelta

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class DataLoader:
    def __init__(self, data_file_name: str, split_ratio: float = 0.9):
        self.data = pd.read_csv(data_file_name, delimiter=";")
        self.data["contact_key"] = self.data["contact_key"].apply(uuid.UUID)
        self.data["transaction"] = self.data["transaction"].apply(uuid.UUID)
        self.data["date_key"] = pd.to_datetime(self.data["date_key"])
        self.data["is_online"] = (
            self.data["is_online"]
            .apply(lambda val: True if val == "Y" else False)
            .astype(bool)
        )
        self.extend_features()
        self.split_data(split_ratio)
        self.user_dataframe = pd.DataFrame(self.training_data["contact_key"].unique())  # type: ignore
        self.user_dataframe.columns = ["contact_key"]

    def split_data(self, split_ratio: float) -> None:
        if split_ratio <= 0.0 or split_ratio >= 1.0:
            raise ValueError("The spit ratio should be between zero and one")
        cutoff_delta = int(
            (self.data["date_key"].max() - self.data["date_key"].min()).days
            * split_ratio
        )
        cutoff_date = self.data["date_key"].min() + timedelta(days=cutoff_delta)
        self.data.sort_values(by="date_key", inplace=True, axis="index")
        self.training_data = self.data[self.data["date_key"] < cutoff_date].reset_index(
            drop=True
        )
        self.validatation_data = self.data[
            self.data["date_key"] >= cutoff_date
        ].reset_index(drop=True)

    def extend_features(self) -> None:
        self.data["total_quantity"] = (
            self.data["core_quantity"] + self.data["other_quantity"]
        )
        self.data["total_revenue"] = (
            self.data["core_revenue"] + self.data["other_revenue"]
        )


class DataVisualizer:
    @staticmethod
    def plot_number_of_sales_per_customer(data_loader: DataLoader) -> None:
        counts = data_loader.data["contact_key"].value_counts()
        counts.plot()
        plt.show()

    @staticmethod
    def plot_total_sales_volume_per_week(data_loader: DataLoader) -> None:
        weekly_quantity = (
            data_loader.data["total_quantity"]
            .groupby(data_loader.data["date_key"].dt.to_period("W"))
            .sum()
        )
        weekly_revenue = (
            data_loader.data["total_revenue"]
            .groupby(data_loader.data["date_key"].dt.to_period("W"))
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
            data_loader.data["total_quantity"]
            .groupby(data_loader.data["contact_key"])
            .mean()
        ).to_numpy()
        average_quantity_per_customer = np.sort(average_quantity_per_customer)

        plt.plot(average_quantity_per_customer)
        plt.show()

    @staticmethod
    def plot_spread_between_first_and_last_purchase(data_loader: DataLoader) -> None:
        grouped_contacts = data_loader.data.groupby(data_loader.data["contact_key"])
        date_ranges = (
            grouped_contacts["date_key"].last() - grouped_contacts["date_key"].first()
        ).dt.days / 7
        date_ranges.sort_values(inplace=True)
        date_ranges.plot()
        plt.show()


if __name__ == "__main__":
    src = "/home/micha/Downloads/ML Engineer/dataset.csv"
    print("Loading data...")
    dl = DataLoader(src)
    print("Plotting data...")
    # DataVisualizer.plot_total_sales_volume_per_week(dl)
    # DataVisualizer.plot_number_of_sales_per_customer(dl)
    # DataVisualizer.plot_average_sales_per_customer(dl)
    # DataVisualizer.plot_spread_between_first_and_last_purchase(dl)
