import uuid

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd


class DataLoader:
    def __init__(self, data_file_name: str, split_ratio: float = 0.1):
        self.data = pd.read_csv(data_file_name, delimiter=";")
        self.data["contact_key"] = self.data["contact_key"].apply(uuid.UUID)
        self.data["transaction"] = self.data["transaction"].apply(uuid.UUID)
        self.data["date_key"] = pd.to_datetime(self.data["date_key"])
        self.data["is_online"] = (
            self.data["is_online"]
            .apply(lambda val: True if val == "Y" else False)
            .astype(bool)
        )
        print(self.data.dtypes)
        self.split_data(split_ratio)

    def split_data(self, split_ratio: float) -> None:
        _unique_contact_keys = self.data["contact_key"].unique()


class DataVisualizer:
    @staticmethod
    def bar_plot_per_column(data_loader: DataLoader, column_names: list[str]):
        fig, axes = plt.subplots(1, len(column_names), tight_layout=True)
        axes = axes.tolist()
        for idx, column_name in enumerate(column_names):
            counts = data_loader.data[column_name].value_counts()
            counts.plot(x=column_name, subplots=True, ax=axes[idx])
        plt.show()


if __name__ == "__main__":
    src = "/home/micha/Downloads/ML Engineer/dataset.csv"
    column_names = ["contact_key", "a_brand_revenue", "other_brand_revenue"]
    print("Loading data...")
    dl = DataLoader(src)
    # print("Plotting data...")
    # DataVisualizer.bar_plot_per_column(dl, column_names)
