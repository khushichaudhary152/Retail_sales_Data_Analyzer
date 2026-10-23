import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class RetailAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        print("\n========== DATASET INFORMATION ==========")
        print("Dataset loaded successfully!")
        print("\nFirst five rows:")
        print(self.data.head())
        print("\nDataset shape:", self.data.shape)
        print("\nColumn names:")
        print(self.data.columns.tolist())

    def clean_data(self):
        required_columns = ["Date", "Product", "Category", "Price", "Quantity Sold", "Total Sales"]
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.data["Date"] = pd.to_datetime(self.data["Date"], errors="coerce")
        numeric_columns = ["Price", "Quantity Sold", "Total Sales"]

        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        self.data["Product"] = self.data["Product"].fillna("Unknown")
        self.data["Category"] = self.data["Category"].fillna("Unknown")
        self.data["Price"] = self.data["Price"].fillna(self.data["Price"].median())
        self.data["Quantity Sold"] = self.data["Quantity Sold"].fillna(0)
        self.data["Total Sales"] = self.data["Total Sales"].fillna(0)
        self.data = self.data.dropna(subset=["Date"])
        print("\nData cleaning completed!")
        print("\nMissing values after cleaning:")
        print(self.data.isnull().sum())

    def calculate_metrics(self):
        sales = self.data["Total Sales"].to_numpy()
        quantities = self.data["Quantity Sold"].to_numpy()
        total_sales = np.sum(sales)
        average_sales = np.mean(sales) if len(sales) else 0
        maximum_sales = np.max(sales) if len(sales) else 0
        minimum_sales = np.min(sales) if len(sales) else 0
        total_quantity = np.sum(quantities)
        popular_product = self.data.groupby("Product")["Quantity Sold"].sum().idxmax() if not self.data.empty else "N/A"

        print("\n========== SALES ANALYSIS ==========")
        print(f"Total Sales: {total_sales:.2f}")
        print(f"Average Sales: {average_sales:.2f}")
        print(f"Maximum Sales: {maximum_sales:.2f}")
        print(f"Minimum Sales: {minimum_sales:.2f}")
        print(f"Total Quantity Sold: {total_quantity}")
        print(f"Most Popular Product: {popular_product}")
        print("====================================")

    def display_summary(self):
        print("\n========== SALES SUMMARY ==========")
        print("Total Records:", len(self.data))
        print("Total Categories:", self.data["Category"].nunique())
        print("Total Products:", self.data["Product"].nunique())
        print("Total Sales:", round(self.data["Total Sales"].sum(), 2))
        print("Average Sales:", round(self.data["Total Sales"].mean(), 2))
        print("===================================")

    def filter_by_category(self, category):
        filtered_data = self.data[self.data["Category"].astype(str).str.lower() == category.lower()]
        print(f"\nRecords for category: {category}")
        print(filtered_data.head())
        return filtered_data

    def category_sales_chart(self):
        category_sales = self.data.groupby("Category")["Total Sales"].sum()
        plt.figure(figsize=(10, 6))
        plt.bar(category_sales.index.astype(str), category_sales.values)
        plt.title("Category-wise Total Sales")
        plt.xlabel("Category")
        plt.ylabel("Total Sales")
        plt.xticks(rotation=30)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def sales_trend_chart(self):
        daily_sales = self.data.groupby("Date")["Total Sales"].sum().sort_index()
        plt.figure(figsize=(12, 6))
        plt.plot(daily_sales.index, daily_sales.values, marker="o", linewidth=2)
        plt.title("Daily Sales Trend")
        plt.xlabel("Date")
        plt.ylabel("Total Sales")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def sales_heatmap(self):
        numeric_data = self.data[["Price", "Quantity Sold", "Total Sales"]]
        correlation = numeric_data.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Retail Sales Correlation Heatmap")
        plt.tight_layout()
        plt.show()

def main():
    try:
        analyzer = RetailAnalyzer("retail_sales.csv")
        analyzer.load_data()
        analyzer.clean_data()
        analyzer.calculate_metrics()
        analyzer.display_summary()
        print("\nAvailable Categories:")
        print(analyzer.data["Category"].unique())
        analyzer.category_sales_chart()
        analyzer.sales_trend_chart()
        analyzer.sales_heatmap()
    except FileNotFoundError:
        print("Error: retail_sales.csv file not found.")
    except ValueError as error:
        print("Data Error:", error)

if __name__ == "__main__":
    main()