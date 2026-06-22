import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# File path
file_path = "housing_transactions.csv"

# Read large dataset in chunks
chunks = []
chunk_size = 100000

for chunk in pd.read_csv(file_path, chunksize=chunk_size):
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)

# ---------------------------
# Data Cleaning
# ---------------------------

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# ---------------------------
# EDA
# ---------------------------

# Plot house price distribution
plt.figure(figsize=(10, 5))
sns.histplot(df["Price"], kde=True)
plt.title("Distribution of House Prices")
plt.show()

# ---------------------------
# Outlier Removal
# ---------------------------

mean_price = df["Price"].mean()
std_price = df["Price"].std()

lower_limit = mean_price - 3 * std_price
upper_limit = mean_price + 3 * std_price

df = df[
    (df["Price"] >= lower_limit) &
    (df["Price"] <= upper_limit)
]

# ---------------------------
# Feature Engineering
# ---------------------------

# Age of Home
current_year = pd.Timestamp.now().year
df["Age_of_Home"] = current_year - df["Year_Built"]

# Distance to City Center
df["Distance_to_City_Center"] = np.sqrt(
    (df["Latitude"] - df["City_Center_Lat"])**2 +
    (df["Longitude"] - df["City_Center_Lon"])**2
)

# Price per Square Foot
df["Price_per_SqFt"] = df["Price"] / df["Square_Feet"]

# ---------------------------
# Save as Parquet
# ---------------------------

df.to_parquet(
    "cleaned_housing_data.parquet",
    index=False,
    engine="pyarrow"
)

print("Cleaned data saved successfully as cleaned_housing_data.parquet")