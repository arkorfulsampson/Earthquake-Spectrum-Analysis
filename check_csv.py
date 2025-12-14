import pandas as pd

# Read the CSV file
file_path = r"C:\Users\arkor\Desktop\Learning Python\Earthquake Spectrum Analysis\RSN1.csv"
df = pd.read_csv(file_path)

# Display information about the file
print("CSV File Information:")
print("-" * 50)
print("Columns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())