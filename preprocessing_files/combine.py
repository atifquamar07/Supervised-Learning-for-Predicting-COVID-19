import pandas as pd

# Load the two CSV files into pandas DataFrames
df1 = pd.read_csv('covid-19-lung-dataset.csv')
df2 = pd.read_csv('training_raw_300mb.csv')

# Concatenate the DataFrames row-wise
combined_df = pd.concat([df1, df2], ignore_index=True)

# Drop duplicate rows, if any
combined_df = combined_df.drop_duplicates()

# Optionally, save the combined DataFrame to a new CSV file
combined_df.to_csv('huge_training.csv', index=False)

print("Combined DataFrame with duplicates removed:")
print(combined_df)
