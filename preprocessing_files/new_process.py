import pandas as pd

# Load the datasets
transposed_df = pd.read_csv('Transposed_covid-19-lung-dataset.csv')
scores_df = pd.read_csv('scoresXgenes.csv')

# Create a list to store the scores for each NAME
scores_list = []

# Iterate through each NAME in the Transposed dataset
for name in transposed_df['NAME']:
    # Search for the NAME in the scores dataset
    scores_row = scores_df[scores_df['cell_name'] == name]
    
    if not scores_row.empty:
        # If NAME is found, append the scores to the list
        scores_list.append(scores_row.iloc[:, 1:].values.flatten())
        print(f"{name} found")
    else:
        # If NAME is not found, append zeros
        scores_list.append([0] * 1163)
        print(f"{name} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

# Convert the scores_list to a DataFrame
scores_df = pd.DataFrame(scores_list, columns=['adjpva.V{}'.format(i) for i in range(1, 1164)])

# Concatenate the scores DataFrame with the Transposed dataset
combined_df = pd.concat([transposed_df, scores_df], axis=1)

# Save the modified dataset
combined_df.to_csv('Modified_Transposed_covid-19-lung-dataset.csv', index=False)
