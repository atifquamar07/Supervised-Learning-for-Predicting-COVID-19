print("started")
import pandas as pd

print("started")
# Load the datasets
cell_markers = pd.read_csv('Cell_Markers_Formatted.csv')
print("loaded cell_markers")
scores_unipath = pd.read_csv('scores_unipath.csv', index_col=0)
print("loaded scores_unipath")
covid_data = pd.read_csv('covid-19-lung-dataset.csv')
print("loaded covid-19-lung-dataset.csv")

# Create a dictionary to map markers to cell names
marker_to_cell = {}
for _, row in cell_markers.iterrows():
    cell_name = row['cell_name']
    for marker in row.index[1:]:  # Skip the first column which is cell_name
        gene_name = row[marker]
        marker_to_cell[gene_name] = cell_name
    print(f"Cell name complted : {cell_name}")

# Function to retrieve scores based on the cell name
def get_scores(cell_name):
    if cell_name in scores_unipath.index:
        return scores_unipath.loc[cell_name]
    return pd.Series([None] * len(scores_unipath.columns), index=scores_unipath.columns)

# Append scores to covid_data for each gene column
for gene in covid_data.columns[1:]:  # Skip the first column which is ID in covid_data
    if gene in marker_to_cell:
        cell_name = marker_to_cell[gene]
        scores_series = get_scores(cell_name)
        covid_data = covid_data.join(scores_series.rename(f'score_{gene}'))
    print("Gene completed: ", gene)

# Save the updated dataset
covid_data.to_csv('updated_covid_19_lung_dataset.csv', index=False)
print("Data has been updated and saved.")
