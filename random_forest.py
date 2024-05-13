import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif

# Load data
# data = pd.read_csv('training_raw_300mb.csv')
data = pd.read_csv('covid-19-lung-dataset.csv')

# Prepare the feature matrix and target vector
# X = data.drop(['NAME', 'disease_ontology_label', 'group'], axis=1)
data['disease_ontology_label'] = (data['disease_ontology_label'] == 'COVID-19').astype(int)
X = data.drop(['NAME', 'disease_ontology_label', 'group'], axis=1)
y = data['disease_ontology_label']

# Display the number of features before feature selection
print(f"Total number of features before selection: {X.shape[1]}")

# # Downsampling to balance classes
# g = data.groupby('disease_ontology_label')
# data_balanced = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

# # Update X and y after downsampling
# X = data_balanced.drop(['NAME', 'disease_ontology_label', 'group'], axis=1)
# # X = data_balanced.drop(['NAME', 'disease_ontology_label'], axis=1)
# y = data_balanced['disease_ontology_label']

# Display new class sizes
print("Class sizes after balancing:")
print(y.value_counts())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature selection using SelectKBest with f_classif
selector = SelectKBest(score_func=f_classif, k=2000)  # Select top 100 features
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# Display the number of features after feature selection
print(f"Total number of features after selection: {X_train.shape[1]}")

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("--------------------------------------------------")
print("Random Forest Classifier Results")
print("--------------------------------------------------")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

print(classification_report(y_test, y_pred))
