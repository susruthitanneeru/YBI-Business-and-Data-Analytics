# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('employee_performance.csv')

# Data Preprocessing

# Handle missing numeric values
df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)

# Handle missing categorical values
df.fillna('Unknown', inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
categorical_cols = ['Department', 'JobRole', 'EducationField']

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Define features and target variable
X = df.drop('PerformanceRating', axis=1)
y = df['PerformanceRating']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Plotting Feature Importance
plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importances, x='Feature', y='Importance', palette="viridis")
plt.title('Feature Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
