import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib

# Load dataset
file_path = "E:/PROJECTDATA/ecommerce_customer_behavior.csv"  # Update this path if needed
df = pd.read_csv(file_path)

# Display dataset info
print("Dataset Loaded Successfully!")
print(df.head())  # Display first few rows
print("\nColumn Names:\n", df.columns)

# Handling missing values
df.dropna(inplace=True)  # Drop rows with missing values

# Extract numeric values from 'customerID'
if 'customerID' in df.columns:
    df['customerID'] = df['customerID'].astype(str).str.extract(r'(\d+)')  # Extract numeric part
    df['customerID'] = pd.to_numeric(df['customerID'], errors='coerce')  # Convert to integer

# Identify categorical columns that need encoding
categorical_cols = ['gender', 'device_type', 'category_visited', 'discount_used', 
                    'return_requested', 'PaymentMethod','chrun_lable']

for col in categorical_cols:
    if col in df.columns:  # Encode only existing columns
        df[col] = LabelEncoder().fit_transform(df[col])

# Convert numeric columns if needed
if 'TotalCharge' in df.columns:
   df['TotalCharge'] = pd.to_numeric(df['TotalCharge'], errors='coerce')
   df.fillna(df.median(), inplace=True)  # Fill missing numeric values

# Define target column for churn
target_col = 'churn_label' if 'churn_label' in df.columns else 'Churn'
y = df[target_col]
X = df.drop(columns=[target_col])  # Drop target column

# Check for non-numeric columns before scaling
print("\nNon-numeric columns before scaling:")
print(X.select_dtypes(include=['object']).columns)

# Convert all remaining object columns to numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()    #intilaizing scaler object
X_train = scaler.fit_transform(X_train) #fit comupute the Mean and Standrad Deviation of feature column and tranform compute scale
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)    #initiate 100 decision tree
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix Visualization
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='orange')
plt.title("Confusion Matrix")
plt.show()

# Save Model
joblib.dump(model, 'E:/PROJECTDATA/customer_churn_model.pkl')
print("\nModel Saved as 'customer_churn_model.pkl'")

# Load the .pkl file (assuming it's a dataset like a DataFrame)
data = joblib.load('E:/PROJECTDATA/customer_churn_model.pkl')  

# Visualize the dataset (example: scatter plot)
plt.scatter(data['feature1'], data['feature2'])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Feature 1 vs Feature 2')

# Save the plot as a PNG file
plt.savefig('visualized_data.png')
plt.close()