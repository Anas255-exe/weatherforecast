import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Step 1: Load the dataset
# Replace 'weather_data.csv' with the path to your dataset
df = pd.read_csv('D:\coding\java\weatherforcaste\Dataset11-Weather-Data.csv')

# Step 2: Preprocess the data
# Convert 'Date/Time' to datetime (optional, as it may not be a useful feature here)
df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# Drop 'Date/Time' as it is not relevant for prediction
df = df.drop(columns=['Date/Time'])

# Encode the target variable ('Weather') as integers
label_encoder = LabelEncoder()
df['Weather'] = label_encoder.fit_transform(df['Weather'])

# Separate features (X) and target (y)
X = df.drop(columns=['Weather'])
y = df['Weather']

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate classification report with target names matching unique test classes
unique_test_classes = sorted(y_test.unique())  # Ensure the unique classes are sorted
target_names = label_encoder.inverse_transform(unique_test_classes)
print("Classification Report:")
print(classification_report(y_test, y_pred, labels=unique_test_classes, target_names=target_names))

# Step 6: Save the model
with open('weather_model.pkl', 'wb') as file:
    pickle.dump((model, label_encoder), file)

print("Model saved as 'weather_model.pkl'")
