import numpy as np
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser="auto")
X, y = mnist.data, mnist.target.astype(int)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training model...")
clf = RandomForestClassifier(n_estimators=300, random_state=42)  # Increase trees
clf.fit(X_train, y_train)

# Save trained model
print("Saving model to 'model/digit_model.pkl'...")
joblib.dump(clf, "model/digit_model.pkl")

# Evaluate model
accuracy = clf.score(X_test, y_test)
print(f"Model Training Complete! Accuracy: {accuracy:.4f}")
