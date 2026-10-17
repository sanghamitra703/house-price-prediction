# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

print("--- Step 1: Loading the Dataset ---")
# Load California Housing dataset (predicts median house values)
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Let's take a small sample to keep things fast and simple
df = df.sample(n=5000, random_state=42)

# Define features (X) and target (y)
X = df[['MedInc', 'AveRooms', 'HouseAge']]  # Median Income, Average Rooms, House Age
y = df['MedHouseVal']                      # House Value in $100k's

print(f"Dataset shape: {df.shape}")
print(df.head())

print("\n--- Step 2: Splitting Data into Training and Testing Sets ---")
# 80% for training the model, 20% for testing its accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

print("\n--- Step 3: Training the Machine Learning Model ---")
# Initialize the Linear Regression model
model = LinearRegression()

# Train (fit) the model using the training data
model.fit(X_train, y_train)
print("Model training complete!")

print("\n--- Step 4: Evaluating the Model ---")
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score (Accuracy measure): {r2:.4f}")

print("\n--- Step 5: Making a Prediction on New Data ---")
# Let's test with a fictional house: [Median Income, Average Rooms, House Age]
sample_house = [[5.5, 6.0, 25]] 
predicted_price = model.predict(sample_house)
print(f"Predicted House Value: ${predicted_price[0] * 100000:,.2f}")