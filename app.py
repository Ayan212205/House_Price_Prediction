import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Load the dataset
df = pd.read_csv("train.csv")

# Display first 5 rows
# print(df.head())

# print("\nDataset Shape:")
# print(df.shape)

# print("\nColumn Names:")
# print(df.columns)

# print("\nDataset Information:")
# print(df.info())

# print("\nMissing Values:")
# print(df.isnull().sum())



# Select required columns
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

print("\nSelected Features:")
print(df.head())

print("\nMissing Values in Selected Features:")
print(df.isnull().sum())

# Input features
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Target
y = df['SalePrice']

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# Predict house prices
y_pred = model.predict(X_test)

print("\nFirst 10 Predicted Prices:")
print(y_pred[:10])


# Compare actual and predicted prices
results = pd.DataFrame({
    'Actual Price': y_test.values,
    'Predicted Price': y_pred
})

print("\nComparison of Actual vs Predicted Prices:")
print(results.head(10))

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# Display model coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nModel Coefficients:")
print(coefficients)

print("\nIntercept:")
print(model.intercept_)


plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

# Ideal prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linewidth=2
)

plt.show()


new_house = pd.DataFrame({
    'GrLivArea': [2000],
    'BedroomAbvGr': [3],
    'FullBath': [2]
})

predicted_price = model.predict(new_house)

print(f"\nPredicted Price of the New House: ${predicted_price[0]:,.2f}")



import joblib

joblib.dump(model, "house_price_model.pkl")

print("\nModel saved successfully as house_price_model.pkl")


# Actual vs Predicted Scatter Plot

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, color='blue', alpha=0.6)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linewidth=2,
    label="Perfect Prediction"
)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

plt.legend()

plt.grid(True)

plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches="tight")

plt.show()


plt.figure(figsize=(6,5))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")

plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")

plt.show()


results.to_csv("prediction_results.csv", index=False)

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

plt.figure(figsize=(7,4))

sns.barplot(
    data=coefficients,
    x="Feature",
    y="Coefficient"
)

plt.title("Feature Coefficients")

plt.xticks(rotation=20)

plt.grid(axis='y')

plt.savefig("feature_coefficients.png", dpi=300, bbox_inches="tight")

plt.show()


plt.figure(figsize=(8,5))

sns.histplot(df["SalePrice"], bins=30, kde=True)

plt.title("Distribution of House Prices")

plt.xlabel("Sale Price")

plt.savefig("house_price_distribution.png", dpi=300, bbox_inches="tight")

plt.show()