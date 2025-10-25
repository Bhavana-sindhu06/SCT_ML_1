import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Define feature constants based on the assumed actual column names
FEATURE_SQ_FT = 'GrLivArea'      # Example for Square Footage
FEATURE_BEDROOMS = 'BedroomAbvGr' # Example for Bedrooms
FEATURE_BATHROOMS = 'FullBath'    # Example for Bathrooms
TARGET_PRICE = 'SalePrice'       # Target variable

# --- 1. Load Data ---
# --------------------------------------------------------------------------
try:
    # Set the correct path to your CSV file
    file_path = r'C:\Users\Sindhu\Downloads\house-prices-advanced-regression-techniques\train.csv'
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully from: {file_path}")

    # Use the ACTUAL column names for feature selection
    X = df[[FEATURE_SQ_FT, FEATURE_BEDROOMS, FEATURE_BATHROOMS]]
    y = df[TARGET_PRICE] 
    
    # Quick Data Cleaning: Drop rows with missing values for features/target
    data_points_before = df.shape[0]
    df.dropna(subset=[FEATURE_SQ_FT, FEATURE_BEDROOMS, FEATURE_BATHROOMS, TARGET_PRICE], inplace=True)
    
    # Re-extract X and y after dropping NaNs to ensure they are synchronized
    X = df[[FEATURE_SQ_FT, FEATURE_BEDROOMS, FEATURE_BATHROOMS]]
    y = df[TARGET_PRICE]
    data_points_after = df.shape[0]
    print(f"Data Cleaning: Dropped {data_points_before - data_points_after} rows with missing values.")

except FileNotFoundError:
    print("Error: train.csv not found. Using sample data for demonstration.")
    
    # Fallback Logic: Synthetic data
    data = {
        FEATURE_SQ_FT: [1500, 2000, 1200, 2500, 1800, 3000, 1400, 2200],
        FEATURE_BEDROOMS: [3, 4, 2, 4, 3, 5, 2, 4],
        FEATURE_BATHROOMS: [2, 3, 1, 3, 2, 3, 1.5, 2.5],
        TARGET_PRICE: [300000, 450000, 200000, 550000, 380000, 650000, 250000, 500000]
    }
    df = pd.DataFrame(data)
    X = df[[FEATURE_SQ_FT, FEATURE_BEDROOMS, FEATURE_BATHROOMS]]
    y = df[TARGET_PRICE]
# --------------------------------------------------------------------------

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 3. Make Predictions
y_pred = model.predict(X_test)

# 4. Evaluate the Model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# 5. Display Results and Interpretation
print("\n--- Linear Regression Model Summary ---")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: ${rmse:.2f}")

# --------------------------------------------------------------------------
# 6. ADDED: GRAPHS FOR VISUAL EVALUATION
# --------------------------------------------------------------------------

# Calculate residuals (errors)
residuals = y_test - y_pred

plt.figure(figsize=(14, 5))

# --- Graph 1: Actual vs. Predicted Plot ---
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
# Plot the ideal 45-degree line (where Actual = Predicted)
max_val = max(y_test.max(), y_pred.max())
min_val = min(y_test.min(), y_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2)

plt.title('Actual vs. Predicted House Prices')
plt.xlabel('Actual Prices (Test Data)')
plt.ylabel('Predicted Prices')
plt.grid(True, linestyle='--', alpha=0.6)

# --- Graph 2: Residual Plot ---
plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, color='green', alpha=0.6)
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), color='red', linestyle='-', linewidth=2)

plt.title('Residual Plot (Errors)')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals (Actual - Predicted)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------
# 7. Example Prediction (same as before)
# --------------------------------------------------------------------------
print("\n--- Example Prediction ---")
# Predict the price for a house: 2100 sq ft, 4 bedrooms, 2 bathrooms
new_house_sq_ft = 2100
new_house_bedrooms = 4
new_house_bathrooms = 2

new_house = pd.DataFrame(
    [[new_house_sq_ft, new_house_bedrooms, new_house_bathrooms]], 
    columns=[FEATURE_SQ_FT, FEATURE_BEDROOMS, FEATURE_BATHROOMS]
)
predicted_price = model.predict(new_house)[0]

print(f"House features: {new_house_sq_ft} sq ft, {new_house_bedrooms} bedrooms, {new_house_bathrooms} bathrooms")
print(f"Predicted Price: ${predicted_price:.2f}")
