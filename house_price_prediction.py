import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the data
data = pd.read_csv('data.csv')
x = data[['Area (sq ft)', 'Bedrooms', 'Age']]
y = data['Price (in lakhs)']

# Train the model
model = LinearRegression()
model.fit(x, y)

# Save the model
joblib.dump(model, 'house_price_model.pkl')

# ---- Interactivity ----
print("🏠 House Price Prediction 🏠")
area = float(input("Enter area in sq ft: "))
bedrooms = int(input("Enter number of bedrooms: "))
age = int(input("Enter age of the house (in years): "))

# Predict
new_data = [[area, bedrooms, age]]
predicted_price = model.predict(new_data)

print(f"\n💰 Predicted Price: ₹{predicted_price[0]:.2f} lakhs")
