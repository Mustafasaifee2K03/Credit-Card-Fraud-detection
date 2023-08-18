import joblib

# Load the trained model
model = joblib.load('model.pkl')

# Prepare input data
input_data = {
    'distance_from_home': 56.6,
    'distance_from_last_transaction': 23.2,
    'ratio_to_median_purchase_price': 85,
    'repeat_retailer': 1,
    'used_chip': 0,
    'used_pin': 1,
    'online_order': 1
}

# Make predictions using the model
prediction = model.predict([list(input_data.values())])[0]

# Display the prediction
print(f"Prediction: {prediction}")
