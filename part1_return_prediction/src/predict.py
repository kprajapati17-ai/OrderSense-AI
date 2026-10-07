import pandas as pd
import joblib

model_data = joblib.load( "part1_return_prediction/models/return_prediction_model.pkl" )

model = model_data["model"]
threshold = model_data["threshold"]

print("\n--- Saved Model Loaded ---") 
print("Threshold:", threshold)

new_order = pd.DataFrame([ { "product_category": "Footwear", "price_inr": 2500, "discount_pct": 30, "payment_method": "UPI", "customer_tenure_days": 300, "num_previous_orders": 10, "num_previous_returns": 4, "delivery_distance_km": 15, "delivery_days": 7, "is_weekend_order": 1 } ])

new_order["previous_return_rate"] = ( new_order["num_previous_returns"] / new_order["num_previous_orders"].replace(0, 1) )

probability = model.predict_proba(new_order)[0][1]

prediction = int(probability >= threshold)

# 6. Display Result
print("\n--- Prediction Result ---") 
print("Return Probability:", round(probability * 100, 2), "%") 

if prediction == 1: 
    print("Prediction: Returned") 
else: 
    print("Prediction: Not Returned")

    