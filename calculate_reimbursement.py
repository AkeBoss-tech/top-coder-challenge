import sys
import joblib
import numpy as np

def calculate_tiered_mileage(miles):
    """Calculate tiered mileage feature based on the rates from your training code"""
    if miles > 100:
        return (100 * 0.58) + ((miles - 100) * 0.45)
    else:
        return miles * 0.58

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    trip_duration_days = float(sys.argv[1])
    miles_traveled = float(sys.argv[2])
    total_receipts_amount = float(sys.argv[3])
    
    # Load the XGBoost model
    xgb_model = joblib.load('xgb_model.joblib')
    
    # Calculate derived features (same as in your training code)
    # Avoid division by zero
    safe_duration = trip_duration_days if trip_duration_days > 0 else 1
    miles_per_day = miles_traveled / safe_duration
    receipts_per_day = total_receipts_amount / safe_duration
    
    # Calculate tiered mileage feature
    tiered_mileage_feature = calculate_tiered_mileage(miles_traveled)
    
    # Calculate boolean flags (same logic as training)
    efficiency_bonus_flag = int(180 <= miles_per_day <= 220)
    is_5_day_trip_flag = int(trip_duration_days == 5)
    is_sweet_spot_duration_flag = int(4 <= trip_duration_days <= 6)
    low_receipt_penalty_flag = int(0 < total_receipts_amount < 50)
    
    # Cents bug flag - check if cents part is 0.49 or 0.99
    cents_part = round(total_receipts_amount % 1, 2)
    cents_bug_flag = int(cents_part in [0.49, 0.99])
    
    # Sweet spot combo flag
    sweet_spot_combo_flag = int(
        trip_duration_days == 5 and 
        miles_per_day >= 180 and 
        receipts_per_day < 100
    )
    
    # Vacation penalty flag
    vacation_penalty_flag = int(
        trip_duration_days >= 8 and 
        receipts_per_day > 90
    )
    
    # Create feature array in the correct order (same as features_to_use in training)
    features = np.array([[
        trip_duration_days,
        miles_traveled,
        total_receipts_amount,
        miles_per_day,
        receipts_per_day,
        tiered_mileage_feature,
        efficiency_bonus_flag,
        is_5_day_trip_flag,
        is_sweet_spot_duration_flag,
        low_receipt_penalty_flag,
        cents_bug_flag,
        sweet_spot_combo_flag,
        vacation_penalty_flag
    ]])
    
    # Predict the output
    output = xgb_model.predict(features)
    print(f"{output[0]:.2f}")