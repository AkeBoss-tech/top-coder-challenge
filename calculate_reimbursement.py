import sys
from joblib import load
import numpy as np
from math import log
import xgboost as xgb

def calculate_reimbursement():
    """
    Predicts a reimbursement amount based on trip details using a pre-trained XGBoost model.
    """
    # --- 1. Argument Parsing ---
    if len(sys.argv) != 4:
        print("Usage: python calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)

    try:
        trip_duration_days = float(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
    except ValueError:
        print("Error: All inputs must be numeric.")
        sys.exit(1)

    # --- 2. Feature Engineering ---
    # Handle potential edge cases like division by zero or log(0) by replacing 0 with 1.
    d_safe = trip_duration_days if trip_duration_days > 0 else 1
    r_safe = total_receipts_amount if total_receipts_amount > 0 else 1
    m_safe = miles_traveled if miles_traveled > 0 else 1

    # Create features directly in a list in the correct order for the model.
    # This avoids the overhead of creating a pandas DataFrame.
    feature_values = [
        trip_duration_days,
        miles_traveled,
        total_receipts_amount,
        miles_traveled / d_safe,  # m/d
        total_receipts_amount / d_safe,  # r/d
        total_receipts_amount ** 2,  # r^2
        miles_traveled ** 2,  # m^2
        trip_duration_days * miles_traveled,  # d*m
        (trip_duration_days ** 2) * (miles_traveled ** 2),  # d^2*m^2
        log(d_safe * r_safe),  # log(d*r)
        log(m_safe * r_safe),  # log(m*r)
        1 if round(total_receipts_amount % 1, 2) in [0.49, 0.99] else 0,  # cents_bug_flag
        1 if trip_duration_days >= 5 else 0,  # is_more_than_5_day_trip_flag
        1 if trip_duration_days == 5 else 0,  # is_5_day_trip_flag
        1 if trip_duration_days == 8 else 0,  # is_8_day_trip_flag
        log(r_safe)  # log(r)
    ]

    # Convert the list to a 2D NumPy array, as the model expects a 2D input.
    input_array = np.array([feature_values])

    # --- 3. Model Loading and Prediction ---
    try:
        model = load('xgb_model.joblib')
    except FileNotFoundError:
        print("Error: Model file 'xgb_model.joblib' not found. Make sure it's in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        sys.exit(1)

    # Use the model to make a prediction on the NumPy array.
    prediction = model.predict(input_array)

    # --- 4. Output ---
    # Print the prediction, formatted to two decimal places.
    print(f"{prediction[0]:.2f}")

if __name__ == "__main__":
    calculate_reimbursement()
