import sys
from joblib import load
from pandas import DataFrame
from math import log
import xgboost as xgb

def calculate_reimbursement():
    """
    Predicts a reimbursement amount based on trip details using a pre-trained XGBoost model.
    """
    # --- 1. Argument Parsing ---
    # Check if the correct number of command-line arguments are provided.
    if len(sys.argv) != 4:
        print("Usage: python calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)

    try:
        # Read and convert command-line arguments to floats.
        trip_duration_days = float(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
    except ValueError:
        print("Error: All inputs must be numeric.")
        sys.exit(1)

    # --- 2. Feature Engineering ---
    # Create the same features that the model was trained on.
    
    # Handle potential edge cases like division by zero or log(0) by replacing 0 with 1.
    d_safe = trip_duration_days if trip_duration_days > 0 else 1
    r_safe = total_receipts_amount if total_receipts_amount > 0 else 1
    m_safe = miles_traveled if miles_traveled > 0 else 1

    # Create a dictionary to build the feature set.
    features = {
        'trip_duration_days': trip_duration_days,
        'miles_traveled': miles_traveled,
        'total_receipts_amount': total_receipts_amount,
        'm/d': miles_traveled / d_safe,
        'r/d': total_receipts_amount / d_safe,
        'r^2': total_receipts_amount ** 2,
        'm^2': miles_traveled ** 2,
        'd*m': trip_duration_days * miles_traveled,
        'd^2*m^2': (trip_duration_days ** 2) * (miles_traveled ** 2),
        'log(d*r)': log(d_safe * r_safe),
        'log(m*r)': log(m_safe * r_safe),
        'log(r)': log(r_safe),
        'cents_bug_flag': 1 if round(total_receipts_amount % 1, 2) in [0.49, 0.99] else 0,
        'is_more_than_5_day_trip_flag': 1 if trip_duration_days >= 5 else 0,
        'is_5_day_trip_flag': 1 if trip_duration_days == 5 else 0,
        'is_8_day_trip_flag': 1 if trip_duration_days == 8 else 0
    }

    # Convert the features dictionary into a pandas DataFrame.
    input_df = DataFrame([features])

    # --- 3. Model Loading and Prediction ---
    try:
        # Load the pre-trained XGBoost model from the file.
        model = load('xgb_model.joblib')
    except FileNotFoundError:
        print("Error: Model file 'xgb_model.joblib' not found. Make sure it's in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        sys.exit(1)

    # Ensure the DataFrame columns are in the exact order the model expects.
    features_to_use = [
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'm/d',
        'r/d', 'r^2', 'm^2', 'd*m', 'd^2*m^2', 'log(d*r)', 'log(m*r)',
        'cents_bug_flag', 'is_more_than_5_day_trip_flag', 'is_5_day_trip_flag',
        'is_8_day_trip_flag', 'log(r)'
    ]
    input_df = input_df[features_to_use]

    # Use the model to make a prediction.
    prediction = model.predict(input_df)

    # --- 4. Output ---
    # Print the final prediction, formatted to two decimal places.
    # The prediction is returned as a single-item array, so we access the first element.
    print(f"{prediction[0]:.2f}")

if __name__ == "__main__":
    calculate_reimbursement()
