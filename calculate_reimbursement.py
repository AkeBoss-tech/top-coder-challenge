import sys
import numpy as np
import pandas as pd

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount, coefficients_df):
    """
    Calculates the reimbursement amount using a pre-trained linear model.

    Args:
        trip_duration_days (float): Duration of the trip in days.
        miles_traveled (float): Total miles traveled.
        total_receipts_amount (float): Total amount from receipts.
        coefficients_df (pd.DataFrame): DataFrame containing model coefficients.

    Returns:
        float: The predicted reimbursement amount.
    """
    
    # --- FIX 1: Separate the intercept from the other feature coefficients ---
    # The intercept is a special value added at the end, not multiplied by any feature.
    try:
        intercept = coefficients_df[coefficients_df['feature'] == 'intercept']['coefficient'].iloc[0]
        feature_coeffs = coefficients_df[coefficients_df['feature'] != 'intercept']
    except IndexError:
        print("Error: 'intercept' not found in coefficients file.")
        sys.exit(1)

    # Ensure the order of coefficients matches the order we build our feature array
    # This is crucial for the dot product to be correct.
    ordered_coeffs = feature_coeffs.set_index('feature').reindex([
        'trip_duration_days', 'miles_traveled', 'total_receipts_amount', 'm/d', 'r/d',
        'r^2', 'm^2', 'd*m', 'log(d*r)', 'cents_bug_flag', 'log(r)'
    ])['coefficient'].values


    # --- FIX 2: Handle potential log(0) errors by adding a small epsilon or using np.log1p ---
    # We use np.log1p(x) which calculates log(1+x), safely handling x=0.
    # For products, we ensure the input to log is never zero.
    log_dr_input = trip_duration_days * total_receipts_amount
    log_r_input = total_receipts_amount
    
    # Create the feature array for the single prediction
    features = np.array([
        trip_duration_days,
        miles_traveled,
        total_receipts_amount,
        miles_traveled / trip_duration_days if trip_duration_days > 0 else 0,
        total_receipts_amount / trip_duration_days if trip_duration_days > 0 else 0,
        total_receipts_amount ** 2,
        miles_traveled ** 2,
        trip_duration_days * miles_traveled,
        np.log(log_dr_input) if log_dr_input > 0 else 0,
        1 if round(total_receipts_amount % 1, 2) in [0.49, 0.99] else 0,
        np.log(log_r_input) if log_r_input > 0 else 0,
    ])

    # --- FIX 3: Perform the correct linear algebra calculation ---
    # The prediction is the dot product of features and their coefficients, plus the intercept.
    # The shapes will now be correct: (1x11) dot (11x1) = a single value.
    prediction = np.dot(features, ordered_coeffs) + intercept
    
    return prediction


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    # Read command-line arguments
    trip_duration_days_arg = float(sys.argv[1])
    miles_traveled_arg = float(sys.argv[2])
    total_receipts_amount_arg = float(sys.argv[3])

    try:
        # Load the coefficients from the specified CSV file
        coefficients_df_main = pd.read_csv('linear_regression_coefficients.csv')
    except FileNotFoundError:
        print("Error: 'linear_regression_coefficients.csv' not found.")
        print("Please ensure the CSV file with coefficients is in the same directory.")
        sys.exit(1)

    # Calculate the final output
    output = calculate_reimbursement(
        trip_duration_days_arg,
        miles_traveled_arg,
        total_receipts_amount_arg,
        coefficients_df_main
    )
    
    # Print the result rounded to 2 decimal places
    print(f"{output:.2f}")

