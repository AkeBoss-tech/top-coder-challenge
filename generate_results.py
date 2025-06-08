import sys
import os
import json
from math import log
import numpy as np
from joblib import load
import xgboost as xgb # Included for model compatibility when loading

# --- Main Script Logic ---

def main():
    """
    Runs the Black Box Challenge implementation against test cases and
    outputs results to private_results.txt.
    """
    print("🧾 Black Box Challenge - Generating Private Results")
    print("====================================================")
    print()

    # --- 1. Pre-run Checks ---
    # Check if the necessary files exist before starting.
    required_files = ['private_cases.json', 'xgb_model.joblib']
    for filename in required_files:
        if not os.path.exists(filename):
            print(f"❌ Error: Required file '{filename}' not found!")
            print(f"Please ensure '{filename}' is in the current directory.")
            sys.exit(1)

    # --- 2. Load Model and Test Cases ---
    print("🧠 Loading XGBoost model and test cases...")
    try:
        model = load('xgb_model.joblib')
        with open('private_cases.json', 'r') as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"❌ Error during file loading: {e}")
        sys.exit(1)

    total_cases = len(test_cases)
    print(f"✅ Model and {total_cases} test cases loaded successfully.")
    print("\n📊 Processing test cases and generating results...")
    print("📝 Output will be saved to private_results.txt")
    print()

    # --- 3. Process Cases and Generate Results ---
    # Open the results file once and write line by line.
    with open('private_results.txt', 'w') as results_file:
        for i, case in enumerate(test_cases):
            # Display progress for large datasets
            if (i + 1) % 100 == 0:
                print(f"   -> Progress: {i + 1}/{total_cases} cases processed...")

            try:
                # Extract data from the test case
                trip_duration = float(case['trip_duration_days'])
                miles_traveled = float(case['miles_traveled'])
                receipts_amount = float(case['total_receipts_amount'])

                # --- Feature Engineering (from calculate_reimbursement.py) ---
                d_safe = trip_duration if trip_duration > 0 else 1
                r_safe = receipts_amount if receipts_amount > 0 else 1
                m_safe = miles_traveled if miles_traveled > 0 else 1

                feature_values = [
                    trip_duration,
                    miles_traveled,
                    receipts_amount,
                    miles_traveled / d_safe,
                    receipts_amount / d_safe,
                    receipts_amount ** 2,
                    miles_traveled ** 2,
                    trip_duration * miles_traveled,
                    (trip_duration ** 2) * (miles_traveled ** 2),
                    log(d_safe * r_safe),
                    log(m_safe * r_safe),
                    1 if round(receipts_amount % 1, 2) in [0.49, 0.99] else 0,
                    1 if trip_duration >= 5 else 0,
                    1 if trip_duration == 5 else 0,
                    1 if trip_duration == 8 else 0,
                    log(r_safe)
                ]

                input_array = np.array([feature_values])

                # --- Prediction ---
                prediction = model.predict(input_array)
                result = f"{prediction[0]:.2f}"
                results_file.write(result + '\n')

            except (ValueError, TypeError) as e:
                # Handle cases with non-numeric data or calculation errors
                print(f"Error on case {i+1}: Invalid data encountered. {e}", file=sys.stderr)
                results_file.write("ERROR\n")
            except Exception as e:
                # Handle any other unexpected errors during prediction
                print(f"Error on case {i+1}: Script failed. {e}", file=sys.stderr)
                results_file.write("ERROR\n")

    print("\n✅ Results generated successfully!")
    print("📄 Output saved to private_results.txt")
    print("📊 Each line contains the result for the corresponding test case in private_cases.json")

    print("\n🎯 Next steps:")
    print("   1. Check private_results.txt - it should contain one result per line.")
    print("   2. Each line corresponds to the same-numbered test case in private_cases.json.")
    print("   3. Lines with 'ERROR' indicate cases where the script failed.")
    print("   4. Submit your private_results.txt file when ready!")
    print()
    print("📈 File format:")
    print("   Line 1: Result for private_cases.json[0]")
    print("   Line 2: Result for private_cases.json[1]")
    print("   Line 3: Result for private_cases.json[2]")
    print("   ...")
    print(f"   Line {total_cases}: Result for private_cases.json[{total_cases-1}]")

if __name__ == "__main__":
    main()