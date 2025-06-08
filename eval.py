import sys
import os
import json
from math import log
import numpy as np
from joblib import load
import xgboost as xgb # Included for model compatibility when loading

def main():
    """
    Evaluates the reimbursement calculation implementation against public
    historical cases and provides a detailed performance summary.
    """
    print("🧾 Black Box Challenge - Reimbursement System Evaluation")
    print("=======================================================")
    print()

    # --- 1. Pre-run Checks ---
    required_files = ['public_cases.json', 'xgb_model.joblib']
    for filename in required_files:
        if not os.path.exists(filename):
            print(f"❌ Error: Required file '{filename}' not found!")
            print(f"Please ensure '{filename}' is in the current directory.")
            sys.exit(1)

    # --- 2. Load Model and Test Cases ---
    print("🧠 Loading XGBoost model and public test cases...")
    try:
        model = load('xgb_model.joblib')
        with open('public_cases.json', 'r') as f:
            test_cases = json.load(f)
    except Exception as e:
        print(f"❌ Error during file loading: {e}")
        sys.exit(1)

    num_cases = len(test_cases)
    print(f"✅ Model and {num_cases} test cases loaded successfully.")
    print("\n📊 Running evaluation...")

    # --- 3. Initialize Metrics and Storage ---
    successful_runs = 0
    exact_matches = 0
    close_matches = 0
    total_error = 0.0
    results_data = []
    errors_list = []

    # --- 4. Process Each Test Case ---
    for i, case in enumerate(test_cases):
        if (i + 1) % 100 == 0:
            print(f"   -> Progress: {i + 1}/{num_cases} cases processed...")

        try:
            # Extract input data and expected output
            inputs = case['input']
            trip_duration = float(inputs['trip_duration_days'])
            miles_traveled = float(inputs['miles_traveled'])
            receipts_amount = float(inputs['total_receipts_amount'])
            expected_output = float(case['expected_output'])

            # --- Feature Engineering (from calculate_reimbursement.py) ---
            d_safe = trip_duration if trip_duration > 0 else 1
            r_safe = receipts_amount if receipts_amount > 0 else 1
            m_safe = miles_traveled if miles_traveled > 0 else 1

            feature_values = [
                trip_duration, miles_traveled, receipts_amount,
                miles_traveled / d_safe, receipts_amount / d_safe,
                receipts_amount ** 2, miles_traveled ** 2,
                trip_duration * miles_traveled, (trip_duration ** 2) * (miles_traveled ** 2),
                log(d_safe * r_safe), log(m_safe * r_safe),
                1 if round(receipts_amount % 1, 2) in [0.49, 0.99] else 0,
                1 if trip_duration >= 5 else 0,
                1 if trip_duration == 5 else 0,
                1 if trip_duration == 8 else 0,
                log(r_safe)
            ]
            input_array = np.array([feature_values])

            # --- Prediction ---
            prediction = model.predict(input_array)
            actual_output = float(prediction[0])

            # --- Evaluation ---
            error = abs(actual_output - expected_output)
            successful_runs += 1
            total_error += error

            if error < 0.01:
                exact_matches += 1
            if error < 1.00:
                close_matches += 1

            # Store detailed results for later analysis
            results_data.append({
                "case_num": i + 1, "expected": expected_output,
                "actual": actual_output, "error": error,
                "inputs": inputs
            })

        except Exception as e:
            errors_list.append(f"Case {i+1}: Script failed with error: {e}")

    print("   -> Evaluation Complete!")
    print()

    # --- 5. Calculate and Display Results ---
    if successful_runs == 0:
        print("❌ No successful test cases!")
        print("\nYour script either:\n - Failed to run properly\n - Produced invalid output format")
    else:
        avg_error = total_error / successful_runs
        exact_pct = (exact_matches / successful_runs) * 100
        close_pct = (close_matches / successful_runs) * 100
        score = avg_error * 100 + (num_cases - exact_matches) * 0.1

        print("📈 Results Summary:")
        print(f"  Total test cases: {num_cases}")
        print(f"  Successful runs:  {successful_runs}")
        print(f"  Exact matches (±$0.01): {exact_matches} ({exact_pct:.1f}%)")
        print(f"  Close matches (±$1.00): {close_matches} ({close_pct:.1f}%)")
        print(f"  Average error:    ${avg_error:.2f}")
        
        # Sort to find max error
        results_data.sort(key=lambda x: x['error'], reverse=True)
        max_error = results_data[0]['error'] if results_data else 0.0
        print(f"  Maximum error:    ${max_error:.2f}")
        print()
        print(f"🎯 Your Score: {score:.2f} (lower is better)")
        print()

        # Provide qualitative feedback
        if exact_matches == num_cases:
            print("🏆 PERFECT SCORE! You have reverse-engineered the system completely!")
        elif exact_matches > 950:
            print("🥇 Excellent! You are very close to the perfect solution.")
        elif exact_matches > 800:
            print("🥈 Great work! You have captured most of the system behavior.")
        elif exact_matches > 500:
            print("🥉 Good progress! You understand some key patterns.")
        else:
            print("📚 Keep analyzing the patterns in the interviews and test cases.")
        
        print()
        print("💡 Tips for improvement:")
        if exact_matches < num_cases:
            print("  Check these high-error cases:")
            for result in results_data[:5]:
                inputs = result['inputs']
                print(f"    Case {result['case_num']}: {inputs['trip_duration_days']} days, "
                      f"{inputs['miles_traveled']} miles, ${inputs['total_receipts_amount']} receipts")
                print(f"      Expected: ${result['expected']:.2f}, Got: ${result['actual']:.2f}, "
                      f"Error: ${result['error']:.2f}")

    # --- 6. Display Errors, if any ---
    if errors_list:
        print("\n⚠️  Errors encountered:")
        for i, err in enumerate(errors_list):
            if i >= 10:
                print(f"  ... and {len(errors_list) - 10} more errors")
                break
            print(f"  {err}")

    print("\n📝 Next steps:")
    print("  1. Fix any script errors shown above")
    print("  2. Analyze the patterns in the interviews and public cases")
    print("  3. Test edge cases around trip length and receipt amounts")
    print("  4. Once you are happy with your score, run generate_results.py")
    print("  5. Submit the generated private_results.txt file!")

if __name__ == "__main__":
    main()