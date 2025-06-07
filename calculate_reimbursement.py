import sys
import numpy as np
import pandas as pd
from io import StringIO

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    # Read command-line arguments
    trip_duration_days_arg = float(sys.argv[1])
    miles_traveled_arg = float(sys.argv[2])
    total_receipts_amount_arg = float(sys.argv[3])
   
    # Load the coefficients from the specified CSV file
    c = {
        "trip_duration_days":11.282605648742441,
        "miles_traveled":0.5532652986584919,
        "total_receipts_amount":1.4549554491320587,
        "m/d":-0.1401926055128268,
        "r/d":0.13848780244124023,
        "r^2":-0.00038661595566191753,
        "m^2":-9.721609133567233e-05,
        "d*m":0.0016065907521636644,
        "log(d*r)":220.4648136137227,
        "cents_bug_flag":-456.3037468071681,
        "log(r)":-339.48745881129236,
        "intercept":414.1590572002822
    }
    

    # Calculate the final output
    output = c['intercept'] + c['trip_duration_days'] * trip_duration_days_arg + c['miles_traveled'] * miles_traveled_arg + c['total_receipts_amount'] * total_receipts_amount_arg + c['m/d'] * (miles_traveled_arg / trip_duration_days_arg) + c['r/d'] * (total_receipts_amount_arg / trip_duration_days_arg) + c['r^2'] * (total_receipts_amount_arg ** 2) + c['m^2'] * (miles_traveled_arg ** 2) + c['d*m'] * (trip_duration_days_arg * miles_traveled_arg) + c['log(d*r)'] * np.log(trip_duration_days_arg * total_receipts_amount_arg) + c['cents_bug_flag'] * (1 if round(total_receipts_amount_arg % 1, 2) in [0.49, 0.99] else 0) + c['log(r)'] * np.log(total_receipts_amount_arg)
    
    # Print the result rounded to 2 decimal places
    print(f"{output:.2f}")

