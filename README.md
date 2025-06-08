# Top Coder Challenge: Black Box Legacy Reimbursement System

Hi! This was a really fun challenge! I was pretty busy today but I'm very happy with what I was able to accomplish in such a short time. Honestly, my code is pretty messy mainly because I was trying to prototype so quickly with Gemini. You may notice that I have extra python scripts and a GitHub action here, this is because I was having trouble running the scripts on my machine. 

The main way I went about this challenge was to find important features and fit some kind of model to them. After using Gemini to go through the interviews, I made a list of 6 potential variables which I then used in my initial models. Concurrently, I also graphed the data separately and in three dimensions to potentially find patterns. (this isn't in my current code but was previously in `analysis.ipynb`) I found that adding some quadratic terms would help with the results. Then I realized that I should add ratios in accordance to the interview responses. Then I went on a bit of a tangent writing a script to find products of the three terms from -2 to 2 to find correlation with the output, which also gave me a couple more variables. (this isn't in my current code but was previously in `analysis.ipynb`) Then, looking at the patterns in the errors for my model, I realized I needed to add some log terms as well. I knew I needed to use some kind of decision tree for this task since I didn't want to have to hard code tiers for example, so naturally XGBoost was my perferred algorithm and gave me a low MSE.

While this code is a bit messy from my rapid prototyping, it all worked out in the end. If I had even 5 more minutes working on this I'd seperate my analysis into two different notebooks and add comments and docs.

**Reverse-engineer a 60-year-old travel reimbursement system using only historical data and employee interviews.**

ACME Corp's legacy reimbursement system has been running for 60 years. No one knows how it works, but it's still used daily.

8090 has built them a new system, but ACME Corp is confused by the differences in results. Your mission is to figure out the original business logic so we can explain why ours is different and better.

Your job: create a perfect replica of the legacy system by reverse-engineering its behavior from 1,000 historical input/output examples and employee interviews.

## What You Have

### Input Parameters

The system takes three inputs:

- `trip_duration_days` - Number of days spent traveling (integer)
- `miles_traveled` - Total miles traveled (integer)
- `total_receipts_amount` - Total dollar amount of receipts (float)

## Documentation

- A PRD (Product Requirements Document)
- Employee interviews with system hints

### Output

- Single numeric reimbursement amount (float, rounded to 2 decimal places)

### Historical Data

- `public_cases.json` - 1,000 historical input/output examples

## Getting Started

1. **Analyze the data**: 
   - Look at `public_cases.json` to understand patterns
   - Look at `PRD.md` to understand the business problem
   - Look at `INTERVIEWS.md` to understand the business logic
2. **Create your implementation**:
   - Copy `run.sh.template` to `run.sh`
   - Implement your calculation logic
   - Make sure it outputs just the reimbursement amount
3. **Test your solution**: 
   - Run `./eval.sh` to see how you're doing
   - Use the feedback to improve your algorithm
4. **Submit**:
   - Run `./generate_results.sh` to get your final results.
   - Add `arjun-krishna1` to your repo.
   - Complete [the submission form](https://forms.gle/sKFBV2sFo2ADMcRt8).

## Implementation Requirements

Your `run.sh` script must:

- Take exactly 3 parameters: `trip_duration_days`, `miles_traveled`, `total_receipts_amount`
- Output a single number (the reimbursement amount)
- Run in under 5 seconds per test case
- Work without external dependencies (no network calls, databases, etc.)

Example:

```bash
./run.sh 5 250 150.75
# Should output something like: 487.25
```

## Evaluation

Run `./eval.sh` to test your solution against all 1,000 cases. The script will show:

- **Exact matches**: Cases within ±$0.01 of the expected output
- **Close matches**: Cases within ±$1.00 of the expected output
- **Average error**: Mean absolute difference from expected outputs
- **Score**: Lower is better (combines accuracy and precision)

Your submission will be tested against `private_cases.json` which does not include the outputs.

## Submission

When you're ready to submit:

1. Push your solution to a GitHub repository
2. Add `arjun-krishna1` to your repository
3. Submit via the [submission form](https://forms.gle/sKFBV2sFo2ADMcRt8).
4. When you submit the form you will submit your `private_results.txt` which will be used for your final score.

---

**Good luck and Bon Voyage!**
