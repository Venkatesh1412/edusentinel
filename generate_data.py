"""Write the synthetic demo dataset to ./data as CSV files.

Usage: python generate_data.py
The four CSVs can be uploaded in the dashboard. ground_truth.csv is for
evaluation only and is not an input to the app.
"""
from pathlib import Path

from edusentinel.data_gen import generate_school

out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)
data = generate_school(n_students=120, seed=42)
for name, df in data.items():
    fname = "ground_truth" if name == "truth" else name
    df.to_csv(out / f"{fname}.csv", index=False)
    print(f"wrote data/{fname}.csv ({len(df)} rows)")
