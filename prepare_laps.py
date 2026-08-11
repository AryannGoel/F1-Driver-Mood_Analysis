"""Export one driver's real lap times to laps.csv.

  python prepare_laps.py 2023 "British Grand Prix" R VER 28 42
"""
import sys
import pandas as pd
from laps import lap_times_fastf1

if __name__ == "__main__":
    year, gp, ses, drv, lo, hi = (
        int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4],
        int(sys.argv[5]), int(sys.argv[6]),
    )
    rows = lap_times_fastf1(year, gp, ses, drv, lo, hi)
    pd.DataFrame(rows, columns=["lap", "t"]).to_csv("laps.csv", index=False)
    print(f"wrote laps.csv with {len(rows)} laps for {drv}")
