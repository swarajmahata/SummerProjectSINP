import csv
from typing import Tuple, List

def load_csv(filepath: str) -> Tuple[List[float], List[float]]:
    x, y = [], []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2:
                try:
                    x.append(float(row[0]))
                    y.append(float(row[1]))
                except ValueError:
                    continue  # Skip non-numeric rows (like headers)
    return x, y
