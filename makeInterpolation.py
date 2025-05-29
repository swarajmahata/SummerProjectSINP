import csv
from interpolator import Interpolator, LinearInterpolator, CubicInterpolator


def load_csv_unique(filename):
    data = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                try:
                    x_val = float(row[0])
                    y_val = float(row[1])
                    if x_val not in data:
                        data[x_val] = y_val  # keep first occurrence
                except ValueError:
                    continue
    x_unique, y_unique = zip(*sorted(data.items()))
    return list(x_unique), list(y_unique)


def main():
    # Load actual data from your CSV with unique x-values
    x, y = load_csv_unique("Bubble Nucleation efficiency vs E_R_E_Rth.csv")

    # Interpolate using cubic method
    method = CubicInterpolator()
    interpolator = Interpolator(method)
    myInterpolation = interpolator.interpolate(x, y)

    # Example usage
    print("Interpolated value at x=1.5:", myInterpolation(1.5))


if __name__ == "__main__":
    main()
