import csv


def read_csv_to_list(filename):
    """Read a CSV file and return its contents as a list of rows."""
    with open(filename, "r") as file:
        reader = csv.reader(file)
        return list(reader)


def compare_csvs(file1, file2):
    """Compare two CSV files and print the rows that don't match."""
    csv1_data = read_csv_to_list(file1)
    csv2_data = read_csv_to_list(file2)

    csv1_set = set(tuple(row) for row in csv1_data)
    csv2_set = set(tuple(row) for row in csv2_data)

    # Rows in file1 but not in file2
    for row1, row2 in csv1_set - csv2_set, csv2_set - csv1_set:
        print(f"In {file1} but not in {file2}:", row1)

    # Rows in file2 but not in file1
    for row in csv2_set - csv1_set:
        print(f"In {file2} but not in {file1}:", row)


if __name__ == "__main__":
    file1 = "C:\school\class repos\csci-4353-JorgeCaPe\labs\lab15\predictions.csv"
    file2 = "C:\school\class repos\csci-4353-JorgeCaPe\labs\lab15\predictions1.csv"

    compare_csvs(file1, file2)
