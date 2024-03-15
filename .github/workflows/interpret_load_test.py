import csv


def csv_to_markdown(csv_file):
    markdown_table = ""

    # Read CSV file
    with open(csv_file, newline="") as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)

        # Create header row
        markdown_table += "|" + " | ".join(header) + "|\n"
        markdown_table += "|" + " | ".join(["---"] * len(header)) + "|\n"

        # Add data rows
        for row in csvreader:
            markdown_table += "|" + " | ".join(row) + "|\n"

    return markdown_table


def interpret_results(csv_file):
    below_300_ms = True
    below_300_s = True
    requests_per_sec = []
    interpreted_results_str = ""
    with open(csv_file, newline="") as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            median_response_time = float(
                row["Median Response Time"].strip().rstrip("ms")
            )
            average_response_time = float(
                row["Average Response Time"].strip().rstrip("s")
            )
            result_str = f"endpoint: {row['Name']}, median_response_time: {median_response_time}, average_response_time: {average_response_time}, requests_per_sec: {row['Requests/s']}"
            if median_response_time < 300 and average_response_time < 300:
                result_str += " Passed ✅\n"
            else:
                result_str += " Failed ❌\n"
            print(result_str)
        interpreted_results_str += result_str
    print(interpreted_results_str)


if __name__ == "__main__":
    csv_file = "load_test_stats.csv"  # Change this to the path of your CSV file
    interpret_results(csv_file)
    markdown_table = csv_to_markdown(csv_file)
    print(markdown_table)
