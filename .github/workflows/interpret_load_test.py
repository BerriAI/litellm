import csv
import os
from github import Github


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
    print("interpreted_results_str() output: ", interpreted_results_str)
    return interpreted_results_str


if __name__ == "__main__":
    csv_file = "load_test_stats.csv"  # Change this to the path of your CSV file
    interpreted_results_str = interpret_results(csv_file)
    markdown_table = csv_to_markdown(csv_file)

    # Update release body with interpreted results
    github_token = os.getenv("GITHUB_TOKEN")
    g = Github(github_token)
    repo = g.get_repo(
        "BerriAI/litellm"
    )  # Replace with your repository's username and name
    latest_release = repo.get_latest_release()
    print("got latest release: ", latest_release)
    print("latest release body: ", latest_release.body)
    print("markdown table: ", markdown_table)
    new_release_body = (
        latest_release.body + "\n\n" + interpreted_results_str + "\n\n" + markdown_table
    )
    print("new release body: ", new_release_body)
    try:
        latest_release.update_release(
            name=latest_release.tag_name,
            message=new_release_body,
        )
    except Exception as e:
        print(e)
