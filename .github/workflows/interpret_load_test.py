import csv
import os
from github import Github


def interpret_results(csv_file):
    with open(csv_file, newline="") as csvfile:
        csvreader = csv.DictReader(csvfile)
        rows = list(csvreader)
        """
        in this csv reader
        - Create 1 new column "Status"
        - if a row has a median response time < 300 and an average response time < 300, Status = "Passed ✅"
        - if a row has a median response time >= 300 or an average response time >= 300, Status = "Failed ❌"
        - Order the table in this order Name, Status, Median Response Time, Average Response Time, Requests/s,Failures/s, Min Response Time, Max Response Time, all other columns
        """

        # Add a new column "Status"
        for row in rows:
            median_response_time = float(
                row["Median Response Time"].strip().rstrip("ms")
            )
            average_response_time = float(
                row["Average Response Time"].strip().rstrip("s")
            )

            request_count = int(row["Request Count"])
            failure_count = int(row["Failure Count"])

            failure_percent = round((failure_count / request_count) * 100, 2)

            # Determine status based on conditions
            if (
                median_response_time < 300
                and average_response_time < 300
                and failure_percent < 5
            ):
                row["Status"] = "Passed ✅"
            else:
                row["Status"] = "Failed ❌"

        # Construct Markdown table header
        markdown_table = "| Name | Status | Median Response Time (ms) | Average Response Time (ms) | Requests/s | Failures/s | Request Count | Failure Count | Min Response Time (ms) | Max Response Time (ms) |"
        markdown_table += (
            "\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        )

        # Construct Markdown table rows
        for row in rows:
            markdown_table += f"\n| {row['Name']} | {row['Status']} | {row['Median Response Time']} | {row['Average Response Time']} | {row['Requests/s']} | {row['Failures/s']} | {row['Request Count']} | {row['Failure Count']} | {row['Min Response Time']} | {row['Max Response Time']} |"
    print("markdown table: ", markdown_table)
    return markdown_table


def _get_docker_run_command_stable_release(release_version):
    return f"""
\n\n
## Docker Run LiteLLM Proxy

```
docker run \\
-e STORE_MODEL_IN_DB=True \\
-p 4000:4000 \\
ghcr.io/berriai/litellm:litellm_stable_release_branch-{release_version}
```
    """


def _get_docker_run_command(release_version):
    return f"""
\n\n
## Docker Run LiteLLM Proxy

```
docker run \\
-e STORE_MODEL_IN_DB=True \\
-p 4000:4000 \\
ghcr.io/berriai/litellm:main-{release_version}
```
    """


def get_docker_run_command(release_version):
    if "stable" in release_version:
        return _get_docker_run_command_stable_release(release_version)
    else:
        return _get_docker_run_command(release_version)


if __name__ == "__main__":
    return
    csv_file = "load_test_stats.csv"  # Change this to the path of your CSV file
    markdown_table = interpret_results(csv_file)

    # Update release body with interpreted results
    github_token = os.getenv("GITHUB_TOKEN")
    g = Github(github_token)
    repo = g.get_repo(
        "BerriAI/litellm"
    )  # Replace with your repository's username and name
    latest_release = repo.get_latest_release()
    print("got latest release: ", latest_release)
    print(latest_release.title)
    print(latest_release.tag_name)

    release_version = latest_release.title

    print("latest release body: ", latest_release.body)
    print("markdown table: ", markdown_table)

    # check if "Load Test LiteLLM Proxy Results" exists
    existing_release_body = latest_release.body
    if "Load Test LiteLLM Proxy Results" in latest_release.body:
        # find the "Load Test LiteLLM Proxy Results" section and delete it
        start_index = latest_release.body.find("Load Test LiteLLM Proxy Results")
        existing_release_body = latest_release.body[:start_index]

    docker_run_command = get_docker_run_command(release_version)
    print("docker run command: ", docker_run_command)

    new_release_body = (
        existing_release_body
        + docker_run_command
        + "\n\n"
        + "### Don't want to maintain your internal proxy? get in touch 🎉"
        + "\nHosted Proxy Alpha: https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat"
        + "\n\n"
        + "## Load Test LiteLLM Proxy Results"
        + "\n\n"
        + markdown_table
    )
    print("new release body: ", new_release_body)
    try:
        latest_release.update_release(
            name=latest_release.tag_name,
            message=new_release_body,
        )
    except Exception as e:
        print(e)
