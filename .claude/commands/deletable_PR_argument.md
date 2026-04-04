This is a table of PRs

$ARGUMENT[0]

from the above table,
for each row,
    retrieve the PR number and PR age (in days)
    checkout to the PR
    analyze the PR and check if still is valid, specially considering its age: this is the verdict for knowing whether to close the PR, or seeing how to reactivate it (if it is worth it)

Output in TSV format, containing PR number, link, author, PR age, verdict, and REASONING for your verdict
