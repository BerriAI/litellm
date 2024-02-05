import sys
import filecmp
import difflib


def show_diff(file1, file2):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    diff = difflib.unified_diff(lines1, lines2, lineterm="")

    for line in diff:
        print(line)


def main(argv=None):
    print(
        "comparing model_prices_and_context_window, and litellm/model_prices_and_context_window_backup.json files.......... checking they match",
        argv,
    )

    file1 = "model_prices_and_context_window.json"
    file2 = "litellm/model_prices_and_context_window_backup.json"
    cmp_result = filecmp.cmp(file1, file2, shallow=False)
    if cmp_result:
        print(f"Passed ! Files {file1} and {file2} match.")
        return 0
    else:
        # show the diff
        print(f"Failed ! Files {file1} and {file2} do not match.")
        print("\nDiff")
        show_diff(file1, file2)

        return 1


if __name__ == "__main__":
    sys.exit(main())
