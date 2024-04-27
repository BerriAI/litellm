import sys
import filecmp
import shutil
from typing import Any


def main(argv: Any = None) -> int:
    print(
        "Comparing model_prices_and_context_window and litellm/model_prices_and_context_window_backup.json files... checking if they match."
    )

    file1 = "model_prices_and_context_window.json"
    file2 = "litellm/model_prices_and_context_window_backup.json"

    cmp_result = filecmp.cmp(file1, file2, shallow=False)

    if cmp_result:
        print(f"Passed! Files {file1} and {file2} match.")
        return 0
    else:
        print(
            f"Failed! Files {file1} and {file2} do not match. Copying content from {file1} to {file2}."
        )
        copy_content(file1, file2)
        return 1


def copy_content(source: str, destination: str) -> None:
    shutil.copy2(source, destination)


if __name__ == "__main__":
    sys.exit(main())
