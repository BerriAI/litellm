import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final


def main() -> int:
    selected: Final = tuple(sys.argv[2:])
    try:
        report: Final = ET.parse(Path(sys.argv[1])).getroot()
    except (ET.ParseError, OSError):
        _ = sys.stdout.write("::error::could not read the test execution report\n")
        return 1
    cases: Final = tuple(report.iter("testcase"))
    passed: Final = frozenset(
        case.get("file") for case in cases if all(case.find(tag) is None for tag in ("skipped", "failure", "error"))
    )
    missing: Final = tuple(path for path in selected if path not in passed)
    for path in selected:
        collected: Final = sum(case.get("file") == path for case in cases)
        skipped: Final = sum(case.get("file") == path and case.find("skipped") is not None for case in cases)
        _ = sys.stdout.write(f"{path}: {collected} collected, {skipped} skipped\n")
        for case in cases:
            if case.get("file") != path or all(case.find(tag) is None for tag in ("failure", "error")):
                continue
            _ = sys.stdout.write(f"  failed: {case.get('classname', '')}::{case.get('name', '')}\n")
    if (
        selected
        and not missing
        and not any(case.find(tag) is not None for case in cases for tag in ("failure", "error"))
    ):
        return 0
    _ = sys.stdout.write("::error::every selected file must execute a passing test, with no failures or errors\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
