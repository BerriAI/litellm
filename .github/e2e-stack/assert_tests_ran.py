import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final


def main() -> int:
    report: Final = ET.parse(Path(sys.argv[1])).getroot()
    suites: Final = tuple(report.iter("testsuite"))
    collected: Final = sum(int(suite.get("tests", "0")) for suite in suites)
    skipped: Final = sum(int(suite.get("skipped", "0")) for suite in suites)
    executed: Final = collected - skipped
    _ = sys.stdout.write(f"executed {executed} of {collected} collected tests ({skipped} skipped)\n")
    if executed > 0:
        return 0
    _ = sys.stdout.write("::error::every selected test was skipped, so nothing was verified\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
