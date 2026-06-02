#!/usr/bin/env python3
"""Coverage gate for the proxy/utils.py PR3 (bottom helpers).

Reads ``.cov_new.xml`` (produced by ``pytest --cov=litellm.proxy.utils
--cov-branch --cov-report=xml:.cov_new.xml``) from the current working
directory, filters to the bottom-helpers line range in
``litellm/proxy/utils.py``, and prints PASS or FAIL.

No numbers are printed. Local stopping signal only.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_FIRST_LINE = 5541
_LAST_LINE = 6274
_LINE_COVERAGE_FLOOR = 0.75

_SOURCE_SUFFIXES = ("proxy/utils.py", "litellm/proxy/utils.py")


def _line_hits_in_range(xml_path: Path) -> tuple[int, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for class_elem in root.iter("class"):
        filename = class_elem.get("filename", "")
        if not any(filename.endswith(s) for s in _SOURCE_SUFFIXES):
            continue
        total = 0
        hit = 0
        for line in class_elem.iter("line"):
            num_str = line.get("number")
            if num_str is None:
                continue
            num = int(num_str)
            if num < _FIRST_LINE or num > _LAST_LINE:
                continue
            total += 1
            if int(line.get("hits", "0")) > 0:
                hit += 1
        return hit, total
    return 0, 0


def main() -> int:
    xml_path = Path.cwd() / ".cov_new.xml"
    if not xml_path.is_file():
        print("FAIL", file=sys.stdout)
        return 1

    hit, total = _line_hits_in_range(xml_path)
    if total == 0:
        print("FAIL")
        return 1

    ratio = hit / total
    ok = ratio >= _LINE_COVERAGE_FLOOR
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
