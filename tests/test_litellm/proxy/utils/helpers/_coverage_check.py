#!/usr/bin/env python3
"""Coverage gate for the proxy/utils.py PR3 (bottom helpers).

Reads ``.cov_new.xml`` (produced by ``pytest --cov=litellm.proxy.utils
--cov-branch --cov-report=xml:.cov_new.xml``) from the current working
directory, filters to the bottom-helpers region of
``litellm/proxy/utils.py``, and prints PASS or FAIL.

The region start is resolved at runtime by locating the first pinned
symbol's ``def`` in the source file, so the gate does not drift when
lines above shift. The region end is the last line of the file.

No numbers are printed. Local stopping signal only.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REGION_ANCHOR = "_get_month_end_date"
_LINE_COVERAGE_FLOOR = 0.75

_SOURCE_SUFFIXES = ("proxy/utils.py", "litellm/proxy/utils.py")


def _find_repo_source() -> Path | None:
    for parent in [_HERE, *_HERE.parents]:
        candidate = parent / "litellm" / "proxy" / "utils.py"
        if candidate.is_file():
            return candidate
    return None


def _resolve_region() -> tuple[int, int]:
    src = _find_repo_source()
    if src is None:
        return 0, 0
    text = src.read_text().splitlines()
    pattern = re.compile(rf"^def\s+{re.escape(_REGION_ANCHOR)}\b")
    for idx, line in enumerate(text, start=1):
        if pattern.match(line):
            return idx, len(text)
    return 0, 0


def _line_hits_in_range(xml_path: Path, first: int, last: int) -> tuple[int, int]:
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
            if num < first or num > last:
                continue
            total += 1
            if int(line.get("hits", "0")) > 0:
                hit += 1
        return hit, total
    return 0, 0


def main() -> int:
    xml_path = Path.cwd() / ".cov_new.xml"
    if not xml_path.is_file():
        print("FAIL")
        return 1

    first, last = _resolve_region()
    if first == 0:
        print("FAIL")
        return 1

    hit, total = _line_hits_in_range(xml_path, first, last)
    if total == 0:
        print("FAIL")
        return 1

    ratio = hit / total
    ok = ratio >= _LINE_COVERAGE_FLOOR
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
