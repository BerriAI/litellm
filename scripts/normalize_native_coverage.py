"""Map coverage collected from an isolated wheel back to repository paths."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def main(report_path: Path, installed_package: Path) -> None:
    package_root = installed_package.resolve()
    report = ET.parse(report_path)
    for entry in report.iter("class"):
        raw = entry.get("filename")
        if raw is None:
            raise ValueError("coverage class has no filename")
        relative = next(
            (
                candidate.resolve().relative_to(package_root)
                for candidate in (Path(raw), Path("/") / raw, package_root / raw, package_root.parent / raw)
                if candidate.resolve().is_relative_to(package_root)
            ),
            None,
        )
        if relative is None:
            raise ValueError(f"coverage file is outside the installed package: {raw}")
        entry.set("filename", (Path("litellm") / relative).as_posix())

    sources = report.find("./sources")
    if sources is None:
        raise ValueError("coverage report has no sources")
    sources.clear()
    ET.SubElement(sources, "source").text = "."
    report.write(report_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
