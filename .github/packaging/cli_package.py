import argparse
import base64
import copy
import csv
import hashlib
import io
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final
from zipfile import ZipFile, ZipInfo

PACKAGE_DIR: Final = "litellm/cli"
PACKAGE_NAME: Final = "litellm_cli"


def editable_support(source: Path) -> tuple[tuple[str, bytes], ...]:
    module: Final = "_litellm_cli_editable"
    finder: Final = (
        "import importlib.util\nimport sys\n\n"
        "class _Finder:\n"
        "    @staticmethod\n"
        "    def find_spec(fullname, path=None, target=None):\n"
        f"        if fullname == {PACKAGE_NAME!r}:\n"
        "            return importlib.util.spec_from_file_location(\n"
        f"                fullname, {str(source / '__init__.py')!r},\n"
        f"                submodule_search_locations=[{str(source)!r}])\n"
        "        return None\n\n"
        "sys.meta_path.insert(0, _Finder())\n"
    )
    return ((f"{module}.py", finder.encode()), (f"{module}.pth", f"import {module}\n".encode()))


def _name(name: str) -> str:
    prefix: Final = f"{PACKAGE_DIR}/"
    return f"{PACKAGE_NAME}/{name.removeprefix(prefix)}" if name.startswith(prefix) else name


def _info(original: ZipInfo) -> ZipInfo:
    mapped: Final = copy.copy(original)
    mapped.filename = _name(original.filename)
    return mapped


def _record(entries: tuple[tuple[ZipInfo, bytes], ...], record: str) -> bytes:
    buffer: Final = io.StringIO(newline="")
    writer: Final = csv.writer(buffer, lineterminator="\n")
    writer.writerows(
        (
            info.filename,
            "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode(),
            str(len(data)),
        )
        for info, data in entries
    )
    writer.writerow((record, "", ""))
    return buffer.getvalue().encode()


def map_wheel(wheel: Path, *, source: Path | None = None) -> None:
    with ZipFile(wheel) as archive:
        records: Final = tuple(info for info in archive.infolist() if info.filename.endswith(".dist-info/RECORD"))
        if len(records) != 1:
            raise ValueError("A wheel must contain exactly one RECORD")
        entries: Final = tuple(
            (_info(info), archive.read(info))
            for info in archive.infolist()
            if info != records[0] and (source is None or not info.filename.startswith(f"{PACKAGE_DIR}/"))
        )
    additions: Final = () if source is None else tuple((ZipInfo(name), data) for name, data in editable_support(source))
    contents: Final = (*entries, *additions)
    names: Final = tuple(info.filename for info, _ in contents)
    if len(names) != len(frozenset(names)):
        raise ValueError("The CLI package mapping would overwrite a wheel member")
    if source is None and not any(name.startswith(f"{PACKAGE_NAME}/") for name in names):
        raise ValueError("The wheel does not contain the CLI package")
    with NamedTemporaryFile(dir=wheel.parent, suffix=".whl", delete=False) as temporary:
        destination: Final = Path(temporary.name)
    try:
        with ZipFile(destination, "w") as archive:
            for info, data in contents:
                archive.writestr(info, data)
            archive.writestr(records[0], _record(contents, records[0].filename))
        os.replace(destination, wheel)
    finally:
        destination.unlink(missing_ok=True)


if __name__ == "__main__":
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    for wheel in parser.parse_args().wheels:
        map_wheel(wheel)
