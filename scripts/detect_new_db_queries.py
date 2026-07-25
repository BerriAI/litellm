import json
import re
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath

from pydantic import BaseModel, TypeAdapter

RAW_SQL_CALL = re.compile(r"\b(?:query_raw|query_first|execute_raw)\b")
PRISMA_MODEL_CALL = re.compile(
    r"\.db\.[A-Za-z_][A-Za-z0-9_]*\.(?:find_many|find_first|find_unique|find_unique_or_raise|count|aggregate|group_by|create_many|update_many|delete_many)\b"
)
SCHEMA_SUFFIX = ".prisma"
IGNORED_PREFIXES = ("tests/", "ui/", "docs/", "litellm/proxy/_experimental/out/")


class ChangedFile(BaseModel):
    filename: str
    patch: str | None = None


CHANGED_FILES = TypeAdapter(tuple[ChangedFile, ...])


@dataclass(frozen=True, slots=True)
class Finding:
    path: str
    detail: str


def _added_lines(patch: str) -> tuple[str, ...]:
    return tuple(line[1:] for line in patch.splitlines() if line.startswith("+") and not line.startswith("+++"))


def _is_ignored(path: str) -> bool:
    return path.startswith(IGNORED_PREFIXES)


def findings_for_file(changed: ChangedFile) -> tuple[Finding, ...]:
    if _is_ignored(changed.filename):
        return ()
    if PurePosixPath(changed.filename).suffix == SCHEMA_SUFFIX:
        return (Finding(path=changed.filename, detail="prisma schema changed"),)
    if changed.patch is None or PurePosixPath(changed.filename).suffix != ".py":
        return ()
    return tuple(
        Finding(path=changed.filename, detail=line.strip())
        for line in _added_lines(changed.patch)
        if RAW_SQL_CALL.search(line) or PRISMA_MODEL_CALL.search(line)
    )


def detect(files: tuple[ChangedFile, ...]) -> tuple[Finding, ...]:
    return tuple(finding for changed in files for finding in findings_for_file(changed))


def main() -> int:
    findings = detect(CHANGED_FILES.validate_python(json.load(sys.stdin)))
    for finding in findings:
        print(f"{finding.path}: {finding.detail}")
    return 0 if findings else 1


if __name__ == "__main__":
    sys.exit(main())
