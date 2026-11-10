import argparse
import hashlib
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter

from .schemas import Frozen

TASK_ID_PATTERN: Final = re.compile(r"^\[(?P<issue>[a-z_]+)\](?P<faults>.*)\[PERSONA:(?P<persona>[^\]]+)\]$")
SPLITS_ADAPTER: Final = TypeAdapter(dict[str, tuple[str, ...]])


class TaskMeta(Frozen):
    task_id: str
    issue_type: str
    persona: str
    fault_count: int


class TelecomSplit(Frozen):
    source_file_sha256: str
    train: tuple[TaskMeta, ...]
    test: tuple[TaskMeta, ...]
    overlap: tuple[str, ...]
    missing_from_base: tuple[str, ...]
    base_not_covered: tuple[str, ...]


def parse_task_id(task_id: str) -> TaskMeta:
    match: Final = TASK_ID_PATTERN.match(task_id)
    if match is None:
        raise ValueError(f"unrecognized telecom task id: {task_id}")
    return TaskMeta(
        task_id=task_id,
        issue_type=match["issue"],
        persona=match["persona"],
        fault_count=len(match["faults"].split("|")) if match["faults"] else 0,
    )


def build_split(splits: Mapping[str, Sequence[str]], source_sha256: str) -> TelecomSplit:
    train: Final = frozenset(splits["train"])
    test: Final = frozenset(splits["test"])
    base: Final = frozenset(splits["base"])
    return TelecomSplit(
        source_file_sha256=source_sha256,
        train=tuple(parse_task_id(t) for t in sorted(train)),
        test=tuple(parse_task_id(t) for t in sorted(test)),
        overlap=tuple(sorted(train & test)),
        missing_from_base=tuple(sorted((train | test) - base)),
        base_not_covered=tuple(sorted(base - (train | test))),
    )


def render_report(split: TelecomSplit) -> str:
    issue_rows: Final = _count_rows(split, lambda t: t.issue_type)
    persona_rows: Final = _count_rows(split, lambda t: t.persona)
    fault_rows: Final = _count_rows(split, lambda t: str(t.fault_count))
    one_sided: Final = [row[0] for row in issue_rows if 0 in row[1:3]]
    checks: Final = (
        ("train and test share no task", not split.overlap),
        ("every split task is in tau2's base set", not split.missing_from_base),
        ("train + test cover the whole base set", not split.base_not_covered),
        ("every issue type appears in both splits", not one_sided),
    )
    return "\n".join(
        (
            "# Stage 1: tau2 telecom train/test split",
            "",
            f"Source `split_tasks.json` sha256 `{split.source_file_sha256}`",
            "",
            "## Checks",
            "",
            *(f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks),
            "",
            _table("Issue type", issue_rows),
            _table("Persona", persona_rows),
            _table("Faults injected", fault_rows),
        )
    )


def _count_rows(split: TelecomSplit, key: Callable[[TaskMeta], str]) -> list[tuple[str, int, int, str, str]]:
    train: Final = Counter(key(t) for t in split.train)
    test: Final = Counter(key(t) for t in split.test)
    return [
        (name, train[name], test[name], _pct(train[name], len(split.train)), _pct(test[name], len(split.test)))
        for name in sorted(train.keys() | test.keys())
    ]


def _table(title: str, rows: list[tuple[str, int, int, str, str]]) -> str:
    header: Final = f"## {title}\n\n| {title} | train | test | train % | test % |\n|---|---|---|---|---|"
    body: Final = "\n".join(f"| {name} | {a} | {b} | {pa} | {pb} |" for name, a, b, pa, pb in rows)
    return f"{header}\n{body}\n"


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}" if whole else "n/a"


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Freeze the tau2 telecom train/test split")
    parser.add_argument(
        "--split-file", type=Path, required=True, help="tau2 data/tau2/domains/telecom/split_tasks.json"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args: Final = parser.parse_args()
    raw: Final = args.split_file.read_bytes()
    split: Final = build_split(SPLITS_ADAPTER.validate_json(raw), hashlib.sha256(raw).hexdigest())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "split.json").write_text(split.model_dump_json(indent=2))
    (args.out_dir / "split_report.md").write_text(render_report(split))


if __name__ == "__main__":
    main()
