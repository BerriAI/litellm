from pathlib import Path


def read_pair(first: Path, second: Path) -> tuple[str, str]:
    with first.open(encoding="utf-8") as left:
        with second.open(encoding="utf-8") as right:
            return left.read(), right.read()
