import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RootFileCheck:
    root_files: tuple[str, ...]
    max_root_files: int

    @property
    def count(self) -> int:
        return len(self.root_files)

    @property
    def within_limit(self) -> bool:
        return self.count <= self.max_root_files


def root_files_of(tracked_files: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path for path in tracked_files if "/" not in path)


def list_tracked_files() -> tuple[str, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        capture_output=True,
        check=True,
        text=True,
    )
    return tuple(path for path in completed.stdout.split("\0") if path)


def report(check: RootFileCheck) -> str:
    if check.within_limit:
        return f"Root file count OK: {check.count} tracked file(s) <= limit {check.max_root_files}"
    listing = "\n".join(f"  {path}" for path in sorted(check.root_files))
    return (
        f"::error::Too many files in the repository root: {check.count} tracked "
        f"file(s), limit is {check.max_root_files}\n"
        f"{listing}\n"
        "Move new files into an appropriate subdirectory instead of the repo root. "
        "If a new root file is genuinely required, raise MAX_ROOT_FILES in "
        ".github/workflows/check-root-file-count.yml in the same PR so the bump is "
        "reviewed"
    )


def main(argv: tuple[str, ...]) -> int:
    max_root_files = int(argv[1])
    check = RootFileCheck(
        root_files=root_files_of(list_tracked_files()),
        max_root_files=max_root_files,
    )
    print(report(check))
    return 0 if check.within_limit else 1


if __name__ == "__main__":
    sys.exit(main(tuple(sys.argv)))
