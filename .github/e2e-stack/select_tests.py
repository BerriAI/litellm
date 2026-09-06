import re
import sys
from typing import Final

SELECTABLE: Final = re.compile(r"^tests/e2e/([A-Za-z0-9_.-]+/)*test_[A-Za-z0-9_.-]+\.py$")
OWN_LANE: Final = re.compile(
    r"^tests/e2e/(ui|claude_code|load)/|^tests/e2e/batches/test_managed_files_enforcement_e2e\.py$"
)
HARNESS: Final = re.compile(
    r"^tests/e2e/[A-Za-z0-9_.-]+\.(py|ini)$"
    r"|^tests/e2e/gateway/"
    r"|^\.github/e2e-stack/"
    r"|^\.github/workflows/test-e2e-changed\.yml$"
)


def select(changed: tuple[str, ...], canary: tuple[str, ...]) -> tuple[str, ...]:
    direct: Final = frozenset(path for path in changed if SELECTABLE.match(path) and not OWN_LANE.match(path))
    harness_changed: Final = any(HARNESS.match(path) for path in changed)
    canary_tests: Final = frozenset(path for path in canary if harness_changed and SELECTABLE.match(path))
    return tuple(sorted(direct | canary_tests))


def main() -> int:
    changed: Final = tuple(line.strip() for line in sys.stdin if line.strip())
    _ = sys.stdout.write(" ".join(select(changed, tuple(sys.argv[1:]))) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
