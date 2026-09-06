import re
import sys
from typing import Final

SELECTABLE: Final = re.compile(r"^tests/e2e/([A-Za-z0-9_.-]+/)*test_[A-Za-z0-9_.-]+\.py$")
UNSUPPORTED: Final = re.compile(
    r"^tests/e2e/(ui|claude_code|load)/"
    r"|^tests/e2e/llm_translation/realtime/test_realtime_pipecat_audio_e2e\.py$"
    r"|^tests/e2e/batches/test_managed_files_enforcement_e2e\.py$"
    r"|^tests/e2e/guardrails/test_presidio_masking_e2e\.py$"
)
HARNESS: Final = re.compile(
    r"^tests/e2e/[A-Za-z0-9_.-]+\.(py|ini)$"
    r"|^tests/e2e/gateway/"
    r"|^\.github/e2e-stack/"
    r"|^\.github/workflows/test-e2e-changed\.yml$"
)
UNEXPANDED: Final = re.compile(r"[*?\[]")


def is_selectable(path: str) -> bool:
    return SELECTABLE.match(path) is not None and UNSUPPORTED.match(path) is None


def select(changed: tuple[str, ...], canary: tuple[str, ...]) -> tuple[str, ...]:
    direct: Final = frozenset(path for path in changed if is_selectable(path))
    harness_changed: Final = any(HARNESS.match(path) for path in changed)
    canary_tests: Final = frozenset(path for path in canary if harness_changed and is_selectable(path))
    return tuple(sorted(direct | canary_tests))


def main() -> int:
    canary: Final = tuple(sys.argv[1:])
    unexpanded: Final = tuple(path for path in canary if UNEXPANDED.search(path))
    if unexpanded:
        _ = sys.stderr.write(f"the canary paths reached the selector unexpanded: {' '.join(unexpanded)}\n")
        return 1
    changed: Final = tuple(line.strip() for line in sys.stdin if line.strip())
    _ = sys.stdout.write(" ".join(select(changed, canary)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
