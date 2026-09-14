from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Final
from xml.etree import ElementTree

SUITES: Final = {
    "startup": (("test_startup.py",), 12),
    "recovery": (("test_recovery.py",), 15),
    "legacy": (("test_legacy.py", "test_pooling.py"), 11),
}


def successful_junit(path: Path, expected: int, exit_code: int) -> bool:
    if exit_code != 0 or not path.is_file():
        return False
    try:
        root: Final = ElementTree.parse(path).getroot()
    except ElementTree.ParseError:
        return False
    cases: Final = tuple(root.iter("testcase"))
    identities: Final = frozenset((case.get("classname"), case.get("name")) for case in cases)
    return len(cases) == len(identities) == expected and all(
        not any(case.find(tag) is not None for tag in ("failure", "error", "skipped")) for case in cases
    )


def output(*command: str) -> str:
    return subprocess.check_output(command, text=True, timeout=90).strip()


def record_image() -> None:
    source: Final = output("git", "rev-parse", "HEAD")
    image: Final = output("docker", "image", "inspect", "litellm-docker-database:ci", "--format", "{{.Id}}")
    revision: Final = output(
        "docker",
        "image",
        "inspect",
        "litellm-docker-database:ci",
        "--format",
        '{{index .Config.Labels "org.opencontainers.image.revision"}}',
    )
    assert re.fullmatch(r"[0-9a-f]{40}", source), "Invalid source revision"
    assert revision == source, "Candidate image revision differs from the tested source"
    Path("migration-image.json").write_text(
        json.dumps(
            {
                "source_sha": source,
                "image_id": image,
                "candidate_image": os.environ.get("MIGRATION_CANDIDATE_IMAGE", ""),
            }
        )
    )


def main() -> int:
    suite: Final = os.environ["MIGRATION_TEST_SUITE"]
    files, expected = SUITES[suite]
    metadata: Final = json.loads(Path("migration-image.json").read_text())
    assert metadata["source_sha"] == output("git", "rev-parse", "HEAD"), "Image and test source revisions differ"
    assert metadata["image_id"] == output(
        "docker", "image", "inspect", os.environ["LITELLM_MIGRATION_TEST_IMAGE"], "--format", "{{.Id}}"
    ), "Loaded image differs from the build output"
    assert metadata["candidate_image"] == os.environ.get("MIGRATION_CANDIDATE_IMAGE", ""), "Wrong release candidate"
    destination: Final = Path(os.environ["MIGRATION_TEST_OUTPUT"])
    junit: Final = destination / "junit" / "results.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    result: Final = subprocess.run(
        (
            sys.executable,
            "-m",
            "pytest",
            *(f"tests/e2e/migrations/{name}" for name in files),
            "-vv",
            "--tb=short",
            "--durations=10",
            f"--junitxml={junit}",
            "-o",
            "addopts=",
            "--reruns=0",
        ),
        check=False,
        timeout=1200,
    )
    passed: Final = successful_junit(junit, expected, result.returncode)
    (destination / "verdict.json").write_text(
        json.dumps(
            {
                **metadata,
                "suite": suite,
                "expected_cases": expected,
                "passed": passed,
                "pytest_exit_code": result.returncode,
                "test_revision": metadata["source_sha"],
                "workflow_id": os.environ.get("CIRCLE_WORKFLOW_ID", ""),
                "job_number": os.environ.get("CIRCLE_BUILD_NUM", ""),
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["record-image"]:
        record_image()
    else:
        raise SystemExit(main())
