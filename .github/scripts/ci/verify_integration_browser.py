import json
import sys
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict


class BrowserAttempt(TypedDict):
    status: ReadOnly[str]
    retry: ReadOnly[int]


class BrowserTest(TypedDict):
    results: ReadOnly[list[BrowserAttempt]]


class BrowserSpec(TypedDict):
    file: ReadOnly[str]
    title: ReadOnly[str]
    tests: ReadOnly[list[BrowserTest]]


class BrowserSuite(TypedDict):
    specs: NotRequired[ReadOnly[list[BrowserSpec]]]
    suites: NotRequired[ReadOnly[list["BrowserSuite"]]]


def main() -> None:
    result: Final = json.loads(Path(sys.argv[1]).read_text())
    assert not result.get("errors"), result.get("errors")
    expected: Final = json.loads(
        (Path(__file__).resolve().parents[2] / "tests/e2e/ui/tests/integrationCritical/expected.json").read_text()
    )
    assert expected and result["stats"]["expected"] == len(expected)
    assert all(result["stats"][name] == 0 for name in ("unexpected", "flaky", "skipped"))

    def cases(suite: BrowserSuite) -> tuple[BrowserSpec, ...]:
        return tuple(suite.get("specs", ())) + tuple(spec for child in suite.get("suites", ()) for spec in cases(child))

    suites: Final = TypeAdapter(list[BrowserSuite]).validate_python(result["suites"], strict=True)
    specs: Final = tuple(spec for suite in suites for spec in cases(suite))
    repository: Final = Path(__file__).resolve().parents[2]
    report_root: Final = Path(result["config"]["rootDir"])
    assert report_root.is_absolute(), "Playwright rootDir must be explicit"
    observed: Final = tuple(
        str((report_root / spec["file"]).resolve().relative_to(repository)) + "::" + spec["title"] for spec in specs
    )
    assert sorted(observed) == sorted(expected)
    for spec in specs:
        tests: Final = spec["tests"]
        assert len(tests) == 1 and len(tests[0]["results"]) == 1
        assert tests[0]["results"][0]["status"] == "passed" and tests[0]["results"][0]["retry"] == 0

    sys.stdout.write("One canonical browser contract passed once without skips or retries\n")


if __name__ == "__main__":
    main()
