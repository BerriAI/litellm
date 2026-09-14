import json
import sys
from pathlib import Path


def main() -> None:
    result = json.loads(Path(sys.argv[1]).read_text())
    assert not result.get("errors"), result.get("errors")
    expected = json.loads((Path(__file__).resolve().parents[2] / "tests/integration/contracts.json").read_text())["browser"]
    assert expected and result["stats"]["expected"] == len(expected)
    assert all(result["stats"][name] == 0 for name in ("unexpected", "flaky", "skipped"))
    def cases(suite):
        return tuple(suite.get("specs", ())) + tuple(spec for child in suite.get("suites", ()) for spec in cases(child))
    specs = tuple(spec for suite in result["suites"] for spec in cases(suite))
    repository = Path(__file__).resolve().parents[2]
    report_root = Path(result["config"]["rootDir"])
    assert report_root.is_absolute(), "Playwright rootDir must be explicit"
    observed = tuple(str((report_root / spec["file"]).resolve().relative_to(repository)) + "::" + spec["title"] for spec in specs)
    assert sorted(observed) == sorted(expected)
    for spec in specs:
        tests = spec["tests"]
        assert len(tests) == 1 and len(tests[0]["results"]) == 1
        assert tests[0]["results"][0]["status"] == "passed" and tests[0]["results"][0]["retry"] == 0

    print("One canonical browser contract passed once without skips or retries")


if __name__ == "__main__":
    main()
