import importlib.util
import json
from pathlib import Path

_MODULE_PATH = Path(__file__).with_name("close_duplicate_issues.py")
_spec = importlib.util.spec_from_file_location("close_duplicate_issues", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
close_duplicate_issues = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(close_duplicate_issues)

parse_slurped_issues = close_duplicate_issues.parse_slurped_issues


def test_parse_handles_line_separator_chars_in_body() -> None:
    payload = [
        [
            {"number": 1, "title": "first", "body": "before\u2028after\x0cend"},
            {"number": 2, "title": "second", "body": "normal"},
        ]
    ]
    raw = json.dumps(payload, ensure_ascii=False)
    assert "\u2028" in raw
    assert len(raw.splitlines()) > 1

    issues = parse_slurped_issues(raw)

    assert [i["number"] for i in issues] == [1, 2]
    assert issues[0]["body"] == "before\u2028after\x0cend"


def test_parse_flattens_pages_and_drops_pull_requests() -> None:
    payload = [
        [
            {"number": 1, "title": "issue one"},
            {"number": 2, "title": "a pr", "pull_request": {"url": "x"}},
        ],
        [
            {"number": 3, "title": "issue three"},
        ],
    ]

    issues = parse_slurped_issues(json.dumps(payload))

    assert [i["number"] for i in issues] == [1, 3]
