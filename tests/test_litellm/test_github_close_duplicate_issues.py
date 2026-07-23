from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "close_duplicate_issues.py"


@pytest.fixture(scope="module")
def closer_module():
    spec = importlib.util.spec_from_file_location("close_duplicate_issues", SCRIPT_PATH)
    assert spec and spec.loader, f"Could not load spec for {SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["close_duplicate_issues"] = module
    spec.loader.exec_module(module)
    return module


def _issue(number: int, title: str, body: str = "") -> dict:
    return {"number": number, "title": title, "body": body}


def _fake_gh_cli(pages: list[list[dict]]):
    def fake_gh(*args: str) -> str:
        assert args[0] == "api"
        assert "--paginate" in args
        if "--slurp" in args:
            return json.dumps(pages, ensure_ascii=False)
        return "".join(json.dumps(page, ensure_ascii=False) for page in pages)

    return fake_gh


class TestFetchOpenIssues:
    def test_should_parse_multi_page_payload_with_unescaped_line_separators(self, closer_module, monkeypatch):
        pages = [
            [_issue(1, "first bug", body="traceback\u2028line two\u2028line three")],
            [_issue(2, "second bug"), _issue(3, "third bug")],
        ]
        monkeypatch.setattr(closer_module, "gh", _fake_gh_cli(pages))

        issues = closer_module.fetch_open_issues("BerriAI/litellm")

        assert [i["number"] for i in issues] == [1, 2, 3]

    def test_should_request_slurp_mode(self, closer_module, monkeypatch):
        calls: list[tuple[str, ...]] = []

        def recording_gh(*args: str) -> str:
            calls.append(args)
            return json.dumps([[]])

        monkeypatch.setattr(closer_module, "gh", recording_gh)

        closer_module.fetch_open_issues("BerriAI/litellm")

        assert len(calls) == 1
        assert "--slurp" in calls[0]

    def test_should_filter_out_pull_requests(self, closer_module, monkeypatch):
        pages = [
            [
                _issue(1, "real issue"),
                {**_issue(2, "a pr"), "pull_request": {"url": "https://x"}},
            ]
        ]
        monkeypatch.setattr(closer_module, "gh", _fake_gh_cli(pages))

        issues = closer_module.fetch_open_issues("BerriAI/litellm")

        assert [i["number"] for i in issues] == [1]
