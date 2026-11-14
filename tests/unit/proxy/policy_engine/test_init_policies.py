"""The policy table printed at boot keeps its terminal form in plain mode and is JSON when json_logs is on."""

import json
import re
import sys
from collections.abc import Mapping, Sequence
from io import StringIO
from typing import Final

import pytest

import litellm
from litellm._logging import turn_on_json
from litellm.proxy.policy_engine.init_policies import _print_policies_on_startup

POLICIES: Final[Mapping[str, Mapping[str, object]]] = {
    "pii-policy": {
        "inherit": "base",
        "description": "Block PII",
        "guardrails": {"add": ["pii-guard"], "remove": ["noop"]},
        "condition": {"model": "gpt-4o"},
    }
}
ATTACHMENTS: Final[Sequence[Mapping[str, object]]] = ({"policy": "pii-policy", "scope": "*", "teams": ["team-a"]},)
POLICY_ROWS: Final = (
    "  - pii-policy (inherits: base)",
    "      description: Block PII",
    "      guardrails.add: ['pii-guard']",
    "      guardrails.remove: ['noop']",
    "      condition.model: gpt-4o",
)
PLAIN_POLICY_TABLE: Final = (
    "\x1b[92m\nLiteLLM Policy Engine: Loaded 1 policies\x1b[0m\n\n"
    "\x1b[94m  - pii-policy (inherits: base)\x1b[0m\n"
    "      description: Block PII\n"
    "      guardrails.add: ['pii-guard']\n"
    "      guardrails.remove: ['noop']\n"
    "      condition.model: gpt-4o\n"
)
NO_ATTACHMENTS_WARNING: Final = (
    "Warning: No policy_attachments configured. Policies will not be applied to any requests."
)


class _Stream(StringIO):
    def __init__(self, terminal: bool) -> None:
        super().__init__()
        self._terminal = terminal

    def isatty(self) -> bool:
        return self._terminal


@pytest.mark.parametrize(
    ("attachments", "expected"),
    [
        (
            ATTACHMENTS,
            PLAIN_POLICY_TABLE
            + "\n\x1b[93mPolicy Attachments: 1 attachment(s)\x1b[0m\n"
            + "  - pii-policy -> scope=* (global), teams=['team-a']\n\n",
        ),
        (None, PLAIN_POLICY_TABLE + f"\n\x1b[93m{NO_ATTACHMENTS_WARNING}\x1b[0m\n\n"),
    ],
    ids=["with-attachments", "without-attachments"],
)
@pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "piped"])
def test_policy_table_prints_unchanged_in_plain_mode(
    monkeypatch: pytest.MonkeyPatch,
    attachments: Sequence[Mapping[str, object]] | None,
    expected: str,
    terminal: bool,
) -> None:
    """Operators without JSON logs keep the colored policy table byte for byte on a terminal, and the same text
    without color codes on a piped stdout."""
    monkeypatch.setattr(litellm, "json_logs", False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    stdout: Final = _Stream(terminal)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", _Stream(terminal))

    _print_policies_on_startup(POLICIES, attachments)

    assert stdout.getvalue() == (expected if terminal else re.sub(r"\x1b\[\d+m", "", expected))


def test_policy_table_is_one_json_record_in_json_mode(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """With JSON logs on the whole table, attachments included, is one INFO record without color codes."""
    monkeypatch.setattr(litellm, "json_logs", True)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    turn_on_json()

    _print_policies_on_startup(POLICIES, ATTACHMENTS)

    out, err = capfd.readouterr()
    assert err == ""
    lines: Final = [raw for raw in out.splitlines() if raw.strip()]
    assert len(lines) == 1, lines
    record: Final = json.loads(lines[0])
    assert record["level"] == "INFO"
    assert record["message"] == "\n".join(
        (
            "LiteLLM Policy Engine: Loaded 1 policies",
            "",
            *POLICY_ROWS,
            "",
            "Policy Attachments: 1 attachment(s)",
            "  - pii-policy -> scope=* (global), teams=['team-a']",
        )
    )


def test_missing_attachments_warning_is_a_second_json_record(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """Without attachments the table and the yellow warning are an INFO and a WARNING record on stdout,
    where the two prints put them."""
    monkeypatch.setattr(litellm, "json_logs", True)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    turn_on_json()

    _print_policies_on_startup(POLICIES, None)

    out, err = capfd.readouterr()
    assert err == ""
    records: Final = [json.loads(raw) for raw in out.splitlines() if raw.strip()]
    assert [record["message"] for record in records] == [
        "\n".join(("LiteLLM Policy Engine: Loaded 1 policies", "", *POLICY_ROWS)),
        NO_ATTACHMENTS_WARNING,
    ]
    assert [record["level"] for record in records] == ["INFO", "WARNING"]
