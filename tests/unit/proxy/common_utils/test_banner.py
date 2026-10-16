"""The CLI banner is terminal art in plain mode and one JSON record when json_logs is on."""

import json
import sys
from typing import Final

import litellm
from litellm._logging import turn_on_json
from litellm.proxy.common_utils.banner import LITELLM_BANNER, show_banner


def test_show_banner_prints_the_banner_between_blank_lines_in_plain_mode(monkeypatch, capfd):
    """Operators without JSON logs see the banner exactly as before."""
    monkeypatch.setattr(litellm, "json_logs", False)

    show_banner()

    assert capfd.readouterr() == (f"\n{LITELLM_BANNER}\n\n", "")


def test_show_banner_is_one_json_record_in_json_mode(monkeypatch, capfd):
    """With JSON logs on, the six banner rows are one INFO record with their indentation kept, so the
    first line a worker writes parses like every other."""
    monkeypatch.setattr(litellm, "json_logs", True)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # turn_on_json swaps it; put it back at teardown
    turn_on_json()

    show_banner()

    out, err = capfd.readouterr()
    assert err == ""
    lines: Final = [raw for raw in out.splitlines() if raw.strip()]
    assert len(lines) == 1, lines
    record: Final = json.loads(lines[0])
    assert record["message"] == LITELLM_BANNER
    assert record["level"] == "INFO"
