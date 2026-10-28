from typing import Final

import pytest

import litellm
from litellm.proxy.post_call_rules import post_response_rule


@pytest.mark.parametrize("redact", (False, True))
@pytest.mark.parametrize("long_response", (False, True))
def test_diagnostic_privacy_preserves_response_rule_decision(
    redact: bool,
    long_response: bool,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(litellm, "redact_messages_in_debug_logs", redact)
    marker: Final = "PRIVATE_RULE_INPUT"
    response: Final = marker * (20 if long_response else 1)
    expected: Final = (
        {"decision": True}
        if long_response
        else {"decision": False, "message": "This violates LiteLLM Proxy Rules. Response too short"}
    )
    assert post_response_rule(response) == expected
    assert (marker in capsys.readouterr().out) is not redact
