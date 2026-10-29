from typing import Final

from litellm.passthrough.utils import BasePassthroughUtils


def test_pass_through_prefixed_headers_keeps_only_x_pass_headers_in_any_case() -> None:
    inbound: Final = {
        "authorization": "Bearer sk-litellm-key",
        "x-litellm-api-key": "sk-litellm-key",
        "content-type": "application/json",
        "x-amzn-bedrock-guardrailidentifier": "native-guardrail",
        "x-passthrough-id": "not-a-pass-header",
        "x-pass-x-amzn-bedrock-trace": "DISABLED",
        "X-Pass-X-Amzn-Bedrock-GuardrailIdentifier": "caller-guardrail",
        "X-PASS-ANTHROPIC-BETA": "beta-1",
    }

    selected: Final = BasePassthroughUtils.pass_through_prefixed_headers(inbound)

    assert dict(selected) == {
        "x-pass-x-amzn-bedrock-trace": "DISABLED",
        "X-Pass-X-Amzn-Bedrock-GuardrailIdentifier": "caller-guardrail",
        "X-PASS-ANTHROPIC-BETA": "beta-1",
    }
