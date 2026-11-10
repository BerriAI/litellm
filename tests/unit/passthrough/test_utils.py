from typing import Final

import httpx
import pytest

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


@pytest.mark.parametrize(
    ("default_query_params", "expected"),
    [
        pytest.param(
            {"api-version": "default", "region": "default"},
            {"api-version": "url", "region": "default", "shared": "request", "tag": ["a", "b"]},
            id="with_defaults",
        ),
        pytest.param(None, {"api-version": "url", "shared": "request", "tag": ["a", "b"]}, id="without_defaults"),
    ],
)
def test_merged_query_parameters_let_the_request_override_the_url_and_the_url_override_defaults(
    default_query_params: dict[str, str | list] | None, expected: dict[str, str | list[str]]
) -> None:
    merged: Final = BasePassthroughUtils.get_merged_query_parameters(
        existing_url=httpx.URL("https://upstream.test/v1?api-version=url&shared=url&tag=a&tag=b"),
        request_query_params={"shared": "request"},
        default_query_params=default_query_params,
    )

    assert merged == expected
