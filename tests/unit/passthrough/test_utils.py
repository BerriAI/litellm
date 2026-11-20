from typing import Final

import httpx
import pytest

from litellm.passthrough.utils import _PASS_THROUGH_PROTECTED_HEADERS, BasePassthroughUtils


def test_pass_through_prefixed_headers_keeps_only_the_x_pass_headers_forwarded_upstream() -> None:
    inbound: Final = {
        "authorization": "Bearer sk-litellm-key",
        "x-litellm-api-key": "sk-litellm-key",
        "content-type": "application/json",
        "x-amzn-bedrock-guardrailidentifier": "native-guardrail",
        "x-passthrough-id": "not-a-pass-header",
        "x-pass-authorization": "Bearer upstream-secret",
        "X-Pass-X-Api-Key": "upstream-api-key",
        "x-pass-X-Amz-Security-Token": "forged-session-token",
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
    "header_name",
    [
        *(f"x-pass-{name}" for name in sorted(_PASS_THROUGH_PROTECTED_HEADERS)),
        "X-Pass-Authorization",
        "x-pass-X-Amz-Date",
        "x-pass-x-amz-security-token",
        "x-pass-X-Amzn-Bedrock-Trace",
        "x-pass-anthropic-beta",
        "x-pass-cookie",
        "x-passthrough-id",
        "authorization",
    ],
)
def test_pass_through_prefixed_headers_select_exactly_the_headers_forwarded_upstream(header_name: str) -> None:
    inbound: Final = {header_name: "caller-value"}

    selected: Final = BasePassthroughUtils.pass_through_prefixed_headers(inbound)
    forwarded: Final = BasePassthroughUtils.forward_headers_from_request(
        request_headers=dict(inbound), headers={}, forward_headers=False
    )

    assert {name[len("x-pass-") :].lower(): value for name, value in selected.items()} == forwarded


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
