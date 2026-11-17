"""The 403 AccessDenied shape real Bedrock sends, on every Bedrock surface, and Bedrock Mantle on `/v1/messages`.

Real Bedrock puts the error class only in the `x-amzn-ErrorType` header
(`AccessDeniedException:http://internal.amazon.com/coral/com.amazon.bedrock/`) and sends a body that is
just `{"message": "You don't have access to the model with the specified model ID."}`, or sometimes no
body at all. The proxy ignores that header entirely (no code reads `x-amzn-ErrorType`), and
`_map_bedrock_exception` (`litellm/litellm_core_utils/exception_mapping_utils.py:987`) matches
`AccessDeniedException` only in the text, so these answers reach the caller through the generic status
mapping (`_map_exception_by_status`, `exception_mapping_utils.py:2421`) as a PermissionDeniedError. The
existing `test_dec1_bedrock_errors_retries_wire.py` pins this on non-streamed Converse chat; these tests
pin the same contract on streaming, Invoke, embeddings, `/v1/messages`, an empty body, retries and Mantle.
Retrying a 403 burns quota for nothing and turning it into a 500 hides a missing IAM grant, so a rewrite
of the error path must keep the status.

Bedrock Mantle (`bedrock_mantle/<id>`) serves Claude on `/v1/messages` at `/anthropic/v1/messages`
whatever `/v1`, `/openai/v1` or `/anthropic/v1/messages` suffix the configured base carries
(`build_mantle_native_messages_url`, `litellm/llms/bedrock_mantle/messages/transformation.py:27`), with a
Bearer key when one is configured and SigV4 for service `bedrock` otherwise
(`BedrockMantleAuthMixin.sign_request`, `litellm/llms/bedrock_mantle/common_utils.py:69`).
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from integration.providers.dec1_bedrock3_support import deployment, only_posts, unique_text
from integration.providers.dec1_bedrock_support import (
    ACCESS_KEY,
    CLAUDE_SONNET_45,
    SECRET_KEY,
    TITAN_EMBED_V2,
    aws_error,
    bedrock,
    error_message,
    sigv4_scope,
)

DENIED: Final = "You don't have access to the model with the specified model ID."
MANTLE_CLAUDE: Final = "anthropic.claude-haiku-4-5"
MANTLE_TOKEN: Final = "synthetic-mantle-bearer-dec1"
MANTLE_MESSAGES: Final = "/anthropic/v1/messages"


def _header_only_denied(_: Request) -> Reply:
    """The real AWS shape: class name only in `x-amzn-ErrorType`, message only body."""
    return aws_error(403, "AccessDeniedException", DENIED)


def _empty_denied(_: Request) -> Reply:
    return Reply(
        status=403,
        body=b"",
        headers={"x-amzn-ErrorType": "AccessDeniedException:http://internal.amazon.com/coral/com.amazon.bedrock/"},
    )


# ------------------------------------------------------------------ Header-only 403 on Bedrock runtime surfaces


def test_header_only_access_denied_keeps_the_aws_message_for_the_caller(gateway: Gateway) -> None:
    """Converse answers the header-only 403.

    The caller must get HTTP 403 whose error message contains the AWS sentence verbatim, since that
    sentence is the only hint of which grant is missing.
    """
    with bedrock(gateway, _header_only_denied, model=f"bedrock/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 403, response.text
        assert DENIED in error_message(response), response.text


def test_an_empty_body_access_denied_is_still_a_403(gateway: Gateway) -> None:
    """Converse answers HTTP 403 with only the `x-amzn-ErrorType` header and an empty body.

    With no text to match, the status alone must decide: the caller gets HTTP 403, not a 500 from a
    failed JSON decode. UNVERIFIED: traced to `_map_exception_by_status`.
    """
    with bedrock(gateway, _empty_denied, model=f"bedrock/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 403, response.text


def test_header_only_access_denied_is_not_retried(gateway: Gateway) -> None:
    """The caller asks for `num_retries: 2` and Converse answers the header-only 403 every time.

    A permission error is not retryable, so the peer must see exactly one POST and the caller a 403.
    UNVERIFIED that the router's non-retryable set covers a 403 mapped by status rather than by text.
    """
    with bedrock(gateway, _header_only_denied, model=f"bedrock/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat(num_retries=2)
        assert response.status_code == 403, response.text
        assert len(rig.posts()) == 1


def test_header_only_access_denied_on_stream_open_is_a_403_not_a_stream(gateway: Gateway) -> None:
    """`stream: true` and `converse-stream` answers the header-only 403 before any frame.

    The caller must get HTTP 403 with a JSON error body, never a 200 SSE stream holding an error event.
    """
    with bedrock(gateway, _header_only_denied, model=f"bedrock/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat(stream=True, num_retries=0)
        assert response.status_code == 403, response.text
        assert not response.text.startswith("data:"), response.text


def test_header_only_access_denied_on_the_invoke_route_is_a_403(gateway: Gateway) -> None:
    """A `bedrock/invoke/<claude>` deployment gets the header-only 403 from `/invoke`.

    The Invoke error path must keep the status: HTTP 403 for the caller.
    """
    with bedrock(gateway, _header_only_denied, model=f"bedrock/invoke/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 403, response.text


def test_header_only_access_denied_on_embeddings_is_a_403(gateway: Gateway) -> None:
    """A Titan v2 embedding deployment gets the header-only 403.

    `/v1/embeddings` must answer HTTP 403 with the AWS sentence in the message.
    """
    with bedrock(gateway, _header_only_denied, model=f"bedrock/{TITAN_EMBED_V2}") as rig:
        response: Final = rig.embed(unique_text(), num_retries=0)
        assert response.status_code == 403, response.text
        assert DENIED in response.text


def test_header_only_access_denied_on_messages_is_an_anthropic_permission_error(gateway: Gateway) -> None:
    """`/v1/messages` on a Bedrock Claude deployment gets the header-only 403.

    The caller must get HTTP 403 with an Anthropic error envelope: `type: error` and
    `error.type: permission_error`. UNVERIFIED: the Anthropic error type mapping for a status-mapped 403
    (`litellm/anthropic_interface/exceptions/exception_mapping_utils.py:23`).
    """
    with bedrock(gateway, _header_only_denied, model=f"bedrock/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.messages(messages=[{"role": "user", "content": unique_text()}])
        assert response.status_code == 403, response.text
        body: Final = response.json()
        assert body["type"] == "error", body
        assert body["error"]["type"] == "permission_error", body


def test_header_only_access_denied_writes_a_failure_spend_row_with_status_failure(gateway: Gateway) -> None:
    """A header-only 403 on Converse chat, with a `litellm_call_id` read from the error response header.

    The failure must still be logged: a SpendLogs row with `status: failure` and `spend: 0` for that call.
    UNVERIFIED: that the 403 answer carries `x-litellm-call-id` and that the row is keyed by it.
    """
    from integration._support.client import eventually
    from integration._support.database import read_rows
    from integration.providers.dec1_bedrock_support import SPEND_COLUMNS

    with bedrock(gateway, _header_only_denied, model=f"bedrock/{CLAUDE_SONNET_45}") as rig:
        response: Final = rig.chat(num_retries=0)
        assert response.status_code == 403, response.text
        call_id: Final = response.headers["x-litellm-call-id"]
        rows: Final = eventually(
            lambda: read_rows(SPEND_COLUMNS + "WHERE request_id=%s", (call_id,)),
            lambda values: len(values) == 1,
            seconds=90,
        )
        assert rows[0]["status"] == "failure", rows
        assert float(rows[0]["spend"]) == pytest.approx(0.0), rows  # pyright: ignore[reportArgumentType]


# ------------------------------------------------------------------ Bedrock Mantle on /v1/messages


def _mantle_reply(message_id: str):  # noqa: ANN202
    """A Mantle native Messages peer answering one Anthropic message with id `message_id`."""

    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(body=b"{}")
        return Reply(
            body=json.dumps(
                {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": MANTLE_CLAUDE,
                    "content": [{"type": "text", "text": "mantle answer"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 9, "output_tokens": 3},
                }
            ).encode()
        )

    return respond


_mantle_message: Final = _mantle_reply("msg_bdrk_mantle_dec1")


def _mantle_messages(gateway: Gateway, name: str) -> object:
    return gateway.request(
        "POST",
        "/v1/messages",
        {"model": name, "max_tokens": 32, "messages": [{"role": "user", "content": unique_text()}]},
        headers={"anthropic-version": "2023-06-01"},
    )


@pytest.mark.parametrize("suffix", ["/v1", "/openai/v1", "/anthropic/v1"], ids=["v1", "openai_v1", "anthropic_v1"])
def test_mantle_messages_strips_a_base_suffix_and_posts_to_the_native_messages_path(
    gateway: Gateway, suffix: str
) -> None:
    """A `bedrock_mantle/anthropic.claude-haiku-4-5` deployment whose `api_base` ends in `suffix`.

    `/v1/messages` must reach the peer as exactly one `POST /anthropic/v1/messages` with the Bearer key, and
    the caller gets the Anthropic message back (`content[0].text == "mantle answer"`).
    """
    with wire_server(_mantle_message) as wire, gateway.scenario() as scenario:
        name: Final = deployment(
            scenario, model=f"bedrock_mantle/{MANTLE_CLAUDE}", api_base=wire.url + suffix, api_key=MANTLE_TOKEN
        )
        response: Final = _mantle_messages(gateway, name)
        assert response.status_code == 200, response.text  # pyright: ignore[reportAttributeAccessIssue]
        assert response.json()["content"][0]["text"] == "mantle answer"  # pyright: ignore[reportAttributeAccessIssue]
        posts: Final = only_posts(wire)
        assert [request.target for request in posts] == [MANTLE_MESSAGES]
        assert posts[0].headers["authorization"] == f"Bearer {MANTLE_TOKEN}"


def test_mantle_messages_without_a_bearer_key_is_signed_for_bedrock_in_the_deployment_region(gateway: Gateway) -> None:
    """The Mantle deployment has no `api_key` but static keys and `aws_region_name: us-west-2`.

    With no Bearer token, Mantle signs with SigV4 for service `bedrock`, the scope region taken from
    `aws_region_name` because the peer host is not a `bedrock-mantle.<region>.api.aws` host. Assumes the
    shared proxy has neither `BEDROCK_MANTLE_API_KEY` nor `AWS_BEARER_TOKEN_BEDROCK` in its environment.
    """
    with wire_server(_mantle_message) as wire, gateway.scenario() as scenario:
        name: Final = deployment(
            scenario,
            model=f"bedrock_mantle/{MANTLE_CLAUDE}",
            api_base=wire.url,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            aws_region_name="us-west-2",
        )
        response: Final = _mantle_messages(gateway, name)
        assert response.status_code == 200, response.text  # pyright: ignore[reportAttributeAccessIssue]
        (sent,) = only_posts(wire)
        assert sent.target == MANTLE_MESSAGES
        assert sigv4_scope(sent) == (ACCESS_KEY, "us-west-2", "bedrock")


def test_mantle_messages_header_only_access_denied_is_a_403(gateway: Gateway) -> None:
    """The Mantle native Messages route answers the header-only 403.

    `bedrock_mantle` shares `_map_bedrock_exception`, so the caller must get HTTP 403 and the peer exactly
    one POST.
    """

    def respond(request: Request) -> Reply:
        return _header_only_denied(request) if request.method == "POST" else Reply(body=b"{}")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        name: Final = deployment(
            scenario, model=f"bedrock_mantle/{MANTLE_CLAUDE}", api_base=wire.url, api_key=MANTLE_TOKEN
        )
        response: Final = _mantle_messages(gateway, name)
        assert response.status_code == 403, response.text  # pyright: ignore[reportAttributeAccessIssue]
        assert len(only_posts(wire)) == 1


def test_mantle_messages_spend_row_names_the_mantle_provider(gateway: Gateway) -> None:
    """A successful Mantle `/v1/messages` call (9 in, 3 out tokens).

    The SpendLogs row keyed by the (unique) message id must carry
    `custom_llm_provider: bedrock_mantle` and `call_type: anthropic_messages`, so Mantle spend is not
    folded into Bedrock runtime spend.
    """
    from integration.providers.dec1_bedrock_support import spend_row

    message_id: Final = f"msg_bdrk_{unique_text('mantle')}"
    with wire_server(_mantle_reply(message_id)) as wire, gateway.scenario() as scenario:
        name: Final = deployment(
            scenario, model=f"bedrock_mantle/{MANTLE_CLAUDE}", api_base=wire.url, api_key=MANTLE_TOKEN
        )
        response: Final = _mantle_messages(gateway, name)
        assert response.status_code == 200, response.text  # pyright: ignore[reportAttributeAccessIssue]
        row: Final = spend_row(message_id)
        assert row["custom_llm_provider"] == "bedrock_mantle", row
        assert row["call_type"] == "anthropic_messages", row
        assert len(only_posts(wire)) == 1
