"""Live e2e for LLM-translation passthrough endpoints.

Each test sends a NATIVE provider request through the proxy's passthrough route
and verifies the proxy still logged a costed SpendLogs row
(call_type="pass_through_endpoint"), correlated by the x-litellm-call-id header.

Covered: gemini ("gemini-2.5-flash") + anthropic ("claude-haiku-4-5"), streaming +
non-streaming, plus native tool calls. See LLM_TRANSLATION_COVERAGE_MATRIX.md.

A passthrough call returning non-2xx fails hard (never a skip); once it returns
2xx, a missing or zero-cost SpendLogs row fails too.
"""

import pytest

from e2e_config import CHEAP_OPENAI_MODEL, unique_marker
from e2e_http import StreamingResponse, require_successful_call, unwrap
from lifecycle import ResourceManager
from models import KeyGenerateBody, SpendLogRow
from passthrough_client import (
    AnthropicTool,
    GeminiFunctionDeclaration,
    GeminiTool,
    JsonSchema,
    JsonSchemaProperty,
    PassthroughClient,
    completed_responses_object,
)

EMBEDDING_MODEL = "text-embedding-3-small"
REALTIME_MODEL = "gpt-realtime-2"

pytestmark = pytest.mark.e2e


def _fetch_cost_breakdown(client: PassthroughClient, result: StreamingResponse) -> SpendLogRow:
    """The passthrough call's logged row, polled until it carries a cost.

    Asserts (not skips) that a 2xx passthrough call produced a costed row - the
    whole point of passthrough spend tracking.
    """
    assert result.call_id, "passthrough response had no x-litellm-call-id header"
    rows = client.proxy.poll_logs_for_request_id(
        result.call_id,
        predicate=lambda rs: (rs[0].spend or 0) > 0,
    )
    assert rows, f"no SpendLogs row for passthrough call_id {result.call_id}"
    row = rows[0]
    assert row.call_type == "pass_through_endpoint"
    assert (row.spend or 0) > 0, f"passthrough call was not costed: {row}"
    assert row.status == "success"
    return row


# ---- Gemini passthrough ------------------------------------------------


def test_gemini_passthrough_nonstreaming_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    tag = f"e2e-passthrough-{unique_marker()}"
    result = client.gemini_generate(
        scoped_key, "gemini-2.5-flash", "Say hello in one word", tags=[tag, "gemini"]
    )
    require_successful_call(result)

    row = _fetch_cost_breakdown(client, result)
    assert row.custom_llm_provider == "gemini"
    assert "gemini" in (row.model or "")
    assert tag in (row.request_tags or []), f"tags not logged: {row.request_tags}"


@pytest.mark.skip(reason="stage red: product gap, native passthrough returns no x-litellm-response-cost or x-ratelimit-* headers")
def test_gemini_passthrough_returns_the_same_header_contract_as_the_managed_route(
    client: PassthroughClient, scoped_key: str
) -> None:
    """Native /gemini/ passthrough must return the same operational headers as
    /chat/completions: x-litellm-response-cost so the call reconciles against
    spend, and x-ratelimit-* so a client can pace itself. It returns neither
    today, which makes native traffic invisible to the same tooling.
    """
    result = client.gemini_generate(
        scoped_key, "gemini-2.5-flash", f"Say hello in one word. {unique_marker()}"
    )
    require_successful_call(result)

    assert result.call_id, "passthrough must stamp x-litellm-call-id"
    assert result.response_cost is not None, (
        "passthrough generateContent returned no x-litellm-response-cost header, so a "
        "native call cannot be reconciled against spend the way /chat/completions can"
    )
    assert result.response_cost > 0, (
        f"x-litellm-response-cost must be a real cost, got {result.response_cost}"
    )

    pacing = tuple(name for name in result.headers if name.startswith("x-ratelimit-"))
    assert pacing, (
        "passthrough generateContent returned no x-ratelimit-* headers, so a client "
        f"cannot pace itself; headers present were {sorted(result.headers)}"
    )


def test_gemini_passthrough_streaming_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.gemini_stream(scoped_key, "gemini-2.5-flash", "Count to five")
    require_successful_call(result)
    assert result.chunks > 0, "streaming passthrough produced no events"

    row = _fetch_cost_breakdown(client, result)
    assert row.custom_llm_provider == "gemini"


def test_gemini_passthrough_tool_call_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.gemini_generate(
        scoped_key,
        "gemini-2.5-flash",
        "What is the weather in Paris? Use the get_weather tool.",
        tools=[
            GeminiTool(
                function_declarations=[
                    GeminiFunctionDeclaration(
                        name="get_weather",
                        description="Get the weather for a city",
                        parameters=JsonSchema(
                            type="object",
                            properties={"city": JsonSchemaProperty(type="string")},
                            required=["city"],
                        ),
                    )
                ]
            )
        ],
    )
    require_successful_call(result)
    assert "functionCall" in result.body, "gemini did not emit a tool call"

    row = _fetch_cost_breakdown(client, result)
    assert row.custom_llm_provider == "gemini"


# ---- Anthropic passthrough ---------------------------------------------


def test_anthropic_passthrough_nonstreaming_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.anthropic_message(scoped_key, "claude-haiku-4-5", "Say hello")
    require_successful_call(result)

    row = _fetch_cost_breakdown(client, result)
    assert row.custom_llm_provider == "anthropic"
    assert "claude" in (row.model or "")


def test_anthropic_passthrough_streaming_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.anthropic_message(
        scoped_key, "claude-haiku-4-5", "Count to five", stream=True
    )
    require_successful_call(result)
    assert result.chunks > 0, "streaming passthrough produced no events"

    row = _fetch_cost_breakdown(client, result)
    assert row.custom_llm_provider == "anthropic"


def test_anthropic_passthrough_tool_call_logs_cost(
    client: PassthroughClient, scoped_key: str
) -> None:
    result = client.anthropic_message(
        scoped_key,
        "claude-haiku-4-5",
        "What is the weather in Paris? Use the get_weather tool.",
        tools=[
            AnthropicTool(
                name="get_weather",
                description="Get the weather for a city",
                input_schema=JsonSchema(
                    type="object",
                    properties={"city": JsonSchemaProperty(type="string")},
                    required=["city"],
                ),
            )
        ],
    )
    require_successful_call(result)
    assert "tool_use" in result.body, "anthropic did not emit a tool call"

    row = _fetch_cost_breakdown(client, result)
    assert row.custom_llm_provider == "anthropic"


class TestPassthroughModelAllowlist:
    """A passthrough route must honor the calling key's model allow-list.

    The customer fronts native provider calls through the proxy with custom auth,
    so a key scoped to one model must not reach a different model just because the
    request goes through the passthrough route rather than /chat/completions.
    """

    @pytest.mark.covers("other.auth.passthrough.model_allowlist_enforced")
    def test_passthrough_denies_model_outside_key_allowlist(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        key = client.proxy.generate_key(KeyGenerateBody(models=["gemini-2.5-flash"]))
        resources.defer(lambda: client.proxy.delete_key(key))

        result = client.anthropic_message(key, "claude-haiku-4-5", f"say hi {unique_marker()}")
        assert result.status_code == 403, (
            "a key restricted to gemini-2.5-flash must be denied a claude passthrough call, "
            f"got {result.status_code}: {result.body[:300]}"
        )


class TestOpenAIPassthroughPrefix:
    """The dedicated `/openai_passthrough` prefix must reach OpenAI, not be
    swallowed by the provider-scoped `/{provider}/v1/...` routes.

    The customer fronts OpenAI's own file and batch APIs through this prefix
    precisely to opt out of the gateway's managed-file handling. `/v1/files` and
    `/v1/batches` also answer `/{provider}/v1/files` and `/{provider}/v1/batches`,
    so `openai_passthrough` used to bind as a provider name and the request died
    inside the gateway with a provider-lookup error, never reaching OpenAI.
    """

    @pytest.mark.covers("llm.files.openai.passthrough.nonstream.works")
    def test_passthrough_prefix_uploads_a_file_to_openai(
        self, client: PassthroughClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """Pins GitHub issue #36086: a file upload through the dedicated prefix
        reaches OpenAI's file API instead of 500ing on a provider-name lookup."""
        content = f'{{"marker":"{unique_marker()}"}}\n'.encode()
        uploaded = unwrap(
            client.openai_passthrough_upload_file(
                scoped_key, content=content, filename="e2e-passthrough-batch.jsonl"
            )
        )
        resources.defer(
            lambda: client.openai_passthrough_delete_file(scoped_key, uploaded.id)
        )

        assert uploaded.object == "file", (
            f"/openai_passthrough/v1/files did not relay OpenAI's file object: {uploaded}"
        )
        assert uploaded.purpose == "batch"
        assert uploaded.bytes == len(content)

    @pytest.mark.covers("llm.batches.openai.passthrough.nonstream.works")
    def test_passthrough_prefix_lists_batches_from_openai(
        self, client: PassthroughClient, scoped_key: str
    ) -> None:
        """Pins GitHub issue #36086 on the batches route: the dedicated prefix
        relays OpenAI's own batch page instead of dying on the provider lookup."""
        listed = unwrap(client.openai_passthrough_list_batches(scoped_key))

        assert listed.object == "list", (
            f"/openai_passthrough/v1/batches did not relay OpenAI's batch page: {listed}"
        )


class TestOpenAIPassthroughSpend:
    """A call relayed to OpenAI's own endpoints must still be costed.

    The customer routes native OpenAI traffic through `/openai_passthrough` and
    budgets against it, so a call that returns 200 while logging no spend is money
    the gateway never sees and a budget that never trips. Streamed Responses calls
    and embeddings each used to land exactly that way, on separate code paths.
    """

    @pytest.mark.covers("llm.responses.openai.passthrough.stream.cost_logged")
    def test_streamed_responses_call_logs_its_cost(
        self, client: PassthroughClient, scoped_key: str
    ) -> None:
        """Pins GitHub issue #36523: a streamed passthrough Responses call is billed
        under the provider id the caller was served, never a $0 row under a random
        id."""
        result = client.openai_passthrough_responses(
            scoped_key,
            CHEAP_OPENAI_MODEL,
            f"Say hi in one word. {unique_marker()}",
            stream=True,
        )
        require_successful_call(result)
        assert result.chunks > 0, "streamed responses passthrough produced no events"

        completed = completed_responses_object(result)
        assert completed is not None, (
            f"the stream never delivered a response.completed frame, so there is no "
            f"provider id to reconcile against: last events {result.stream_events[-3:]}"
        )
        assert completed.usage is not None, (
            f"the completed response carried no usage to price from: {completed}"
        )

        rows = client.proxy.poll_logs_for_request_id(
            completed.id, predicate=lambda rows: (rows[0].spend or 0) > 0
        )
        assert rows, (
            f"no spend row for the response the customer was served ({completed.id}); "
            "a streamed passthrough call OpenAI bills them for is invisible to the "
            "gateway's own spend and budgets"
        )
        row = rows[0]
        assert (row.spend or 0) > 0, f"streamed responses passthrough was not costed: {row}"
        assert row.prompt_tokens == completed.usage.input_tokens, (
            f"logged {row.prompt_tokens} prompt tokens, the response the customer read "
            f"reported {completed.usage.input_tokens}"
        )
        assert row.completion_tokens == completed.usage.output_tokens, (
            f"logged {row.completion_tokens} completion tokens, the response the customer "
            f"read reported {completed.usage.output_tokens}"
        )

    @pytest.mark.covers("llm.embeddings.openai.passthrough.nonstream.cost_logged")
    def test_embeddings_call_logs_its_cost(
        self, client: PassthroughClient, scoped_key: str
    ) -> None:
        """Pins GitHub issue #36646: a passthrough embeddings call writes a priced
        spend row instead of no row at all."""
        result = client.openai_passthrough_embed(
            scoped_key, EMBEDDING_MODEL, f"cost this sentence {unique_marker()}"
        )
        require_successful_call(result)
        assert result.call_id, "embeddings passthrough returned no x-litellm-call-id"

        rows = client.proxy.poll_logs_for_request_id(
            result.call_id, predicate=lambda rows: (rows[0].spend or 0) > 0
        )
        assert rows, (
            f"no spend row for embeddings call {result.call_id}; the customer is billed "
            "by OpenAI for tokens the gateway never counted against their budget"
        )
        row = rows[0]
        assert (row.spend or 0) > 0, f"embeddings passthrough was not costed: {row}"
        assert (row.prompt_tokens or 0) > 0, (
            f"the embeddings row logged no prompt tokens, so whatever cost it carries "
            f"was not computed from the real usage: {row}"
        )


class TestOpenAIPassthroughWebsocket:
    """The OpenAI passthrough prefixes must answer a websocket upgrade, not only a POST.

    The customer points realtime and responses.connect clients at the same prefixes
    their HTTP traffic already uses. Only HTTP routes were registered under those
    prefixes, so every upgrade was refused before a socket existed and those clients
    could not reach the gateway at all. A refused upgrade is an HTTP response, not a
    close frame, which is why these assert on the handshake rather than a close code.
    """

    @pytest.mark.covers("llm.realtime.openai.passthrough.stream.works")
    def test_realtime_upgrade_reaches_openai_through_the_passthrough_prefix(
        self, client: PassthroughClient, scoped_key: str
    ) -> None:
        """Pins GitHub issue #36088: /openai_passthrough/v1/realtime accepts the
        upgrade and relays OpenAI's own session, instead of rejecting it with a 403."""
        handshake = client.openai_passthrough_websocket(
            scoped_key, "/openai_passthrough/v1/realtime", model=REALTIME_MODEL
        )

        assert handshake.rejected_status is None, (
            f"/openai_passthrough/v1/realtime refused the websocket upgrade with HTTP "
            f"{handshake.rejected_status}, so a realtime client cannot connect through "
            "the gateway at all"
        )
        assert handshake.first_event_type == "session.created", (
            "the accepted socket never carried OpenAI's opening session event, so the "
            f"upgrade was not relayed upstream; the first frame was "
            f"{handshake.first_event_type}"
        )

    @pytest.mark.covers("llm.responses.openai.passthrough_websocket.stream.works")
    def test_responses_upgrade_is_accepted_on_the_openai_prefix(
        self, client: PassthroughClient, scoped_key: str
    ) -> None:
        """Pins GitHub issue #36088 on the second prefix: /openai/v1/responses upgrades
        as well. A responses.connect socket waits for the client to speak first, so the
        accepted handshake is the whole signal here."""
        handshake = client.openai_passthrough_websocket(
            scoped_key, "/openai/v1/responses", first_event_timeout=2.0
        )

        assert handshake.rejected_status is None, (
            f"/openai/v1/responses refused the websocket upgrade with HTTP "
            f"{handshake.rejected_status}; the prefix relays this route over HTTP but "
            "drops a responses.connect client before the socket opens"
        )
