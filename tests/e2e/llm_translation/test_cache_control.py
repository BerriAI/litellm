"""Live e2e: provider-specific /chat/completions features take real effect.

Each case asserts the feature actually happened, not just a 200. Coverage matrix
(register-on-demand deployments, deleted on teardown):

- Bedrock (anthropic claude-haiku-4-5): prompt caching. A large cacheable prefix
  marked with ``cache_control`` is sent twice; the second call must report
  cache-read usage tokens > 0. service_tier is out of scope for Bedrock; AWS
  Bedrock does not expose an OpenAI-style request service tier, so that cell is
  intentionally not covered here.
- Vertex (gemini-2.5-flash): explicit context caching via ``cache_control``
  with a 5-minute ttl. litellm builds the Vertex cache before the generate
  call, so a never-seen prefix must come back cached on its very first call
  (Gemini's implicit caching cannot hit a cold prefix), the cached count must
  cover the marked block, and the spend row must be billed below the uncached
  price of the cached tokens. Vertex's cache create is nondeterministic:
  identical bodies come back 200 or with the minimum-token 400 ("The cached
  content is of 1 tokens"), in failure bursts of 45 seconds and more, so up to
  eight never-seen prefixes are tried with a pause after each rejection. The
  billing check prices the cached tokens rather than prompt_tokens, which
  Vertex reports inclusive of the cached prefix on some calls and exclusive
  of it on others.
- Anthropic (claude-haiku-4-5, direct): the same ``cache_control`` prefix over
  the OpenAI-compatible route; the second call must report cache-read tokens > 0.
- OpenAI (gpt-5.6): automatic prompt caching needs no request marker, so the
  cacheable prefix goes out as a plain system string with a ``prompt_cache_key``
  and the second call must report ``prompt_tokens_details.cached_tokens`` > 0.

- OpenAI (gpt-5.6) in explicit cache mode: OpenAI caches nothing unless a
  breakpoint reaches it, so a ``cache_control`` mark on a system part, a string
  system message, a user part or a tool result must write the cache on the
  first call and read it on the second.

- Claude (claude-haiku-5-5 on anthropic, Bedrock Converse and Bedrock Invoke,
  claude-sonnet-5-5 on Vertex, which serves no haiku-5-5): OpenAI's ``prompt_cache_breakpoint`` on a system part, a string
  system message, a user part, a tool result, a tool call or a tool definition,
  or beside a ``cache_control`` on the same system part, must write the cache
  on the first call and read it on the second. On
  anthropic and Converse, a 30m breakpoint and five breakpoints under
  drop_params must still cache. On anthropic, ``prompt_cache_options`` with
  ``"mode": "implicit"`` must cache with no breakpoint at all.

service_tier lives in test_provider_features_e2e.py.

The provider-native cache_control request shape is not expressible with the
shared ``ChatBody`` (whose content is a plain string), so the cacheable body is
built from the typed content blocks shared in ``models.py``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Final, Literal, TypeAlias

import pytest
from _pytest.mark.structures import ParameterSet
from e2e_config import unique_marker
from e2e_http import Result, UnknownApiError, unwrap
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import (
    CacheControl,
    ChatAssistantTurn,
    ChatBody,
    ChatMessage,
    ChatResponse,
    ChatTool,
    ChatToolFunction,
    ChatToolResultTurn,
    ChatTurn,
    ContentPart,
    LiteLLMParamsBody,
    PromptCacheBreakpoint,
    PromptCacheOptions,
    RichMessage,
    TextBlock,
    TextContentPart,
    ToolCall,
    ToolCallFunction,
    Usage,
)
from passthrough_client import PassthroughClient
from pydantic import BaseModel, ConfigDict

pytestmark = [pytest.mark.e2e, pytest.mark.provider_live]

BEDROCK_MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
VERTEX_MODEL = "vertex_ai/gemini-2.5-flash"
ANTHROPIC_MODEL = "anthropic/claude-haiku-4-5-20251001"
OPENAI_MODEL = "openai/gpt-5.6"
VERTEX_CACHE_TTL: Final = "300s"
VERTEX_COLD_CALL_ATTEMPTS: Final = 8
VERTEX_COLD_CALL_PAUSE_SECONDS: Final = 15.0
VERTEX_MINIMUM_CACHED_TOKENS: Final = 1024
CACHED_SHARE_OF_PROMPT: Final = 0.9
VERTEX_CACHE_REJECTION_MARKER: Final = "minimum token count to start explicit caching"


class CacheChatBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    messages: list[RichMessage]
    max_tokens: int = 64
    cache: dict[str, bool] = {"no-cache": True}


def _cacheable_prefix() -> str:
    """A prefix long enough to clear provider minimum cacheable sizes (Haiku is
    2048 tokens), unique per run so the first call writes and the second reads."""
    marker = unique_marker()
    body = " ".join(f"Cacheable reference paragraph {index} for run {marker}." for index in range(600))
    return f"{body}\nEnd of reference material {marker}."


def _cached_read_tokens(usage: Usage | None) -> int:
    """Cache-read tokens however the provider reports them: Anthropic-style
    ``cache_read_input_tokens`` or OpenAI-style ``prompt_tokens_details.cached_tokens``."""
    if usage is None:
        return 0
    if usage.cache_read_input_tokens:
        return usage.cache_read_input_tokens
    if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
        return usage.prompt_tokens_details.cached_tokens
    return 0


def _cache_chat(
    client: PassthroughClient, key: str, model: str, prefix: str, ttl: str | None = None
) -> Result[ChatResponse]:
    body = CacheChatBody(
        model=model,
        messages=[
            RichMessage(
                role="system",
                content=[TextBlock(text=prefix, cache_control=CacheControl(ttl=ttl))],
            ),
            RichMessage(role="user", content=[TextBlock(text="Reply with one word.")]),
        ],
    )
    return client.proxy.transport.post(
        "/chat/completions",
        headers=client.proxy.transport.bearer(key),
        json=body,
        response_type=ChatResponse,
    )


def _plain_cache_chat(
    client: PassthroughClient, key: str, model: str, prefix: str, cache_key: str
) -> Result[ChatResponse]:
    """The same cacheable prefix as a plain system string, for providers that cache
    automatically and take no per-block marker (OpenAI)."""
    return client.proxy.chat(
        key,
        ChatBody(
            model=model,
            messages=[
                ChatMessage(role="system", content=prefix),
                ChatMessage(role="user", content="Reply with one word."),
            ],
            max_tokens=64,
            prompt_cache_key=cache_key,
        ),
    )


def _assert_cache_read_on_second_call(model: str, send: Callable[[str], Result[ChatResponse]]) -> None:
    prefix = _cacheable_prefix()

    first = unwrap(send(prefix))
    assert first.choices, f"{model}: first cache-priming call returned no choices: {first}"

    deadline = time.monotonic() + 30.0
    while True:
        second = unwrap(send(prefix))
        read_tokens = _cached_read_tokens(second.usage)
        if read_tokens > 0 or time.monotonic() >= deadline:
            break
        time.sleep(3.0)

    assert read_tokens > 0, (
        f"{model}: second identical call reported no cache-read tokens "
        f"({second.usage}); prompt caching did not take effect"
    )


def _cache_write_tokens(usage: Usage | None) -> int:
    if usage is None or usage.prompt_tokens_details is None:
        return 0
    return usage.prompt_tokens_details.cache_write_tokens or 0


def _assert_openai_writes_then_reads_cache(client: PassthroughClient, key: str, body: ChatBody) -> None:
    first: Final = unwrap(client.proxy.chat(key, body))
    assert _cache_write_tokens(first.usage) > 0, (
        f"{body.model}: the first call on a never-seen prefix wrote no cache ({first.usage}), "
        "so the cache_control mark never reached OpenAI as a breakpoint"
    )

    deadline: Final = time.monotonic() + 30.0
    while True:
        second = unwrap(client.proxy.chat(key, body))
        read_tokens = _cached_read_tokens(second.usage)
        if read_tokens > 0 or time.monotonic() >= deadline:
            break
        time.sleep(3.0)

    assert read_tokens > 0, f"{body.model}: the identical second call read no cache ({second.usage})"


def _cold_cache_call(send: Callable[[str], Result[ChatResponse]]) -> ChatResponse | None:
    result: Final = send(_cacheable_prefix())
    match result:
        case UnknownApiError(status_code=400, body=body) if VERTEX_CACHE_REJECTION_MARKER in body:
            return None
        case _:
            return unwrap(result)


def _first_engaged_cold_call(send: Callable[[str], Result[ChatResponse]]) -> ChatResponse | None:
    for attempt in range(1, VERTEX_COLD_CALL_ATTEMPTS + 1):
        candidate = _cold_cache_call(send)
        if candidate is not None and _cached_read_tokens(candidate.usage) >= VERTEX_MINIMUM_CACHED_TOKENS:
            return candidate
        if attempt < VERTEX_COLD_CALL_ATTEMPTS:
            print(
                f"cache_control: vertex did not engage the cache on cold attempt {attempt}/{VERTEX_COLD_CALL_ATTEMPTS}; "
                f"pausing {VERTEX_COLD_CALL_PAUSE_SECONDS}s before the next never-seen prefix",
                flush=True,
            )
            time.sleep(VERTEX_COLD_CALL_PAUSE_SECONDS)
    return None


def _first_cold_call_reads_cache(model: str, send: Callable[[str], Result[ChatResponse]]) -> ChatResponse:
    completion: Final = _first_engaged_cold_call(send)
    assert completion is not None, (
        f"{model}: {VERTEX_COLD_CALL_ATTEMPTS} never-seen prompts marked with cache_control, spread over "
        f"{VERTEX_COLD_CALL_PAUSE_SECONDS * (VERTEX_COLD_CALL_ATTEMPTS - 1):.0f}s, were each either rejected by "
        f"Vertex's minimum-token check or served with fewer than {VERTEX_MINIMUM_CACHED_TOKENS} cached tokens on "
        "their first call; explicit context caching did not engage"
    )
    assert completion.choices, f"{model}: cached call returned no choices: {completion}"
    usage: Final = completion.usage
    cached: Final = _cached_read_tokens(usage)
    assert usage and usage.prompt_tokens, f"{model}: cached completion carried no prompt_tokens: {usage}"
    assert cached >= CACHED_SHARE_OF_PROMPT * usage.prompt_tokens, (
        f"{model}: only {cached} of {usage.prompt_tokens} prompt tokens were served from the cache; the "
        "cache_control block was not cached whole"
    )
    return completion


def _input_rate(client: PassthroughClient, model: str) -> float:
    entry: Final = next((row for row in client.proxy.model_info() if row.model_name == model), None)
    assert entry and entry.model_info.input_cost_per_token, f"/model/info resolved no input rate for {model}"
    return entry.model_info.input_cost_per_token


def _assert_billed_below_uncached_prompt(client: PassthroughClient, model: str, completion: ChatResponse) -> None:
    assert completion.id, f"{model}: cached completion carried no id to find its spend row by"
    usage: Final = completion.usage
    assert usage and usage.prompt_tokens, f"{model}: cached completion carried no prompt_tokens: {usage}"
    rows: Final = client.proxy.poll_logs_for_request_id(completion.id, predicate=lambda rs: (rs[0].spend or 0) > 0)
    assert rows, f"{model}: no costed /spend/logs row for request {completion.id}"
    row: Final = rows[0]
    assert row.prompt_tokens == usage.prompt_tokens, (
        f"{model}: spend row prompt_tokens {row.prompt_tokens} != response prompt_tokens {usage.prompt_tokens}"
    )
    cached: Final = _cached_read_tokens(usage)
    uncached_read_cost: Final = cached * _input_rate(client, model)
    assert row.spend is not None and row.spend < uncached_read_cost, (
        f"{model}: spend {row.spend} is not below the uncached price of the {cached} tokens read from the cache "
        f"({uncached_read_cost}); cache-read pricing was not applied"
    )


class TestCacheControl:
    @pytest.mark.covers(
        "llm.chat_completions.bedrock_converse.prompt_cache_5m.nonstream.works",
        exercised_on=[],
    )
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_MODEL,),
            capabilities=(Capability.PROMPT_CACHING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_bedrock_prompt_caching_reads_cache(self, client: PassthroughClient, resources: ResourceManager) -> None:
        model = f"e2e-bedrock-cache-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=BEDROCK_MODEL, aws_region_name="us-east-1"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()
        _assert_cache_read_on_second_call(model, lambda prefix: _cache_chat(client, key, model, prefix))

    @pytest.mark.covers(
        "llm.chat_completions.vertex.prompt_cache_5m.nonstream.works",
        exercised_on=[],
    )
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.VERTEX_AI,),
            models=(VERTEX_MODEL,),
            capabilities=(Capability.PROMPT_CACHING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_vertex_prompt_caching_reads_cache(self, client: PassthroughClient, resources: ResourceManager) -> None:
        model = f"e2e-vertex-cache-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=VERTEX_MODEL,
                vertex_project=os.environ.get("VERTEXAI_PROJECT"),
                vertex_location="us-central1",
                vertex_credentials=os.environ.get("VERTEXAI_CREDENTIALS"),
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()
        completion: Final = _first_cold_call_reads_cache(
            model, lambda prefix: _cache_chat(client, key, model, prefix, ttl=VERTEX_CACHE_TTL)
        )
        _assert_billed_below_uncached_prompt(client, model, completion)

    @pytest.mark.covers(
        "llm.chat_completions.anthropic.prompt_cache_5m.nonstream.works",
        exercised_on=[],
    )
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_MODEL,),
            capabilities=(Capability.PROMPT_CACHING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_anthropic_prompt_caching_reads_cache(self, client: PassthroughClient, resources: ResourceManager) -> None:
        model = f"e2e-anthropic-cache-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=ANTHROPIC_MODEL, api_key="os.environ/ANTHROPIC_API_KEY"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()
        _assert_cache_read_on_second_call(model, lambda prefix: _cache_chat(client, key, model, prefix))

    @pytest.mark.covers("llm.chat_completions.anthropic.prompt_cache_5m.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_MODEL,),
            capabilities=(Capability.PROMPT_CACHING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_anthropic_prompt_caching_reads_cache_with_version_header(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model: Final = f"e2e-anthropic-cache-version-{unique_marker()}"
        model_id: Final = client.proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=ANTHROPIC_MODEL,
                api_key="os.environ/ANTHROPIC_API_KEY",
                extra_headers={"anthropic-version": "2023-06-01"},
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key: Final = resources.key()
        _assert_cache_read_on_second_call(model, lambda prefix: _cache_chat(client, key, model, prefix))

    @pytest.mark.covers("llm.chat_completions.anthropic.prompt_cache_5m.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_MODEL,),
            capabilities=(Capability.PROMPT_CACHING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_anthropic_prompt_caching_reports_creation_tokens(
        self, client: PassthroughClient, resources: ResourceManager
    ) -> None:
        model: Final = f"e2e-anthropic-cache-create-{unique_marker()}"
        model_id: Final = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=ANTHROPIC_MODEL, api_key="os.environ/ANTHROPIC_API_KEY"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key: Final = resources.key()
        body: Final = CacheChatBody(
            model=model,
            messages=[
                RichMessage(
                    role="system",
                    content=[TextBlock(text=_cacheable_prefix(), cache_control=CacheControl(ttl="1h"))],
                ),
                RichMessage(
                    role="user",
                    content=[TextBlock(text="Reply with one word.", cache_control=CacheControl(ttl="5m"))],
                ),
            ],
        )
        response: Final = unwrap(
            client.proxy.transport.post(
                "/chat/completions",
                headers=client.proxy.transport.bearer(key),
                json=body,
                response_type=ChatResponse,
            )
        )
        assert response.usage and response.usage.cache_creation_input_tokens, (
            f"{model}: a never-seen prefix marked 1h then 5m reported no cache-creation tokens ({response.usage})"
        )

    @pytest.mark.covers(
        "llm.chat_completions.openai.prompt_cache_5m.nonstream.works",
        exercised_on=[],
    )
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.OPENAI,),
            models=(OPENAI_MODEL,),
            capabilities=(Capability.PROMPT_CACHING,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_openai_prompt_caching_reads_cache(self, client: PassthroughClient, resources: ResourceManager) -> None:
        model = f"e2e-openai-cache-{unique_marker()}"
        model_id = client.proxy.create_model(
            model,
            LiteLLMParamsBody(model=OPENAI_MODEL, api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        key = resources.key()
        cache_key = f"e2e-openai-cache-{unique_marker()}"
        _assert_cache_read_on_second_call(
            model, lambda prefix: _plain_cache_chat(client, key, model, prefix, cache_key)
        )


@pytest.mark.covers("llm.chat_completions.openai.prompt_cache_5m.nonstream.works")
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.CHAT_COMPLETIONS,
        providers=(Provider.OPENAI,),
        models=(OPENAI_MODEL,),
        capabilities=(Capability.PROMPT_CACHING,),
        mode=Mode.NONSTREAM,
    )
)
class TestCacheControlOnOpenAIExplicitCaching:
    @pytest.fixture
    def model(self, client: PassthroughClient, resources: ResourceManager) -> str:
        name: Final = f"e2e-openai-explicit-cache-{unique_marker()}"
        model_id: Final = client.proxy.create_model(
            name,
            LiteLLMParamsBody(
                model=OPENAI_MODEL,
                api_key="os.environ/OPENAI_API_KEY",
                prompt_cache_options=PromptCacheOptions(mode="explicit"),
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        return name

    def test_cache_control_on_a_system_text_part(self, client: PassthroughClient, scoped_key: str, model: str) -> None:
        body: Final = ChatBody(
            model=model,
            max_tokens=64,
            messages=[
                ChatMessage(
                    role="system",
                    content=[TextContentPart(text=_cacheable_prefix(), cache_control=CacheControl())],
                ),
                ChatMessage(role="user", content="Reply with one word."),
            ],
        )
        _assert_openai_writes_then_reads_cache(client, scoped_key, body)

    def test_cache_control_on_a_string_system_message(
        self, client: PassthroughClient, scoped_key: str, model: str
    ) -> None:
        body: Final = ChatBody(
            model=model,
            max_tokens=64,
            messages=[
                ChatMessage(role="system", content=_cacheable_prefix(), cache_control=CacheControl()),
                ChatMessage(role="user", content="Reply with one word."),
            ],
        )
        _assert_openai_writes_then_reads_cache(client, scoped_key, body)

    def test_cache_control_on_a_user_text_part(self, client: PassthroughClient, scoped_key: str, model: str) -> None:
        body: Final = ChatBody(
            model=model,
            max_tokens=64,
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        TextContentPart(text=_cacheable_prefix(), cache_control=CacheControl()),
                        TextContentPart(text="Reply with one word."),
                    ],
                ),
            ],
        )
        _assert_openai_writes_then_reads_cache(client, scoped_key, body)

    def test_cache_control_on_a_tool_result(self, client: PassthroughClient, scoped_key: str, model: str) -> None:
        body: Final = ChatBody(
            model=model,
            max_tokens=64,
            reasoning_effort="none",
            tools=[ChatTool(function=ChatToolFunction(name="fetch_report", parameters={"type": "object"}))],
            messages=[
                ChatMessage(role="user", content="Fetch the report and summarize it in one sentence."),
                ChatAssistantTurn(
                    tool_calls=[
                        ToolCall(
                            id="call_1", type="function", function=ToolCallFunction(name="fetch_report", arguments="{}")
                        )
                    ]
                ),
                ChatToolResultTurn(
                    tool_call_id="call_1",
                    content=[TextContentPart(text=_cacheable_prefix(), cache_control=CacheControl())],
                ),
            ],
        )
        _assert_openai_writes_then_reads_cache(client, scoped_key, body)


CLAUDE_ANTHROPIC_MODEL: Final = "anthropic/claude-haiku-5-5"
CLAUDE_BEDROCK_CONVERSE_MODEL: Final = "bedrock/converse/us.anthropic.claude-haiku-5-5"
CLAUDE_BEDROCK_INVOKE_MODEL: Final = "bedrock/invoke/us.anthropic.claude-haiku-5-5"
CLAUDE_VERTEX_MODEL: Final = "vertex_ai/claude-sonnet-5-5"
ClaudeRoute: TypeAlias = Literal["anthropic", "bedrock_converse", "bedrock_invoke", "vertex"]
BreakpointSite: TypeAlias = Literal[
    "system_part", "user_part", "string_message", "tool_result_part", "tool_call", "tool_definition"
]
_LOOKUP_TOOL: Final = ChatToolFunction(
    name="lookup_reference", description="Look up the reference material.", parameters={"type": "object"}
)


def _claude_route_param(route: ClaudeRoute) -> ParameterSet:
    return pytest.param(
        route, id=route, marks=pytest.mark.covers(f"llm.chat_completions.{route}.prompt_cache_5m.nonstream.works")
    )


def _claude_deployment(route: ClaudeRoute, *, drop_params: bool | None = None) -> LiteLLMParamsBody:
    match route:
        case "anthropic":
            return LiteLLMParamsBody(
                model=CLAUDE_ANTHROPIC_MODEL, api_key="os.environ/ANTHROPIC_API_KEY", drop_params=drop_params
            )
        case "bedrock_converse":
            return LiteLLMParamsBody(
                model=CLAUDE_BEDROCK_CONVERSE_MODEL, aws_region_name="us-east-1", drop_params=drop_params
            )
        case "bedrock_invoke":
            return LiteLLMParamsBody(
                model=CLAUDE_BEDROCK_INVOKE_MODEL, aws_region_name="us-east-1", drop_params=drop_params
            )
        case "vertex":
            return LiteLLMParamsBody(
                model=CLAUDE_VERTEX_MODEL,
                vertex_project="os.environ/VERTEXAI_PROJECT",
                vertex_location="os.environ/VERTEXAI_LOCATION",
                vertex_credentials="os.environ/VERTEXAI_CREDENTIALS",
                drop_params=drop_params,
            )


def _register_claude(
    client: PassthroughClient, resources: ResourceManager, route: ClaudeRoute, *, drop_params: bool | None = None
) -> str:
    name: Final = f"e2e-claude-breakpoint-{route}-{unique_marker()}"
    model_id: Final = client.proxy.create_model(name, _claude_deployment(route, drop_params=drop_params))
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return name


def _marked(text: str, ttl: Literal["30m"] | None = None) -> TextContentPart:
    return TextContentPart(text=text, prompt_cache_breakpoint=PromptCacheBreakpoint(ttl=ttl))


def _tool_turns(result: list[ContentPart] | str, *, marked_call: bool = False) -> list[ChatTurn]:
    return [
        ChatAssistantTurn(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=ToolCallFunction(name="lookup_reference", arguments="{}"),
                    prompt_cache_breakpoint=PromptCacheBreakpoint() if marked_call else None,
                )
            ]
        ),
        ChatToolResultTurn(tool_call_id="call_1", content=result),
    ]


def _breakpoint_messages(site: BreakpointSite, prefix: str) -> list[ChatTurn]:
    ask: Final = ChatMessage(role="user", content="Reply with one word.")
    match site:
        case "system_part":
            return [ChatMessage(role="system", content=[_marked(prefix)]), ask]
        case "user_part":
            return [ChatMessage(role="user", content=[_marked(prefix), TextContentPart(text="Reply with one word.")])]
        case "string_message":
            return [ChatMessage(role="system", content=prefix, prompt_cache_breakpoint=PromptCacheBreakpoint()), ask]
        case "tool_result_part":
            return [ChatMessage(role="user", content="Look up the reference."), *_tool_turns([_marked(prefix)])]
        case "tool_call":
            return [ChatMessage(role="user", content=prefix), *_tool_turns("Done.", marked_call=True)]
        case "tool_definition":
            return [ask]


def _breakpoint_tool(site: BreakpointSite, prefix: str) -> ChatTool:
    if site != "tool_definition":
        return ChatTool(function=_LOOKUP_TOOL)
    return ChatTool(
        function=ChatToolFunction(name="lookup_reference", description=prefix, parameters={"type": "object"}),
        prompt_cache_breakpoint=PromptCacheBreakpoint(),
    )


def _breakpoint_body(model: str, site: BreakpointSite) -> ChatBody:
    """A never-seen cacheable prefix whose one ``prompt_cache_breakpoint`` sits at ``site``."""
    prefix: Final = _cacheable_prefix()
    return ChatBody(
        model=model,
        max_tokens=64,
        tools=[_breakpoint_tool(site, prefix)],
        messages=_breakpoint_messages(site, prefix),
    )


def _assert_claude_writes_then_reads_cache(client: PassthroughClient, key: str, body: ChatBody) -> None:
    first: Final = unwrap(client.proxy.chat(key, body))
    assert first.usage and first.usage.cache_creation_input_tokens, (
        f"{body.model}: the first call on a never-seen prefix wrote no cache ({first.usage}), "
        "so the prompt_cache_breakpoint never reached Claude as a cache mark"
    )

    deadline: Final = time.monotonic() + 30.0
    while True:
        second = unwrap(client.proxy.chat(key, body))
        read_tokens = _cached_read_tokens(second.usage)
        if read_tokens > 0 or time.monotonic() >= deadline:
            break
        time.sleep(3.0)

    assert read_tokens > 0, f"{body.model}: the identical second call read no cache ({second.usage})"


@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.CHAT_COMPLETIONS,
        providers=(Provider.ANTHROPIC, Provider.BEDROCK, Provider.VERTEX_AI),
        models=(
            CLAUDE_ANTHROPIC_MODEL,
            CLAUDE_BEDROCK_CONVERSE_MODEL,
            CLAUDE_BEDROCK_INVOKE_MODEL,
            CLAUDE_VERTEX_MODEL,
        ),
        capabilities=(Capability.PROMPT_CACHING,),
        mode=Mode.NONSTREAM,
    )
)
class TestPromptCacheBreakpointOnClaude:
    """OpenAI's ``prompt_cache_breakpoint`` sent to a Claude deployment caches the prefix it marks.

    The 400s that need drop_params off (a 30m ttl, or implicit ``prompt_cache_options`` on Converse) happen before any
    provider call, so the wire tests in tests/integration prove them: the shared stack sets
    ``litellm_settings.drop_params: true``, which no deployment can switch back off."""

    @pytest.mark.parametrize(
        "route", [_claude_route_param(route) for route in ("anthropic", "bedrock_converse", "bedrock_invoke", "vertex")]
    )
    @pytest.mark.parametrize(
        "site", ["system_part", "user_part", "string_message", "tool_result_part", "tool_call", "tool_definition"]
    )
    def test_a_breakpoint_writes_then_reads_the_cache(
        self,
        client: PassthroughClient,
        resources: ResourceManager,
        scoped_key: str,
        route: ClaudeRoute,
        site: BreakpointSite,
    ) -> None:
        model: Final = _register_claude(client, resources, route)
        _assert_claude_writes_then_reads_cache(client, scoped_key, _breakpoint_body(model, site))

    @pytest.mark.parametrize("route", [_claude_route_param(route) for route in ("anthropic", "bedrock_converse")])
    def test_a_30m_breakpoint_still_caches(
        self, client: PassthroughClient, resources: ResourceManager, scoped_key: str, route: ClaudeRoute
    ) -> None:
        model: Final = _register_claude(client, resources, route)
        body: Final = ChatBody(
            model=model,
            max_tokens=64,
            messages=[
                ChatMessage(role="system", content=[_marked(_cacheable_prefix(), ttl="30m")]),
                ChatMessage(role="user", content="Reply with one word."),
            ],
        )
        _assert_claude_writes_then_reads_cache(client, scoped_key, body)

    @pytest.mark.parametrize("route", [_claude_route_param(route) for route in ("anthropic", "bedrock_converse")])
    def test_five_breakpoints_keep_the_latest_four_under_drop_params(
        self, client: PassthroughClient, resources: ResourceManager, scoped_key: str, route: ClaudeRoute
    ) -> None:
        model: Final = _register_claude(client, resources, route, drop_params=True)
        body: Final = ChatBody(
            model=model,
            max_tokens=64,
            messages=[
                ChatMessage(
                    role="user",
                    content=[
                        _marked(_cacheable_prefix()),
                        *(_marked(f"Note {index}.") for index in range(4)),
                        TextContentPart(text="Reply with one word."),
                    ],
                )
            ],
        )
        _assert_claude_writes_then_reads_cache(client, scoped_key, body)

    @pytest.mark.parametrize("route", [_claude_route_param("anthropic")])
    def test_implicit_prompt_cache_options_cache_without_any_breakpoint(
        self, client: PassthroughClient, resources: ResourceManager, scoped_key: str, route: ClaudeRoute
    ) -> None:
        model: Final = _register_claude(client, resources, route)
        body: Final = ChatBody(
            model=model,
            max_tokens=64,
            prompt_cache_options=PromptCacheOptions(mode="implicit"),
            messages=[
                ChatMessage(role="system", content=_cacheable_prefix()),
                ChatMessage(role="user", content="Reply with one word."),
            ],
        )
        _assert_claude_writes_then_reads_cache(client, scoped_key, body)

    @pytest.mark.parametrize(
        "route", [_claude_route_param(route) for route in ("anthropic", "bedrock_converse", "bedrock_invoke", "vertex")]
    )
    def test_a_block_carrying_both_keys_still_caches(
        self, client: PassthroughClient, resources: ResourceManager, scoped_key: str, route: ClaudeRoute
    ) -> None:
        model: Final = _register_claude(client, resources, route)
        both_keys: Final = TextContentPart(
            text=_cacheable_prefix(),
            cache_control=CacheControl(),
            prompt_cache_breakpoint=PromptCacheBreakpoint(ttl="30m"),
        )
        body: Final = ChatBody(
            model=model,
            max_tokens=64,
            messages=[
                ChatMessage(role="system", content=[both_keys]),
                ChatMessage(role="user", content="Reply with one word."),
            ],
        )
        _assert_claude_writes_then_reads_cache(client, scoped_key, body)
