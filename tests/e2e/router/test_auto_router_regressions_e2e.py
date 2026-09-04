"""Live e2e regression pins for strategy-router (auto-router) routing.

A strategy marker (an ``auto_router/complexity_router`` deployment) and a plain
deployment can share one ``model_name``, split by tags once
``enable_tag_filtering`` is on: tagged requests route through the marker to its
tier models, untagged requests go to the plain deployment. That split, and the
strategy-router alias behaviors around it, regressed repeatedly; each test here
pins one fixed behavior:

- GitHub issue #36619: a tagged request selects the tagged marker under a
  shared name even when a plain deployment was registered first.
- GitHub issue #36620: untagged requests keep being served by the plain
  deployment on every call, never captured or 400'd by the tagged marker.
- GitHub issue #36621: a request tagged via the ``x-litellm-tags`` header
  routes through the marker even when the tier deployments carry no tags
  (the marker consumes the routing tags before deployment selection), while a
  tagged call aimed straight at an untagged deployment stays denied.
- GitHub issues #36620/#36621 on /v1/responses: the same tag split holds for
  string and list input, whether the tag arrives in litellm_metadata or the
  x-litellm-tags header.
- GitHub PR #37333: /v1/responses input is resolved into messages for a
  semantic ``auto_router`` deployment's pre-routing hook; such requests used
  to fail with 400 "Unmapped LLM provider auto_router" because only chat
  messages fed the route matcher.
- GitHub PR #36691: custom pricing on the marker alias never prices the routed
  request; spend logs at the routed tier deployment's own rate.
- GitHub PR #36721: the heuristic complexity classifier scores the caller's
  current ask only, so a large agent system prompt cannot inflate the tier.
- GitHub PR #36626: connection params on the marker alias (``api_key``,
  ``api_base``) stay with the alias; the routed tier calls its provider with
  its own credentials.

Every deployment is registered via /model/new (stage has no static config for
these) and ``enable_tag_filtering`` is enabled through key-level
``router_settings`` on the keys the tag tests mint, so the switch rides only
this module's own requests and the rest of the suite is never filtered.
The served deployment is always read back from the spend log's ``model``,
which stores either the registered alias or the provider-prefixed form.
"""

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest
from pydantic import BaseModel, ConfigDict, Field

from e2e_config import unique_marker
from e2e_http import AnthropicHeaders, AuthHeaders, UnauthorizedError, unwrap
from lifecycle import ResourceManager
from models import (
    AnthropicMessagesBody,
    AnthropicMessagesResponse,
    ChatBody,
    ChatMessage,
    ChatMetadata,
    KeyGenerateBody,
    LiteLLMParamsBody,
    RouterSettingsOverride,
    SpendLogRow,
)
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

PLAIN_MODEL = "anthropic/claude-sonnet-5"
CHEAP_MODEL = "anthropic/claude-haiku-4-5"
STRONG_MODEL = "openai/gpt-5.6"
MAX_TOKENS = 16
TAG_DENIAL_MESSAGE = "Not allowed to access model due to tags configuration"
PLAIN_SERVED = frozenset({PLAIN_MODEL, "claude-sonnet-5"})
CHEAP_SERVED = frozenset({CHEAP_MODEL, "claude-haiku-4-5"})
EMBEDDING_MODEL = "openai/text-embedding-3-small"
SEMANTIC_ROUTE_UTTERANCE = "summarize this quarterly revenue report into three bullet points"

KEYWORD_HEAVY_SYSTEM_PROMPT = (
    "You are the principal architecture assistant for a distributed systems platform. "
    "Analyze every request step by step: design the algorithm, prove its correctness, "
    "evaluate time and space complexity, and reason about concurrency, consistency, and "
    "fault tolerance tradeoffs. When asked, refactor and debug multi-threaded code, "
    "optimize database query plans, derive mathematical proofs, and explain the theorem "
    "or lemma behind each optimization. Think through edge cases rigorously before answering. "
) * 4


class TaggedAuthHeaders(AuthHeaders):
    x_litellm_tags: str | None = Field(default=None, serialization_alias="x-litellm-tags")


class TaggedAnthropicHeaders(AnthropicHeaders):
    x_litellm_tags: str | None = Field(default=None, serialization_alias="x-litellm-tags")


class ResponsesTagMetadata(BaseModel):
    tags: list[str]


class ResponsesInputItem(BaseModel):
    role: str
    content: str


class ResponsesBody(BaseModel):
    model: str
    input: str | list[ResponsesInputItem]
    max_output_tokens: int | None = None
    litellm_metadata: ResponsesTagMetadata | None = None


class ResponsesApiResponse(BaseModel):
    """Minimal /v1/responses answer shape; routing is proven from spend logs,
    so only the fields the assertions read are modeled."""

    model_config = ConfigDict(extra="allow")
    id: str | None = None
    status: str | None = None
    model: str | None = None


@dataclass(frozen=True, slots=True)
class TagSplitDeployments:
    """Scenario A mirrors the customer-shaped config from GitHub issue #36619:
    plain deployment registered first, tier deployment and marker both tagged.
    Scenario B flips both axes for GitHub issue #36621: marker registered first
    and its tier deployment left untagged, so routing depends neither on
    registration order nor on tier deployments carrying tags."""

    tag_a: str
    shared_a: str
    tier_a: str
    tag_b: str
    shared_b: str
    tier_b: str


@dataclass(frozen=True, slots=True)
class ZeroPricedAlias:
    alias: str
    tier: str


@dataclass(frozen=True, slots=True)
class HeuristicSplit:
    alias: str
    cheap: str
    strong: str


@dataclass(frozen=True, slots=True)
class SemanticAutoRouter:
    marker: str
    target: str
    fallback: str
    embedding: str


@dataclass(frozen=True, slots=True)
class CredentialedAlias:
    alias: str
    tier: str


def _provider_key(env_var: str) -> str:
    return os.environ.get(env_var) or f"os.environ/{env_var}"


def _uniform_tier_config(tier_model: str) -> dict[str, object]:
    return {
        "classifier_type": "heuristic",
        "tiers": {"SIMPLE": tier_model, "MEDIUM": tier_model, "COMPLEX": tier_model, "REASONING": tier_model},
    }


def _key_for(
    proxy: ProxyClient, resources: ResourceManager, models: list[str], tag_filtering: bool = False
) -> str:
    key: Final = proxy.generate_key(
        KeyGenerateBody(
            models=models,
            user_id="e2e-auto-router-regressions",
            router_settings=RouterSettingsOverride(enable_tag_filtering=True) if tag_filtering else None,
        )
    )
    resources.defer(lambda: proxy.delete_key(key))
    return key


def _hello_chat_body(model: str, tags: list[str] | None = None) -> ChatBody:
    return ChatBody(
        model=model,
        messages=[ChatMessage(role="user", content=f"say hello {unique_marker()}")],
        max_tokens=MAX_TOKENS,
        metadata=ChatMetadata(tags=tags) if tags is not None else None,
    )


def _hello_messages_body(model: str) -> AnthropicMessagesBody:
    return AnthropicMessagesBody(
        model=model,
        messages=[ChatMessage(role="user", content=f"say hello {unique_marker()}")],
        max_tokens=MAX_TOKENS,
    )


def _assert_served_only_by(rows: list[SpendLogRow], allowed: frozenset[str], context: str) -> None:
    served: Final = tuple(row.model for row in rows)
    assert served and all(model in allowed for model in served), (
        f"{context}: expected every request to be served by one of {sorted(allowed)}, spend logs show {served}"
    )


@pytest.fixture(scope="module")
def split(proxy: ProxyClient) -> Iterator[TagSplitDeployments]:
    marker: Final = unique_marker()
    deployments: Final = TagSplitDeployments(
        tag_a=f"e2e-split-a-{marker}",
        shared_a=f"e2e-autoroute-a-{marker}",
        tier_a=f"e2e-tier-a-{marker}",
        tag_b=f"e2e-split-b-{marker}",
        shared_b=f"e2e-autoroute-b-{marker}",
        tier_b=f"e2e-tier-b-{marker}",
    )
    anthropic_key: Final = _provider_key("ANTHROPIC_API_KEY")
    marker_params_a: Final = LiteLLMParamsBody(
        model="auto_router/complexity_router",
        complexity_router_config=_uniform_tier_config(deployments.tier_a),
        tags=[deployments.tag_a],
    )
    marker_params_b: Final = LiteLLMParamsBody(
        model="auto_router/complexity_router",
        complexity_router_config=_uniform_tier_config(deployments.tier_b),
        tags=[deployments.tag_b],
    )
    registrations: Final[tuple[tuple[str, LiteLLMParamsBody], ...]] = (
        (deployments.shared_a, LiteLLMParamsBody(model=PLAIN_MODEL, api_key=anthropic_key)),
        (deployments.tier_a, LiteLLMParamsBody(model=CHEAP_MODEL, api_key=anthropic_key, tags=[deployments.tag_a])),
        (deployments.shared_a, marker_params_a),
        (deployments.shared_b, marker_params_b),
        (deployments.tier_b, LiteLLMParamsBody(model=CHEAP_MODEL, api_key=anthropic_key)),
        (deployments.shared_b, LiteLLMParamsBody(model=PLAIN_MODEL, api_key=anthropic_key)),
    )
    created: Final = tuple(proxy.create_model(name, params) for name, params in registrations)
    try:
        yield deployments
    finally:
        for model_id in created:
            proxy.delete_model(model_id)


@pytest.fixture(scope="module")
def zero_priced_alias(proxy: ProxyClient) -> Iterator[ZeroPricedAlias]:
    marker: Final = unique_marker()
    named: Final = ZeroPricedAlias(alias=f"e2e-priced-alias-{marker}", tier=f"e2e-priced-tier-{marker}")
    alias_params: Final = LiteLLMParamsBody(
        model="auto_router/complexity_router",
        complexity_router_config=_uniform_tier_config(named.tier),
        input_cost_per_token=0.0,
        output_cost_per_token=0.0,
    )
    registrations: Final[tuple[tuple[str, LiteLLMParamsBody], ...]] = (
        (named.tier, LiteLLMParamsBody(model=CHEAP_MODEL, api_key=_provider_key("ANTHROPIC_API_KEY"))),
        (named.alias, alias_params),
    )
    created: Final = tuple(proxy.create_model(name, params) for name, params in registrations)
    try:
        yield named
    finally:
        for model_id in created:
            proxy.delete_model(model_id)


@pytest.fixture(scope="module")
def heuristic_split(proxy: ProxyClient) -> Iterator[HeuristicSplit]:
    marker: Final = unique_marker()
    named: Final = HeuristicSplit(
        alias=f"e2e-heuristic-router-{marker}",
        cheap=f"e2e-heuristic-cheap-{marker}",
        strong=f"e2e-heuristic-strong-{marker}",
    )
    config: Final[dict[str, object]] = {
        "classifier_type": "heuristic",
        "token_thresholds": {"simple": 15, "complex": 400},
        "tiers": {"SIMPLE": named.cheap, "MEDIUM": named.strong, "COMPLEX": named.strong, "REASONING": named.strong},
    }
    registrations: Final[tuple[tuple[str, LiteLLMParamsBody], ...]] = (
        (named.cheap, LiteLLMParamsBody(model=CHEAP_MODEL, api_key=_provider_key("ANTHROPIC_API_KEY"))),
        (named.strong, LiteLLMParamsBody(model=STRONG_MODEL, api_key=_provider_key("OPENAI_API_KEY"))),
        (named.alias, LiteLLMParamsBody(model="auto_router/complexity_router", complexity_router_config=config)),
    )
    created: Final = tuple(proxy.create_model(name, params) for name, params in registrations)
    try:
        yield named
    finally:
        for model_id in created:
            proxy.delete_model(model_id)


@pytest.fixture(scope="module")
def semantic_auto_router(proxy: ProxyClient) -> Iterator[SemanticAutoRouter]:
    marker: Final = unique_marker()
    named: Final = SemanticAutoRouter(
        marker=f"e2e-semantic-router-{marker}",
        target=f"e2e-semantic-target-{marker}",
        fallback=f"e2e-semantic-fallback-{marker}",
        embedding=f"e2e-semantic-embedding-{marker}",
    )
    router_config: Final = json.dumps(
        {"routes": [{"name": named.target, "utterances": [SEMANTIC_ROUTE_UTTERANCE], "score_threshold": 0.3}]}
    )
    marker_params: Final = LiteLLMParamsBody(
        model=f"auto_router/{named.marker}",
        auto_router_config=router_config,
        auto_router_default_model=named.fallback,
        auto_router_embedding_model=named.embedding,
    )
    registrations: Final[tuple[tuple[str, LiteLLMParamsBody], ...]] = (
        (named.embedding, LiteLLMParamsBody(model=EMBEDDING_MODEL, api_key=_provider_key("OPENAI_API_KEY"))),
        (named.target, LiteLLMParamsBody(model=CHEAP_MODEL, api_key=_provider_key("ANTHROPIC_API_KEY"))),
        (named.fallback, LiteLLMParamsBody(model=PLAIN_MODEL, api_key=_provider_key("ANTHROPIC_API_KEY"))),
        (named.marker, marker_params),
    )
    created: Final = tuple(proxy.create_model(name, params) for name, params in registrations)
    try:
        yield named
    finally:
        for model_id in created:
            proxy.delete_model(model_id)


@pytest.fixture(scope="module")
def credentialed_alias(proxy: ProxyClient) -> Iterator[CredentialedAlias]:
    marker: Final = unique_marker()
    named: Final = CredentialedAlias(alias=f"e2e-cred-alias-{marker}", tier=f"e2e-cred-tier-{marker}")
    alias_params: Final = LiteLLMParamsBody(
        model="auto_router/complexity_router",
        complexity_router_config=_uniform_tier_config(named.tier),
        api_key=f"sk-alias-never-used-{marker}",
    )
    registrations: Final[tuple[tuple[str, LiteLLMParamsBody], ...]] = (
        (named.tier, LiteLLMParamsBody(model=CHEAP_MODEL, api_key=_provider_key("ANTHROPIC_API_KEY"))),
        (named.alias, alias_params),
    )
    created: Final = tuple(proxy.create_model(name, params) for name, params in registrations)
    try:
        yield named
    finally:
        for model_id in created:
            proxy.delete_model(model_id)


class TestTagSplitRouting:
    @pytest.mark.covers("reliability.routing.tagged_marker.request_tag_selects_marker")
    def test_body_tagged_chat_routes_through_the_marker_to_its_tier(
        self, proxy: ProxyClient, resources: ResourceManager, split: TagSplitDeployments
    ) -> None:
        """Pins GitHub issue #36619: with tag filtering on, a chat request whose
        body metadata tags match the tagged marker under a shared model name is
        answered by the marker's tier deployment, not by the plain deployment
        that was registered under the name first."""
        key: Final = _key_for(proxy, resources, [split.shared_a, split.tier_a], tag_filtering=True)
        chat: Final = unwrap(proxy.chat(key, _hello_chat_body(split.shared_a, tags=[split.tag_a])))
        assert chat.choices, "tagged chat through the shared name returned no choices"
        rows: Final = proxy.poll_logs_for_key(key, min_rows=1)
        _assert_served_only_by(rows, CHEAP_SERVED | {split.tier_a}, "body-tagged chat on the shared name")

    @pytest.mark.covers("reliability.routing.tagged_marker.untagged_request_served_by_plain_deployment")
    def test_untagged_chat_is_always_served_by_the_plain_deployment(
        self, proxy: ProxyClient, resources: ResourceManager, split: TagSplitDeployments
    ) -> None:
        """Pins GitHub issue #36620: untagged chat requests to the shared name
        succeed on every call and are all served by the plain deployment; the
        tagged marker never captures them, so no intermittent auto-router
        errors and no tier hijacking."""
        key: Final = _key_for(proxy, resources, [split.shared_a, split.tier_a], tag_filtering=True)
        for _ in range(5):
            chat = unwrap(proxy.chat(key, _hello_chat_body(split.shared_a)))
            assert chat.choices, "untagged chat through the shared name returned no choices"
        rows: Final = proxy.poll_logs_for_key(key, min_rows=5)
        _assert_served_only_by(rows, PLAIN_SERVED | {split.shared_a}, "untagged chat on the shared name")

    @pytest.mark.covers("reliability.routing.tagged_marker.untagged_request_served_by_plain_deployment")
    def test_untagged_messages_is_served_by_the_plain_deployment(
        self, proxy: ProxyClient, resources: ResourceManager, split: TagSplitDeployments
    ) -> None:
        """Pins GitHub issue #36620 on the /v1/messages surface: an untagged
        Anthropic-native request to the shared name is served by the plain
        deployment, not captured by the tagged marker."""
        key: Final = _key_for(proxy, resources, [split.shared_a, split.tier_a], tag_filtering=True)
        answer: Final = unwrap(proxy.messages(key, _hello_messages_body(split.shared_a)))
        assert answer.content or answer.choices, "untagged /v1/messages returned neither content nor choices"
        rows: Final = proxy.poll_logs_for_key(key, min_rows=1)
        _assert_served_only_by(rows, PLAIN_SERVED | {split.shared_a}, "untagged /v1/messages on the shared name")


class TestUntaggedTierDeployments:
    @pytest.mark.covers("reliability.routing.tagged_marker.header_tag_selects_marker")
    def test_header_tagged_messages_routes_through_the_marker_to_an_untagged_tier(
        self, proxy: ProxyClient, resources: ResourceManager, split: TagSplitDeployments
    ) -> None:
        """Pins GitHub issue #36621: a /v1/messages request tagged only via the
        x-litellm-tags header selects the tagged marker, and the rewrite still
        lands on the tier deployment even though that deployment carries no
        tags, because the marker consumed the routing tags."""
        key: Final = _key_for(proxy, resources, [split.shared_b, split.tier_b], tag_filtering=True)
        headers: Final = TaggedAnthropicHeaders(authorization=f"Bearer {key}", x_litellm_tags=split.tag_b)
        answer: Final = unwrap(
            proxy.transport.post(
                "/v1/messages",
                headers=headers,
                json=_hello_messages_body(split.shared_b),
                response_type=AnthropicMessagesResponse,
            )
        )
        assert answer.content or answer.choices, "header-tagged /v1/messages returned neither content nor choices"
        rows: Final = proxy.poll_logs_for_key(key, min_rows=1)
        _assert_served_only_by(rows, CHEAP_SERVED | {split.tier_b}, "header-tagged /v1/messages on the shared name")

    @pytest.mark.covers("reliability.routing.tagged_marker.untagged_tier_deployments_still_served")
    def test_body_tagged_chat_reaches_the_untagged_tier_after_marker_rewrite(
        self, proxy: ProxyClient, resources: ResourceManager, split: TagSplitDeployments
    ) -> None:
        """Pins the tag-consumption half of GitHub issue #36621: after the
        tagged marker rewrites the request to its tier model, the consumed
        routing tags no longer constrain deployment selection, so the untagged
        tier deployment serves the request instead of a strict-tag denial."""
        key: Final = _key_for(proxy, resources, [split.shared_b, split.tier_b], tag_filtering=True)
        chat: Final = unwrap(proxy.chat(key, _hello_chat_body(split.shared_b, tags=[split.tag_b])))
        assert chat.choices, "body-tagged chat through the marker-first shared name returned no choices"
        rows: Final = proxy.poll_logs_for_key(key, min_rows=1)
        _assert_served_only_by(rows, CHEAP_SERVED | {split.tier_b}, "body-tagged chat with untagged tier")

    @pytest.mark.covers("reliability.routing.tagged_marker.tag_semantics_stay_strict")
    def test_tagged_call_straight_at_an_untagged_deployment_stays_denied(
        self, proxy: ProxyClient, resources: ResourceManager, split: TagSplitDeployments
    ) -> None:
        """The tag-consumption fix must not loosen strict tag semantics: a
        tagged request aimed directly at an untagged deployment (no marker
        involved) is still rejected with the 401 tags-configuration error."""
        key: Final = _key_for(proxy, resources, [split.tier_b], tag_filtering=True)
        result: Final = proxy.chat(key, _hello_chat_body(split.tier_b, tags=[split.tag_b]))
        assert isinstance(result, UnauthorizedError), (
            f"expected the tagged direct call to an untagged deployment to be denied with 401, got {result}"
        )
        assert TAG_DENIAL_MESSAGE in result.body, (
            f"expected the denial to come from tag routing, got a 401 reading {result.body[:300]}"
        )


class TestResponsesApiTagRouting:
    @pytest.mark.covers("reliability.routing.tagged_marker.responses_input_routes_through_marker")
    def test_header_tagged_responses_with_string_input_routes_to_the_tier(
        self, proxy: ProxyClient, resources: ResourceManager, split: TagSplitDeployments
    ) -> None:
        """Pins the /v1/responses surface of the tag split (GitHub issues
        #36620/#36621): a /v1/responses request with string input, tagged via
        the x-litellm-tags header, succeeds and routes through the tagged
        marker to its tier."""
        key: Final = _key_for(proxy, resources, [split.shared_a, split.tier_a], tag_filtering=True)
        headers: Final = TaggedAuthHeaders(authorization=f"Bearer {key}", x_litellm_tags=split.tag_a)
        body: Final = ResponsesBody(
            model=split.shared_a, input=f"say hello {unique_marker()}", max_output_tokens=64
        )
        answer: Final = unwrap(
            proxy.transport.post("/v1/responses", headers=headers, json=body, response_type=ResponsesApiResponse)
        )
        assert answer.id, "header-tagged /v1/responses returned no response id"
        rows: Final = proxy.poll_logs_for_key(key, min_rows=1)
        _assert_served_only_by(rows, CHEAP_SERVED | {split.tier_a}, "header-tagged /v1/responses string input")

    @pytest.mark.covers("reliability.routing.tagged_marker.responses_input_routes_through_marker")
    def test_body_tagged_responses_with_list_input_routes_to_the_tier(
        self, proxy: ProxyClient, resources: ResourceManager, split: TagSplitDeployments
    ) -> None:
        """Pins the body-tag and list-input combination of the same split:
        /v1/responses with litellm_metadata.tags and structured input items
        routes through the tagged marker to its tier."""
        key: Final = _key_for(proxy, resources, [split.shared_a, split.tier_a], tag_filtering=True)
        body: Final = ResponsesBody(
            model=split.shared_a,
            input=[ResponsesInputItem(role="user", content=f"say hello {unique_marker()}")],
            max_output_tokens=64,
            litellm_metadata=ResponsesTagMetadata(tags=[split.tag_a]),
        )
        answer: Final = unwrap(
            proxy.transport.post(
                "/v1/responses",
                headers=proxy.transport.bearer(key),
                json=body,
                response_type=ResponsesApiResponse,
            )
        )
        assert answer.id, "body-tagged /v1/responses returned no response id"
        rows: Final = proxy.poll_logs_for_key(key, min_rows=1)
        _assert_served_only_by(rows, CHEAP_SERVED | {split.tier_a}, "body-tagged /v1/responses list input")

    @pytest.mark.covers("reliability.routing.tagged_marker.untagged_request_served_by_plain_deployment")
    def test_untagged_responses_is_served_by_the_plain_deployment(
        self, proxy: ProxyClient, resources: ResourceManager, split: TagSplitDeployments
    ) -> None:
        """Pins the untagged half of the /v1/responses tag split: an untagged
        request to the shared name is served by the plain deployment, matching
        the chat and messages surfaces."""
        key: Final = _key_for(proxy, resources, [split.shared_a, split.tier_a], tag_filtering=True)
        body: Final = ResponsesBody(
            model=split.shared_a, input=f"say hello {unique_marker()}", max_output_tokens=64
        )
        answer: Final = unwrap(
            proxy.transport.post(
                "/v1/responses",
                headers=proxy.transport.bearer(key),
                json=body,
                response_type=ResponsesApiResponse,
            )
        )
        assert answer.id, "untagged /v1/responses returned no response id"
        rows: Final = proxy.poll_logs_for_key(key, min_rows=1)
        _assert_served_only_by(rows, PLAIN_SERVED | {split.shared_a}, "untagged /v1/responses on the shared name")


class TestStrategyAliasPricing:
    @pytest.mark.covers("reliability.routing.strategy_alias.custom_pricing_ignored")
    def test_zero_priced_alias_still_logs_spend_at_the_tier_rate(
        self, proxy: ProxyClient, resources: ResourceManager, zero_priced_alias: ZeroPricedAlias
    ) -> None:
        """Pins GitHub PR #36691: custom pricing registered on a strategy-router
        alias never prices the routed request. The alias here carries explicit
        zero pricing, so any zero-spend row would prove the alias pricing was
        applied; the routed tier deployment's real rate must produce spend > 0."""
        key: Final = _key_for(proxy, resources, [zero_priced_alias.alias, zero_priced_alias.tier])
        chat: Final = unwrap(proxy.chat(key, _hello_chat_body(zero_priced_alias.alias)))
        assert chat.choices, "chat through the zero-priced alias returned no choices"
        rows: Final = proxy.poll_logs_for_key(
            key, min_rows=1, predicate=lambda logged: all((row.spend or 0.0) > 0.0 for row in logged)
        )
        _assert_served_only_by(rows, CHEAP_SERVED | {zero_priced_alias.tier}, "chat through the zero-priced alias")
        priced: Final = tuple((row.model, row.spend) for row in rows)
        assert all((row.spend or 0.0) > 0.0 for row in rows), (
            f"expected spend at the tier deployment's own rate, got zero-spend rows: {priced}"
        )


class TestComplexityHeuristicScope:
    @pytest.mark.covers("reliability.routing.complexity_heuristic.scores_current_ask_only")
    def test_trivial_ask_behind_keyword_heavy_system_prompt_stays_on_the_cheap_tier(
        self, proxy: ProxyClient, resources: ResourceManager, heuristic_split: HeuristicSplit
    ) -> None:
        """Pins GitHub PR #36721: the heuristic complexity classifier scores the
        caller's current ask alone. The trivial ask scores SIMPLE on its own,
        while the accompanying ~2KB agent system prompt is packed with enough
        reasoning and complexity keywords that scoring the combined text lands
        in REASONING; only ask-only scoring keeps this on the cheap tier."""
        key: Final = _key_for(
            proxy, resources, [heuristic_split.alias, heuristic_split.cheap, heuristic_split.strong]
        )
        body: Final = ChatBody(
            model=heuristic_split.alias,
            messages=[
                ChatMessage(role="system", content=KEYWORD_HEAVY_SYSTEM_PROMPT),
                ChatMessage(role="user", content=f"hi {unique_marker()}"),
            ],
            max_tokens=MAX_TOKENS,
        )
        chat: Final = unwrap(proxy.chat(key, body))
        assert chat.choices, "chat through the heuristic router returned no choices"
        rows: Final = proxy.poll_logs_for_key(key, min_rows=1)
        _assert_served_only_by(
            rows, CHEAP_SERVED | {heuristic_split.cheap}, "trivial ask behind a keyword-heavy system prompt"
        )


class TestSemanticAutoRouterResponses:
    @pytest.mark.covers("reliability.routing.semantic_auto_router.responses_input_routed")
    def test_responses_input_reaches_the_semantic_auto_router(
        self, proxy: ProxyClient, resources: ResourceManager, semantic_auto_router: SemanticAutoRouter
    ) -> None:
        """Pins GitHub PR #37333: /v1/responses input is resolved into messages
        for the semantic auto-router's pre-routing hook, so the marker embeds
        the input, matches its route, and the target deployment serves the
        request; before the fix the hook saw no messages and the request
        failed with 400 "Unmapped LLM provider auto_router"."""
        key: Final = _key_for(
            proxy,
            resources,
            [semantic_auto_router.marker, semantic_auto_router.target, semantic_auto_router.fallback],
        )
        body: Final = ResponsesBody(
            model=semantic_auto_router.marker, input=SEMANTIC_ROUTE_UTTERANCE, max_output_tokens=64
        )
        answer: Final = unwrap(
            proxy.transport.post(
                "/v1/responses",
                headers=proxy.transport.bearer(key),
                json=body,
                response_type=ResponsesApiResponse,
            )
        )
        assert answer.id, "/v1/responses through the semantic auto-router returned no response id"
        rows: Final = proxy.poll_logs_for_key(key, min_rows=1)
        _assert_served_only_by(
            rows, CHEAP_SERVED | {semantic_auto_router.target}, "semantic auto-router /v1/responses string input"
        )


class TestAliasParamForwarding:
    @pytest.mark.covers("reliability.routing.tagged_marker.alias_connection_params_stay_with_tier")
    def test_alias_api_key_never_overrides_the_tier_credential(
        self, proxy: ProxyClient, resources: ResourceManager, credentialed_alias: CredentialedAlias
    ) -> None:
        """Pins GitHub PR #36626: an api_key set on the marker alias entry is
        never forwarded onto the routed request, so the tier deployment calls
        its provider with its own credential. Before the fix the alias's key
        was copied into the request, overriding the tier's credential, and
        every routed call failed provider auth."""
        key: Final = _key_for(proxy, resources, [credentialed_alias.alias, credentialed_alias.tier])
        chat: Final = unwrap(proxy.chat(key, _hello_chat_body(credentialed_alias.alias)))
        assert chat.choices, "chat through the credentialed alias returned no choices"
        rows: Final = proxy.poll_logs_for_key(key, min_rows=1)
        _assert_served_only_by(rows, CHEAP_SERVED | {credentialed_alias.tier}, "chat through the credentialed alias")
