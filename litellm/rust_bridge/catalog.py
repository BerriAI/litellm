"""Ordered rollout policy for routes.

The first matching rule wins; unmatched contexts stay on Python. Native
admission separately decides whether the selected implementation can execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, TypeAlias

from litellm.rust_bridge.configuration import Decision, Rollout
from litellm.rust_bridge.configuration import decision as _decision


class Route(str, Enum):
    CHAT_COMPLETIONS = "chat_completions"
    EMBEDDINGS = "embeddings"
    MESSAGES = "messages"
    RESPONSES = "responses"
    TRANSCRIPTION = "transcription"
    OCR = "ocr"
    TOKEN_COUNTER = "token_counter"
    TOKENIZER = "tokenizer"


class Target(str, Enum):
    CHAT_COMPLETIONS = "chat_completions"
    MESSAGES = "messages"
    RESPONSES = "responses"


class Feature(str, Enum):
    SAFEGUARDS_CLASSIFIER = "safeguards_classifier"
    """A non-Claude model with dangerous_tool_use safeguards, which the proxy classifier answers."""
    MOCK_RESPONSE = "mock_response"
    """mock_response short-circuits before any provider call."""
    PROXY_MCP_TOOLS = "proxy_mcp_tools"
    """litellm_proxy MCP tools that Python expands and may auto-execute."""
    CALLER_CLIENT = "caller_client"
    """A caller-supplied HTTP client or shared session that Python sends through."""
    TIMEOUT_VARIANTS = "timeout_variants"
    """request_timeout, stream_timeout or httpx.Timeout phases beyond a plain timeout."""
    DROP_PARAMS = "drop_params"
    """drop_params or additional_drop_params dropping fields per provider."""
    MODIFY_PARAMS = "modify_params"
    """litellm.modify_params rewriting the request, like clamping max_tokens."""
    STREAMING = "streaming"
    """stream=True."""
    PROMPT_MANAGEMENT = "prompt_management"
    """prompt_id resolved through a prompt manager."""
    TEXT_FORMAT = "text_format"
    """A text_format model that Python turns into a JSON schema and parses back."""
    CHAT_COMPLETIONS_BRIDGE = "chat_completions_bridge"
    """use_chat_completions_api or the openai/chat_completions/ prefix, served through Chat Completions."""
    RETRIES_FALLBACKS = "retries_fallbacks"
    """num_retries, max_retries or fallbacks run by the SDK."""
    DEPLOYMENT_RESPONSE_ID = "deployment_response_id"
    """A deployment id that Python encodes into the returned response id."""
    ENCODED_ITEM_IDS = "encoded_item_ids"
    """Input items carrying litellm-encoded ids that Python restores before sending."""
    INPUT_REWRITE = "input_rewrite"
    """Responses input items that Python normalizes or removes before replay."""
    EXTRA_BODY = "extra_body"
    """Extra body fields that Python transforms and merges into the provider request."""
    PROVIDER_AFFINITY = "provider_affinity"
    """A provider affinity header that Python validates and populates from the session id."""
    PRE_REQUEST_HOOK = "pre_request_hook"
    """A logger overriding async_pre_request_hook, which may rewrite the request."""
    WEBSEARCH_INTERCEPTION = "websearch_interception"
    """A WebSearchInterceptionLogger that may answer without calling the provider."""
    MESSAGES_INTERCEPTOR = "messages_interceptor"
    """A registered MessagesInterceptor that takes over the call."""
    AGENTIC_LOOP_HOOK = "agentic_loop_hook"
    """A logger overriding async_should_run_agentic_loop, which may run follow-up calls."""
    RESPONSE_CACHE = "response_cache"
    """litellm.cache, whose hit and store order differs from native."""
    TLS_POLICY = "tls_policy"
    """An OpenSSL security level or ECDH curve that native TLS cannot apply."""
    MANAGED_RESOURCE_IDS = "managed_resource_ids"
    """Managed file or vector store ids that Python maps to provider ids."""
    CACHE_CONTROL_MARKS = "cache_control_marks"
    """Anthropic cache_control marks that Python strips or turns into prompt cache breakpoints."""
    ANTHROPIC_TOOL_SEARCH = "anthropic_tool_search"
    """Anthropic tool search tools that Python maps to OpenAI tool search."""
    TOOL_SCHEMA_REPAIR = "tool_schema_repair"
    """Tool schemas that Python decodes, flattens or strips of non-Python regex patterns."""
    MODEL_PARAM_GATING = "model_param_gating"
    """Params that Python clamps, drops or rejects from the model map, like gpt-5 temperature."""
    WORKLOAD_IDENTITY = "workload_identity"
    """No static key, so Python mints a workload identity federation token."""
    CACHE_CONTROL_INJECTION = "cache_control_injection"
    """Prompt caching injection that Python applies to messages and system."""
    STREAM_COST_INJECTION = "stream_cost_injection"
    """litellm.include_cost_in_streaming_usage, where Python adds cost to streamed usage."""
    THINKING_SIGNATURE_RETRY = "thinking_signature_retry"
    """Replayed thinking blocks, which Python strips and retries once if Anthropic rejects their signature."""
    CLIENT_HEADERS = "client_headers"
    """Proxy-forwarded headers that Python merges into extra_headers."""
    DEPLOYMENT_HOOK = "deployment_hook"
    """A logger overriding async_pre_call_deployment_hook, which the sync Python handler runs itself."""
    MODEL_LIST = "model_list"
    """A model_list that Python fans out across its deployments."""
    MODEL_ALIAS_MAP = "model_alias_map"
    """A model that litellm.model_alias_map rewrites before the call."""


@dataclass(frozen=True, slots=True)
class RouteContext:
    route: Route
    provider: str | None = None
    model: str | None = None
    target: Target | None = None
    features: frozenset[Feature] = frozenset()


@dataclass(frozen=True, slots=True)
class RouteRule:
    route: Route
    rollout: Rollout
    providers: frozenset[str] | None = None
    models: frozenset[str] | None = None
    targets: frozenset[Target] | None = None
    features: frozenset[Feature] | None = None

    def unconstrained(self) -> bool:
        return self.providers is None and self.models is None and self.targets is None and self.features is None

    def matches(self, context: RouteContext) -> bool:
        return (
            context.route is self.route
            and (self.providers is None or context.provider in self.providers)
            and (self.models is None or context.model in self.models)
            and (self.targets is None or context.target in self.targets)
            and (self.features is None or not self.features.isdisjoint(context.features))
        )


Rules: TypeAlias = tuple[RouteRule, ...]

COMMON_PYTHON_FEATURES: Final = frozenset(
    {
        Feature.MOCK_RESPONSE,
        Feature.PROXY_MCP_TOOLS,
        Feature.CALLER_CLIENT,
        Feature.TIMEOUT_VARIANTS,
        Feature.AGENTIC_LOOP_HOOK,
        Feature.TLS_POLICY,
    }
)

CHAT_PYTHON_FEATURES: Final = COMMON_PYTHON_FEATURES | {
    Feature.RESPONSE_CACHE,
    Feature.STREAMING,
    Feature.DROP_PARAMS,
    Feature.MODIFY_PARAMS,
    Feature.EXTRA_BODY,
    Feature.RETRIES_FALLBACKS,
    Feature.PROMPT_MANAGEMENT,
    Feature.CACHE_CONTROL_INJECTION,
    Feature.MODEL_PARAM_GATING,
    Feature.MODEL_LIST,
    Feature.MODEL_ALIAS_MAP,
}

MESSAGES_PYTHON_FEATURES: Final = COMMON_PYTHON_FEATURES | {
    Feature.SAFEGUARDS_CLASSIFIER,
    Feature.PRE_REQUEST_HOOK,
    Feature.WEBSEARCH_INTERCEPTION,
    Feature.MESSAGES_INTERCEPTOR,
    Feature.RESPONSE_CACHE,
    Feature.MODIFY_PARAMS,
    Feature.WORKLOAD_IDENTITY,
    Feature.CACHE_CONTROL_INJECTION,
    Feature.STREAM_COST_INJECTION,
    Feature.THINKING_SIGNATURE_RETRY,
}

RESPONSES_PYTHON_FEATURES: Final = COMMON_PYTHON_FEATURES | {
    Feature.DROP_PARAMS,
    Feature.MODIFY_PARAMS,
    Feature.STREAMING,
    Feature.PROMPT_MANAGEMENT,
    Feature.TEXT_FORMAT,
    Feature.CHAT_COMPLETIONS_BRIDGE,
    Feature.RETRIES_FALLBACKS,
    Feature.DEPLOYMENT_RESPONSE_ID,
    Feature.ENCODED_ITEM_IDS,
    Feature.INPUT_REWRITE,
    Feature.EXTRA_BODY,
    Feature.PROVIDER_AFFINITY,
    Feature.RESPONSE_CACHE,
    Feature.MANAGED_RESOURCE_IDS,
    Feature.CACHE_CONTROL_MARKS,
    Feature.ANTHROPIC_TOOL_SEARCH,
    Feature.TOOL_SCHEMA_REPAIR,
    Feature.MODEL_PARAM_GATING,
    Feature.WORKLOAD_IDENTITY,
    Feature.CLIENT_HEADERS,
    Feature.DEPLOYMENT_HOOK,
}

RULES: Final[Rules] = (
    RouteRule(Route.CHAT_COMPLETIONS, Rollout.PYTHON_ONLY, features=CHAT_PYTHON_FEATURES),
    RouteRule(Route.CHAT_COMPLETIONS, Rollout.PYTHON_ONLY),
    RouteRule(Route.EMBEDDINGS, Rollout.PYTHON_ONLY),
    RouteRule(Route.OCR, Rollout.RUST_REQUIRED),
    RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, targets=frozenset({Target.CHAT_COMPLETIONS, Target.RESPONSES})),
    RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY, features=MESSAGES_PYTHON_FEATURES),
    RouteRule(Route.MESSAGES, Rollout.RUST_OPT_IN, providers=frozenset({"anthropic"})),
    RouteRule(Route.MESSAGES, Rollout.PYTHON_ONLY),
    RouteRule(Route.RESPONSES, Rollout.PYTHON_ONLY, features=RESPONSES_PYTHON_FEATURES),
    RouteRule(Route.RESPONSES, Rollout.PYTHON_ONLY),
    RouteRule(Route.TOKEN_COUNTER, Rollout.PYTHON_ONLY),
    RouteRule(Route.TOKENIZER, Rollout.PYTHON_ONLY),
    RouteRule(Route.TRANSCRIPTION, Rollout.RUST_REQUIRED, providers=frozenset({"bedrock"})),
)


def rollout(context: RouteContext, rules: Rules | None = None) -> Rollout:
    selected_rules: Final = RULES if rules is None else rules
    return next((rule.rollout for rule in selected_rules if rule.matches(context)), Rollout.PYTHON_ONLY)


def decision(context: RouteContext, rules: Rules | None = None) -> Decision:
    return _decision(rollout(context, rules))


def rust_might_handle(route: Route, rules: Rules) -> bool:
    own: Final = tuple(rule for rule in rules if rule.route is route)
    last: Final = next((i for i, rule in enumerate(own) if rule.unconstrained()), len(own))
    return any(_decision(rule.rollout) is not Decision.PYTHON for rule in own[: last + 1])
