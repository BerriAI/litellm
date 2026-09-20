"""Every built-in that has a v1 port, its legacy twin, how far the port has come, and the
gaps it is waiting on.

This whole module is migration scaffolding, and it is all of it: `port.py` and the ports
themselves carry none. Read by tests only. Nothing here registers a callback or decides
what runs in a call -- a port goes live by removing its legacy twin, not by a switch in
this table -- and when the last twin is gone this file is deleted outright.

The registration name lives here rather than on the port because it is the host's: a port
is a value, and the same one configured twice is two subscribers under two names.

`python -m litellm.callbacks_v1.builtin.manifest` prints the gap ledger and the backlog.
"""

from dataclasses import dataclass
from types import ModuleType
from typing import Final, Literal, TypeAlias

from litellm.callbacks_v1.builtin import anthropic_cache_control, generic_api, openmeter

Kind: TypeAlias = Literal["sink", "interceptor"]
GapKind: TypeAlias = Literal["fact", "event", "capability"]

# exploring: the port exists and its payload tests pass.
# parity:    its differential test against the legacy twin passes, every difference explained.
# ready:     parity, no open gap, and every route the legacy twin serves emits v1 envelopes.
Status: TypeAlias = Literal["exploring", "parity", "ready"]


@dataclass(frozen=True, slots=True)
class Gap:
    """One legacy input the v1 contract does not supply.

    `fact` is a value missing from an envelope, `event` is a moment no envelope marks,
    and `capability` is something the contract does not let a callback do at all.

    `key` names what the contract would have to grow, not what the legacy logger read, so
    two ports missing the same thing share a key and the ledger can count them. It is also
    what an `Allowed` parity difference cites, which is how closing a gap finds its tests.
    """

    kind: GapKind
    key: str
    legacy: str
    note: str


@dataclass(frozen=True, slots=True)
class Entry:
    """One port: where it lives, what it registers as, what it replaces, and where it has got to."""

    module: ModuleType
    name: str
    kind: Kind
    legacy: str
    status: Status
    gaps: tuple[Gap, ...]


OPENMETER_GAPS: Final = (
    Gap("fact", "cost", "kwargs['response_cost']", "no cost fact on call.succeeded; `data.cost` is always null"),
    Gap(
        "fact",
        "usage",
        "response_obj['usage'] of a ModelResponse",
        "token counts are only reachable by parsing `response` per provider shape; needs a usage fact",
    ),
    Gap(
        "fact",
        "request_user",
        "kwargs['user']",
        "the request's `user` is not projected; subject comes from metadata only",
    ),
    Gap(
        "fact",
        "litellm_metadata",
        "litellm_params['metadata'] fed by `litellm_metadata`",
        "call.started projects only the `metadata` keyword; the proxy's identity on Messages-style routes is invisible",
    ),
    Gap(
        "fact",
        "terminal_model",
        "kwargs['model']",
        "the terminal envelope has no model, so the port joins request.sending by call_id",
    ),
    Gap(
        "capability",
        "off_path_delivery",
        "Logging runs success callbacks on an executor / the logging worker",
        "v1 observers run inline in the call; the port needs its own outbox to stay off the call path",
    ),
)

ANTHROPIC_CACHE_CONTROL_GAPS: Final = (
    Gap(
        "fact",
        "request_params",
        "kwargs['cache_control_injection_points']",
        "per-request injection points are not in RequestFacts; the port only knows the points it was built with",
    ),
    Gap(
        "fact",
        "request_call_type",
        "call_type",
        "RequestFacts has no call_type, so the port recognises a Messages request by the body's shape",
    ),
    Gap(
        "capability",
        "route_neutral_rewrite",
        "get_chat_completion_prompt rewrites the route-neutral messages before transformation",
        "before_send sees only the provider's wire body, so every dialect needs its own body walker",
    ),
)

# What the legacy StandardLoggingPayload has and v1 facts cannot fill, by what the
# contract would have to grow to fill it.
_MISSING_FACTS: Final = (
    ("cost", "response_cost, cost_breakdown, saved_cache_cost, autorouter_savings", "no cost facts"),
    ("usage", "total_tokens, prompt_tokens, completion_tokens", "no usage fact; only the raw provider response"),
    ("cache", "cache_hit, cache_key", "no cache fact"),
    (
        "routing",
        "model_id, model_group, api_base, model_map_information",
        "no routing fact; `url` is the wire URL, not api_base",
    ),
    ("identity", "metadata.user_api_key_*, end_user, requester_ip_address, user_agent", "no identity fact"),
    ("tags", "request_tags, request_model_access_groups", "no tags fact"),
    ("correlation", "trace_id, session_id", "no correlation beyond call_id"),
    ("first_token_time", "completionStartTime", "no first-token time in `timing`"),
    ("no_equivalent", "hidden_params, guardrail_information, standard_built_in_tools_params", "no equivalent"),
    ("prompt", "messages", "only the provider-shaped wire body; no route-neutral prompt projection"),
    (
        "normalised_response",
        "response",
        "legacy logs the OpenAI-normalised ModelResponse; v1 carries the route's own public response",
    ),
    (
        "error_detail",
        "error_information.traceback, llm_provider, error_provider_request_id",
        "ErrorFacts is class/message/status only",
    ),
)

GENERIC_API_GAPS: Final = (
    *(Gap("fact", key, f"standard_logging_object: {fields}", note) for key, fields, note in _MISSING_FACTS),
    Gap(
        "fact",
        "terminal_model",
        "standard_logging_object: model, custom_llm_provider, model_parameters",
        "the terminal envelope has none of them, so the port joins request.sending by call_id",
    ),
    Gap(
        "capability",
        "off_path_delivery",
        "CustomBatchLogger.periodic_flush on the proxy's event loop",
        "v1 observers run inline in the call; batching and I/O live on the port's outbox thread",
    ),
)

PORTS: Final = (
    Entry(
        openmeter,
        "openmeter",
        "sink",
        "litellm.integrations.openmeter:OpenMeterLogger",
        "parity",
        OPENMETER_GAPS,
    ),
    Entry(
        generic_api,
        "generic_api",
        "sink",
        "litellm.integrations.generic_api.generic_api_callback:GenericAPILogger",
        "parity",
        GENERIC_API_GAPS,
    ),
    Entry(
        anthropic_cache_control,
        "anthropic_cache_control",
        "interceptor",
        "litellm.integrations.anthropic_cache_control_hook:AnthropicCacheControlHook",
        "parity",
        ANTHROPIC_CACHE_CONTROL_GAPS,
    ),
)


def entry(name: str) -> Entry:
    """The manifest entry a port registers under; how a test reaches its gaps."""
    (found,) = (candidate for candidate in PORTS if candidate.name == name)
    return found


def ledger() -> str:
    """Every open gap, grouped by port: what the contract has to grow before ports can be `ready`."""
    return "\n".join(
        line
        for port in PORTS
        for line in (
            f"{port.name} [{port.kind}, {port.status}] <- {port.legacy}",
            *(f"  {gap.kind:<10} {gap.key}: {gap.legacy}\n             {gap.note}" for gap in port.gaps),
        )
    )


def backlog() -> str:
    """Every gap key with the ports that are waiting on it, most wanted first: the order the contract should grow in."""
    keys: Final = frozenset(gap.key for port in PORTS for gap in port.gaps)
    waiting: Final = tuple(
        (key, tuple(port.name for port in PORTS if any(gap.key == key for gap in port.gaps))) for key in sorted(keys)
    )
    return "\n".join(
        f"  {key:<22} {', '.join(names)}" for key, names in sorted(waiting, key=lambda item: -len(item[1]))
    )


if __name__ == "__main__":
    print(f"{ledger()}\n\nbacklog, by how many ports wait on it\n{backlog()}")  # noqa: T201  # the ledger is this module's command-line output
