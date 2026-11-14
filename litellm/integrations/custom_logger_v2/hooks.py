from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from litellm.integrations.custom_logger_v2.enums import DERIVED_ONLY, Activation, Field, Hook
from litellm.integrations.custom_logger_v2.outcomes import Abort, Reroute, Respond

LegacyGuardrailEvent: TypeAlias = Literal["pre_call", "during_call", "post_call", "post_mcp_call"]
OutcomeType: TypeAlias = type[Abort] | type[Respond] | type[Reroute]


class InvalidDeclarationError(TypeError):
    pass


@dataclass(frozen=True, slots=True)
class HookSpec:
    legacy_methods: tuple[str, ...]
    request_path: bool
    writes: frozenset[Field] = frozenset()
    outcomes: frozenset[OutcomeType] = frozenset()
    legacy_guardrail_events: frozenset[LegacyGuardrailEvent] = frozenset()
    has_call: bool = True
    subscribable: bool = True
    fail_fast_parallel: bool = False


_NO_OUTCOME: Final[frozenset[OutcomeType]] = frozenset()
_ABORT: Final[frozenset[OutcomeType]] = frozenset({Abort})
_ABORT_OR_RESPOND: Final[frozenset[OutcomeType]] = frozenset({Abort, Respond})
_ANY_OUTCOME: Final[frozenset[OutcomeType]] = frozenset({Abort, Respond, Reroute})

HOOK_SPECS: Final[Mapping[Hook, HookSpec]] = MappingProxyType(
    {
        Hook.prompt_resolve: HookSpec(
            legacy_methods=("get_chat_completion_prompt", "async_get_chat_completion_prompt"),
            request_path=True,
            writes=frozenset({Field.messages, Field.optional_params, Field.model}),
            outcomes=_ABORT_OR_RESPOND,
        ),
        Hook.pre_route: HookSpec(
            legacy_methods=("async_pre_routing_hook",),
            request_path=True,
            writes=frozenset({Field.model, Field.messages}),
            outcomes=_ABORT_OR_RESPOND,
        ),
        Hook.filter_deployments: HookSpec(
            legacy_methods=("async_filter_deployments",),
            request_path=True,
            writes=frozenset({Field.candidate_deployments}),
            outcomes=_ABORT,
        ),
        Hook.pre_call: HookSpec(
            legacy_methods=("async_pre_call_hook",),
            request_path=True,
            writes=frozenset(
                {
                    Field.messages,
                    Field.optional_params,
                    Field.metadata,
                    Field.request_headers,
                    Field.guardrail_info,
                    Field.tools,
                    Field.model,
                    Field.input,
                    Field.budget_reservation,
                    Field.rate_limit_statuses,
                }
            ),
            outcomes=_ANY_OUTCOME,
            legacy_guardrail_events=frozenset({"pre_call"}),
        ),
        Hook.during_call: HookSpec(
            legacy_methods=("async_moderation_hook",),
            request_path=True,
            outcomes=_ABORT_OR_RESPOND,
            legacy_guardrail_events=frozenset({"during_call"}),
            fail_fast_parallel=True,
        ),
        Hook.pre_deployment_call: HookSpec(
            legacy_methods=("async_pre_call_deployment_hook", "async_pre_request_hook"),
            request_path=True,
            writes=frozenset({Field.provider_request_body, Field.request_headers}),
            outcomes=_ABORT_OR_RESPOND,
        ),
        Hook.post_deployment_success: HookSpec(
            legacy_methods=("async_post_call_success_deployment_hook", "async_post_call_streaming_deployment_hook"),
            request_path=True,
            writes=frozenset({Field.response}),
            outcomes=_ABORT_OR_RESPOND,
        ),
        Hook.post_deployment_failure: HookSpec(
            legacy_methods=("async_post_call_failure_deployment_hook",),
            request_path=True,
            writes=frozenset({Field.error}),
            outcomes=_ABORT,
        ),
        Hook.post_call_success: HookSpec(
            legacy_methods=("async_post_call_success_hook",),
            request_path=True,
            writes=frozenset({Field.response, Field.guardrail_info}),
            outcomes=_ABORT_OR_RESPOND,
            legacy_guardrail_events=frozenset({"post_call"}),
        ),
        Hook.post_call_failure: HookSpec(
            legacy_methods=("async_post_call_failure_hook",),
            request_path=True,
            writes=frozenset({Field.error}),
            outcomes=_ABORT,
        ),
        Hook.response_headers: HookSpec(
            legacy_methods=("async_post_call_response_headers_hook",),
            request_path=True,
            writes=frozenset({Field.response_headers}),
            outcomes=_NO_OUTCOME,
        ),
        Hook.stream_chunk: HookSpec(
            legacy_methods=(
                "async_post_call_streaming_hook",
                "async_post_call_streaming_iterator_hook",
                "log_stream_event",
            ),
            request_path=True,
            writes=frozenset({Field.stream_chunk}),
            outcomes=_ABORT,
            subscribable=False,
        ),
        Hook.list_models: HookSpec(
            legacy_methods=("async_filter_listed_models",),
            request_path=True,
            writes=frozenset({Field.model_list}),
            outcomes=_ABORT,
            has_call=False,
        ),
        Hook.mcp_tool_result: HookSpec(
            legacy_methods=("async_post_mcp_tool_call_hook",),
            request_path=True,
            writes=frozenset({Field.tool_result}),
            outcomes=_ABORT_OR_RESPOND,
            legacy_guardrail_events=frozenset({"post_mcp_call"}),
        ),
        Hook.log_prepare: HookSpec(
            legacy_methods=(
                "logging_hook",
                "async_logging_hook",
                "redact_standard_logging_payload_from_model_call_details",
            ),
            request_path=False,
            writes=frozenset({Field.standard_logging_payload}),
        ),
        Hook.provider_request_sent: HookSpec(legacy_methods=("log_pre_api_call",), request_path=False),
        Hook.provider_response_received: HookSpec(legacy_methods=("log_post_api_call",), request_path=False),
        Hook.attempt_ended: HookSpec(
            legacy_methods=(
                "log_success_event",
                "log_failure_event",
                "async_log_success_event",
                "async_log_failure_event",
            ),
            request_path=False,
        ),
        # Not dispatched: the five ports settle on attempt_ended and none needs a once-per-call end event.
        Hook.call_ended: HookSpec(legacy_methods=(), request_path=False),
        Hook.router_event: HookSpec(
            legacy_methods=(
                "log_success_fallback_event",
                "log_failure_fallback_event",
                "log_model_group_rate_limit_error",
            ),
            request_path=False,
        ),
        Hook.audit_event: HookSpec(
            legacy_methods=("async_log_audit_log_event", "async_log_batch_line_items"),
            request_path=False,
            has_call=False,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class Subscription:
    hook: Hook
    handler: str
    reads: frozenset[Field] = frozenset()
    writes: frozenset[Field] = frozenset()
    may_abort: bool = False
    parallel: bool = False
    awaited: bool = False
    activation: Activation | None = None
    legacy: bool = False

    def __post_init__(self) -> None:
        problem: Final = subscription_problem(self)
        if problem is not None:
            raise InvalidDeclarationError(f"{self.handler} on {self.hook.value}: {problem}")


def subscription_problem(subscription: Subscription) -> str | None:
    spec: Final = HOOK_SPECS[subscription.hook]
    if not spec.subscribable:
        return "this hook point has no V2 subscription support yet"
    derived: Final = subscription.writes & DERIVED_ONLY
    if derived:
        return (
            f"writes {', '.join(sorted(field.value for field in derived))}, "
            "which are derived from the hook's arguments and never writable"
        )
    legacy_writes: Final = frozenset({Field.deprecated_kwargs}) if subscription.legacy else frozenset[Field]()
    forbidden: Final = subscription.writes - spec.writes - legacy_writes
    if forbidden:
        return f"writes {', '.join(sorted(field.value for field in forbidden))}, which this hook point cannot apply"
    if subscription.may_abort and not spec.outcomes:
        return "may_abort is set, but this hook point allows no outcome"
    if subscription.awaited and (subscription.writes or subscription.may_abort):
        return "an awaited subscription must neither write nor abort"
    if subscription.parallel and (subscription.writes or not subscription.may_abort):
        return "parallel is only for subscriptions that may abort and write nothing"
    return None
