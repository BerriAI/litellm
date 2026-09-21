"""Typed view of the scalar keyword payload guardrails forward to ``CustomGuardrail.__init__``.

Guardrail subclasses collect their base-class options in ``**kwargs`` and splat them into
``super().__init__``. Declaring the payload's shape here lets the checker resolve each
forwarded argument to its real parameter type instead of ``Any``.
"""

from typing_extensions import ReadOnly, TypedDict


class GuardrailBaseInitKwargs(TypedDict, total=False):
    guardrail_name: ReadOnly[str | None]
    default_on: ReadOnly[bool]
    mask_request_content: ReadOnly[bool]
    mask_response_content: ReadOnly[bool]
    violation_message_template: ReadOnly[str | None]
    end_session_after_n_fails: ReadOnly[int | None]
    on_violation: ReadOnly[str | None]
    realtime_violation_message: ReadOnly[str | None]
    on_sensitive_data: ReadOnly[str | None]
    sensitive_data_route_to_model: ReadOnly[str | None]
    sticky_session_routing: ReadOnly[bool]
    run_in_parallel: ReadOnly[bool]
    only_scan_new_messages: ReadOnly[bool]
