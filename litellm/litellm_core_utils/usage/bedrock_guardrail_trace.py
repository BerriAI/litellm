"""What a Bedrock guardrail trace reports: the guardrail section of a Converse or Invoke response and the units
its assessments used."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import object_mapping

BEDROCK_TRACE_FIELDS: Final = ("trace", "amazon-bedrock-trace")
_INTERVENING_ACTIONS: Final = frozenset({"BLOCKED", "ANONYMIZED"})


def _trace_field(response: object, field: str) -> Mapping[str, object] | None:
    source: Final = object_mapping(response)
    trace: Final = object_mapping(source.get(field) if source is not None else getattr(response, field, None))
    if trace is None or object_mapping(trace.get("guardrail")) is None:
        return None
    return trace


def bedrock_guardrail_trace(response: object) -> Mapping[str, object] | None:
    """The guardrail trace a Bedrock response carries: Converse puts it under ``trace``, Invoke under
    ``amazon-bedrock-trace``. Returned as is, or None when the response has no guardrail section."""
    traces: Final = map(partial(_trace_field, response), BEDROCK_TRACE_FIELDS)
    return next((trace for trace in traces if trace is not None), None)


class _Assessment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    invocation_metrics: Mapping[str, JsonValue] = Field(default_factory=dict, alias="invocationMetrics")

    @property
    def billable_usage(self) -> Mapping[str, int]:
        usage: Final = self.invocation_metrics.get("usage")
        if not isinstance(usage, dict):
            return {}
        return {
            counter: units
            for counter, units in usage.items()
            if counter.endswith("Units") and isinstance(units, int) and not isinstance(units, bool)
        }


class _GuardrailTrace(BaseModel):
    """Both trace shapes: Converse ``inputAssessment`` {id: a} and ``outputAssessments`` {id: [a, ...]}, Invoke
    ``input`` {id: a} and ``outputs`` [{id: a}, ...]."""

    model_config = ConfigDict(frozen=True, extra="allow")
    input_assessment: Mapping[str, _Assessment] = Field(default_factory=dict, alias="inputAssessment")
    input: Mapping[str, _Assessment] = Field(default_factory=dict)
    output_assessments: Mapping[str, tuple[_Assessment, ...]] = Field(default_factory=dict, alias="outputAssessments")
    outputs: tuple[Mapping[str, _Assessment], ...] = ()

    @property
    def assessments(self) -> tuple[tuple[str, _Assessment], ...]:
        listed: Final = (
            (gid, a) for gid, items in self.output_assessments.items() for a in items
        )  # comprehension-ok: id then its list
        return (
            *self.input_assessment.items(),
            *self.input.items(),
            *listed,
            *chain.from_iterable(by_id.items() for by_id in self.outputs),
        )


_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _summed_usage(usages: Sequence[Mapping[str, int]]) -> Mapping[str, int]:
    counters: Final = tuple(dict.fromkeys(chain.from_iterable(usages)))
    return {counter: sum(usage.get(counter, 0) for usage in usages) for counter in counters}


def _actions(node: JsonValue) -> Iterator[str]:
    if isinstance(node, list):
        yield from chain.from_iterable(map(_actions, node))
        return
    if not isinstance(node, dict):
        return
    action: Final = node.get("action")
    if isinstance(action, str):
        yield action
    yield from chain.from_iterable(map(_actions, node.values()))


@dataclass(frozen=True, slots=True)
class BedrockGuardrailUsage:
    """A guardrail trace's section, its first assessment's guardrail id, every assessment's billable units summed,
    and whether any action in it intervened."""

    guardrail: Mapping[str, JsonValue]
    first_guardrail_id: str
    usage: Mapping[str, int]
    intervened: bool


def bedrock_guardrail_usage(trace: Mapping[str, object]) -> BedrockGuardrailUsage | None:
    """The units a trace's assessments used, None when its guardrail section does not validate or reports none."""
    try:
        guardrail: Final = _JSON_OBJECT.validate_python(trace.get("guardrail"))
        pairs: Final = _GuardrailTrace.model_validate(guardrail).assessments
    except ValidationError:
        return None
    usage: Final = _summed_usage(tuple(assessment.billable_usage for _, assessment in pairs))
    if not usage:
        return None
    return BedrockGuardrailUsage(
        guardrail=guardrail,
        first_guardrail_id=pairs[0][0],
        usage=usage,
        intervened=any(action in _INTERVENING_ACTIONS for action in _actions(guardrail)),
    )
