"""An SDK caller that reuses one ``metadata=`` dict across ``litellm.completion`` calls to a Bedrock deployment whose
``guardrailConfig`` has ``trace: enabled`` is billed, on each call, for that call's guardrail trace only, and the
caller's dict is left as the caller wrote it (LIT-10239)"""

import json
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support.client import eventually
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

import litellm

_MODEL: Final = "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0"
_PROMPT_TOKENS: Final = 5
_COMPLETION_TOKENS: Final = 1
_USAGE: Final[Mapping[str, int]] = {"contentPolicyUnits": 1, "topicPolicyUnits": 1}
_GUARDRAIL_IDS: Final = ("guardrail-one", "guardrail-two")


def _bedrock(request: Request) -> Reply:
    """Bedrock's Converse runtime: a one-token answer and the trace of the guardrail the request named."""
    guardrail_id: Final = json.loads(request.body)["guardrailConfig"]["guardrailIdentifier"]
    body: Final[dict[str, JsonValue]] = {
        "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": _PROMPT_TOKENS, "outputTokens": _COMPLETION_TOKENS, "totalTokens": 6},
        "metrics": {"latencyMs": 3},
        "trace": {"guardrail": {"inputAssessment": {guardrail_id: {"invocationMetrics": {"usage": dict(_USAGE)}}}}},
    }
    return Reply(body=json.dumps(body).encode())


def _guardrail_cost() -> float:
    rates: Final = litellm.model_cost["bedrock/guardrails"]["guardrail_cost_per_unit"]
    return sum(units * rates[counter] for counter, units in _USAGE.items())


def _llm_cost() -> float:
    prices: Final = litellm.model_cost[_MODEL.removeprefix("bedrock/")]
    return _PROMPT_TOKENS * prices["input_cost_per_token"] + _COMPLETION_TOKENS * prices["output_cost_per_token"]


def test_each_call_sharing_one_metadata_dict_is_billed_its_own_trace_only() -> None:
    billed: list[float] = []  # mutable-ok: the success callback appends what each call was billed
    litellm.success_callback.append(
        lambda kwargs, response, start, end: billed.append(kwargs["standard_logging_object"]["response_cost"])
    )
    shared_metadata: Final = {"tag": "x"}
    with wire_server(_bedrock) as wire:
        for guardrail_id in _GUARDRAIL_IDS:
            litellm.completion(
                model=_MODEL,
                messages=[{"role": "user", "content": "hello"}],
                api_base=wire.url,
                aws_access_key_id="AKIASCRIPTED",
                aws_secret_access_key="scripted",
                aws_region_name="us-east-1",
                guardrailConfig={"guardrailIdentifier": guardrail_id, "guardrailVersion": "1", "trace": "enabled"},
                metadata=shared_metadata,
            )
    costs: Final = eventually(lambda: tuple(billed), lambda seen: len(seen) == len(_GUARDRAIL_IDS), seconds=30)

    assert costs == pytest.approx((_llm_cost() + _guardrail_cost(),) * len(_GUARDRAIL_IDS))
    assert shared_metadata == {"tag": "x"}
