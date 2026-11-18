from __future__ import annotations

from typing import Final

from cases.large_bodies_300_users import CASE
from pydantic import TypeAdapter
from setup.corpus.session import request_body
from setup.plan import build_plan

_MESSAGES: Final = TypeAdapter(list[dict[str, object]])


def test_large_body_plan_keeps_every_request_within_the_default_response_cache_message_limit() -> None:
    plan: Final = build_plan(CASE)
    assert len(plan) == 1_200
    bodies: Final = tuple(
        request_body(
            endpoint="chat",
            model=item.model_group,
            marker=item.marker,
            input_tokens=item.input_tokens,
            tools=item.shape.tools,
            output_tokens=item.shape.output_tokens,
            stream=item.stream,
        )
        for item in plan
    )
    messages: Final = tuple(_MESSAGES.validate_python(body["messages"]) for body in bodies)
    assert all(2 <= len(turns) <= 4 for turns in messages)
    assert all(item.marker in str(turns[-1]["content"]) for item, turns in zip(plan, messages, strict=True))
    assert all(body["max_completion_tokens"] == 400_000 for body in bodies)
