"""bedrock_converse cost tracking cases, one CostTrackingTestCase literal per request shape (moved from cost_tracking_cases.json).

PARITY pairs the streamed case with its plain twin where both bill the same row."""

from typing import Final

from integration.cost_calculation.cost_tracking_case import (
    CostTrackingTestCase,
    Deployment,
    EventStreamEvent,
    EventStreamResponse,
    ExactExpected,
    JsonResponse,
    RecountExpected,
    RecountRates,
)
from integration.cost_calculation.stream_parity.case import StreamParityTestCase


# anthropic.claude-sonnet-5-v1:0
ANTHROPIC_CLAUDE_SONNET_5_V1_0_INPUT_TEXT: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "9aad4de0556c summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 9aad4de0556c"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.01287, input_cost=0.006072, output_cost=0.006798, prompt_tokens=1840, completion_tokens=412
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_CACHE_READ: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "708bfb28f35a summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 708bfb28f35a"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 640, "outputTokens": 380, "totalTokens": 13308, "cacheReadInputTokens": 12288},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.01243704,
        input_cost=0.00616704,
        output_cost=0.00627,
        prompt_tokens=12928,
        completion_tokens=380,
        cache_read_cost=0.00405504,
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_CACHE_WRITE_5M: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-cache_write_5m",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "08c49d1b837c summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 08c49d1b837c"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 512,
                "outputTokens": 350,
                "totalTokens": 10078,
                "cacheWriteInputTokens": 9216,
                "cacheDetails": [{"inputTokens": 9216, "ttl": "5m"}],
            },
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.0454806,
        input_cost=0.0397056,
        output_cost=0.005775,
        prompt_tokens=9728,
        completion_tokens=350,
        cache_creation_cost=0.038016,
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_CACHE_WRITE_1H: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-cache_write_1h",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "b96166d8affb summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer b96166d8affb"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 512,
                "outputTokens": 350,
                "totalTokens": 10078,
                "cacheWriteInputTokens": 9216,
                "cacheDetails": [{"inputTokens": 2048, "ttl": "5m"}, {"inputTokens": 7168, "ttl": "1h"}],
            },
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.0632214,
        input_cost=0.0574464,
        output_cost=0.005775,
        prompt_tokens=9728,
        completion_tokens=350,
        cache_creation_cost=0.0557568,
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_SERVICE_TIER_FLEX: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-service_tier_flex",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "41dbeb5496b2 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "service_tier": "flex",
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 41dbeb5496b2"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
            "serviceTier": {"type": "flex"},
        },
    ),
    expected=ExactExpected(
        spend=0.006435, input_cost=0.003036, output_cost=0.003399, prompt_tokens=1840, completion_tokens=412
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "f6c242da055f summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "service_tier": "priority",
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer f6c242da055f"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
            "serviceTier": {"type": "priority"},
        },
    ),
    expected=ExactExpected(
        spend=0.0160875, input_cost=0.00759, output_cost=0.0084975, prompt_tokens=1840, completion_tokens=412
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "a9257967d38a summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer a9257967d38a"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
                    "metrics": {"latencyMs": 42},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.01287, input_cost=0.006072, output_cost=0.006798, prompt_tokens=1840, completion_tokens=412
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "ca62b8bbf5b6 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer ca62b8bbf5b6"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=3.3e-06, output_cost_per_token=1.65e-05)),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "6f00980cd47c summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockStart",
                payload={
                    "start": {"toolUse": {"toolUseId": "tooluse_$REQUEST_ID", "name": "get_weather"}},
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil'
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi"
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}'
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "tool_use"}),
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=3.3e-06, output_cost_per_token=1.65e-05)),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "2a972e053197 summarize the attached material in one line and name the city weather",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAMK0lEQVR4nAEgDN/zAM0HLNi+b59irEwJwoIG5+NVlKprNC9dCjpeSEL6tCj3YubiguXBZXx4w6lns2cR6zkGp8hgPXHUCeelTYe9wfcEQgJ6rx+pW3+GWJV430PkExZ66Nnc6zd2KDOBGnGnIwBzhiZIL2HGI3lifMEk1EYYPG5NnqGlpcz3LiFAwwS9/Goh5egSF1aINdSX+yErhrSeZWrPBkEWmgtZ9OYpQ58l2dRlT+yNSBn7QNa6ssjgEhxEGuYUp7jZKpIZrx7Oh1QAV1jeeB7zT4/0jccZhxGSWqLiJW9VRPJQuRZjnL/J8qO8F7vpSKdYNMGDc/cEMXKNBlAdehNaVHGe84Tdmm53hcOfr0ICjvEPuk0WzuaDIOumjneMSZ1+6qg8mAM8quAXAOyQPrjRNxDXsUwZZhYr07VMCinT0OP4yIEWDKvlaxGgReVKAJ1JpZx/Hlt+evT70yo3G94iWEhVavFwPnqJ87rKl0BT6+ohtOgz197MvB8QzMXpME/BwepPYkiRLpbBOAD37hU9bAWozdK3sPczg3okfSudzcFjAYtEIq5yw+1ZF9EYmBSexEP+IhnvUWC4BeDWZQiCjhF7/4syzu4C5kF9ETfrG+udK0232R+NA+uESn014bRJmPMfahYljDojL1UAbuaA0PuOGOzBBlCKXAkFNHIfvvZHQafMkl9qmqdFF4x3Em6WDuijSQXN6nFtM3UXbUGmmCF4Rcyl4YhifPspUXLdXZMIvPo9zAhTSlQFEi8m83sw5KxL0rWBzS9c4XAIAOaz3py4dTb7d9QaqDMLk0J87/15PZKvEaC6/hZ62sCtuFTywatjViHrBXTgOO5IJsiyYuzmaeQJKHmr11gnixR2rO7lqNEGs3shT+xK/lDUucFkigvA+a5C+itk/VV+1gDxc420UHpKhj31j0YnCZSFJebGz2/XSTyT6XfZhG4XNwZGIeUGCPKt0DX9lpB0RNN0yiPzQVJfa3LkZpRBN3RGgRpYc8WpHn5Q1wWpmnklpPHACv/N+kGz0Ki86g2GgvsAWloXy5pwfFtwZRYV9fMGU4Za35yfj4cddJuHfAyHSpYHVlGhNknUVQIBV9gOq7wwLZU3Pl5GJgSy4EK7f75iRVKD/B0ItpC0FBpwODhWP184ymnLgLGkK90WIVUY7hZtAEOu39BF2OsPDWrBGWN0esj6v3clX2z22gaLmrJHiwE4sXWUC8TLLtFp4uiS/VlbojLP9uiewb/vrTLBiFjYJ5rsFjuu/3XxEuiZ1QYyi9sfsFqPoiXjQjCA/jibX4aA1AAfp3GT1lykHtVMJmSxoW4Xvn3BXhXUdtWhIwP7NDO1HRP9UAlEUgKbd/iJBWS20DFBJQb2/UN/+CpSWi97Rta3+pe3H4kOr3qaV+g1Udgmupu6/cxmQqMP+DXd77Sy6a0AMxTVBR0DU4sVW/Vs2qPfnh3r+xlqt/3VIhyKQknN6xF5RIg4j7xsEpjsnKV/XhJNmN2sWekxom9lUCkuP3qgD25S7oCG6ZV3CrkUCm48s5hw+dUZANcGs4H6/PxIrSpkAC77CDPFF5hCvkfKWzerhufLBkq7AhZgeMKRnNboaP3m9qMh6+9Z3JEyaV8rfUacshIsMqwS/BI0uL9v9+fwcMQrbdwOstrkyWCPG61dXIAoEb9t2HnSdSlxy6FXa5+LhwCvCy1ANKkBHgVSx5hoE+LrBX47caskZapZ+MAsTFJhA3VxvHgKaWiuX4/vaG7TbOZdX7GRSzPz3yOeM4JeAuLqwOy6T0OBIKbnSm5b3A5+Y2j2cNYMiVmogx89QCIPRicA936Dj6rC2bDKBi8DmXw8dZvR170ELj4UjqD+VRopML3ww7ILyoVYi5P150ekt4QioS95PZdaHcPXSAD0GwVZe1t0K1qy2TGc7V2ySUwrZKwEnPRbLXAcl6praPJ8aVbkAJ1MPaI09ZDaZOT+npRQ4SFv1DK3halvT/cYVWNFxZy/HEwXatv4MtSF+5ymHEWqFC/kYwCJMTaYtzI7Ma9Q1rJ1WZtVBP36KFFdSj1G8Bw5b5ssozn/uHJxFO9g937ZtQBQvxvgA4h8rApfcpEHs+HfrYkWalhdEwiJ+fpmF/Um3ywbq7MF3kWROuUQa4H2rcVhq4WpdGuBte/A+QvHrGkq45kCcmvaWhAAslxCV5CWs7QlXig29URyTAgPh//Ji+IAJXC9fNNMdeirO7iPEEOemj9zZ8FMiAQAYKRF4o8E9glP+Jx+Vw9xU5oM40/363XW5T+Gd4xtwwwQoR3P38Wd0uURAeDvp2P53Wz6zxpyRmpc2iAwN5BujDYD2v+lpIn2AMUaEqKkrYP6sRheFZ0EY9Yt7r25gt9pITZVoPxxTMUDlHfdZgyMFfPLKLOtUk3gaiv88FD1Wd4AdI2pNnmYqAMKjqK35Ys3wVuBmgAbUPH6k4ae0ku7/qyRrkGHwRewnAAVZQgZavvBMFJ8cB4Ndb2bXEI2phv40TLtxqffTyNrTfbyrEdKKESwv/h/+pkLpi4bcaUZx0fBeRewm9oUozrsfuGNZcn0rKC8DdMUtqqXBWClU0bAHPTp2GD2c1FNwcwAFOXWyuK3ox0nFYLckD+pQW3ZJyKBKlCnanFXhVDRONHgupmm89QapXZ32FhAzncd/Xcy34ljiSee3UVHJtnq2YJJwQQVBNQKjuhoCv0YqzT/zVWuTbF5QtRm8I7cC5FQANj9TfJqwOxdaoY98K6RRZGzAezoRsF02Q3PwAowC6vYqUzHv85kEfb7BYqzoXGLm4fLxafznJZ+JhJdtsmV5qQiAxjnFQdrdTtL4KNZCXlvtdVYXxLGFN9os7WJoU2kLQB5RTU1rvAlYHfbTZZTTYFKXxRHAizqcSNldKkms7p4RYVAWlUXIGdWTb4kym3qAF4clO+3Njf9F3zxl1b73YfH8pXbnk9fIQnHRowK9vQFRcfD8ikeIUAmZrhe+3cOXZUAt7EeSjZgZFxWFhFll/ZC/Y93aYwMJjAhvbgcS/zZaR0ucmJ663gDu7xgXT3tgQrvb4daDFM10U5HZ/kt2r9m26me5oXdgom65hEkEgfTe03ZFzZnPz7n1fzuGVThCppMAAAyK2pIlNAb40xyGcWNko8VyJ3oghsue2leWHkJ6UpWCdZB13x+QcxkLOr5a8X0zx6N+VcXwNMfGGqlelK0sh1MrRizk4drf0prMW3Q264J3IVJTZ3tx1ULpBi8CnnmrwBeYvfkI5rZJ2W6cOv3bKMqoCpyoNo+gpCZcSVgB8hdzlfMfPl67dL9/IqNo2mBQBaVitoQ4y7KzqHnrBWSxLIyznP3v/VNlzgF7qcPaQiDYhyGAgLfihsZE5nQZLcedY4Apm3dhEOyMUpdzNNyViCP6ykfFn35Ss3JDUS7lfZV47bpxeatoe2dNzU5TGxY9GoVYvhKXmP0WjgOpwv91XPF/x1ty0JRgNGVMwnwxk4PpSpDgGRu/k4jMev9x1vlc3EdAIz+WBCWU23SZoIT3WfyVNpXi7azEQkSB7e1K+fS25uGiNScc4/Nn8BVsUqtwtGhM5nncB7tH0F/k0ngZeGmqEGNJw01v2UYpLQoxwqhCz7/X5it8liVJMxk2kQ3iCGYnwDwfHcl7gLJGiAqrjLBOlp17bPGYIGTlFw+TOCLsGS103TZvSbyptAazsca/TY3o5hZLqX7SBvnfL6s/SU7BVyQY4PpZwHLa6PI2w+uFmJZbBsaruCVPW+FL+zSfvPO560ABJLD6nKsReNBwoxShnNi1XPSmX+C7Ij/ik2nfzubk6IFlomrI0bs386ztTMDifWKNO0ZnC+Oau6ZgwE4PwF8AqKtGFd5yqG0aMFq9rvwPb8DZCXmUHN5Uc40cNG5HWNSAK1yFAkCcG2GPV/7JYEi3t9ZA8NBeh9Dmvsjv8jiaeknPCij14x6BmGOFzyVNqRcS8x517fM39S0cC6bzu8wbnh96fwQ9z/JzCEsqxV9KjuES+xv3rYkX+r3JxcK7p78EACPip85gRBoIU7cZgrwLepMArrpA9YvR8Icag3YILzWcgXVoVRTs6Xc2PsiKasZ98y5CBkmP9Ri3lwZ9acgvPJ1fw729yDzDl/0qEeBOf7ZYZ7dreaPeaGHIymfgKAwmwhnjvAGaqLWhgAAAABJRU5ErkJggg==",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer 2a972e053197"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=3.3e-06, output_cost_per_token=1.65e-05)),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "c300c8153393 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer c300c8153393"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.01287, input_cost=0.006072, output_cost=0.006798, prompt_tokens=1840, completion_tokens=412
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "4c599e93dfba summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer 4c599e93dfba"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
                    "metrics": {"latencyMs": 42},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.01287, input_cost=0.006072, output_cost=0.006798, prompt_tokens=1840, completion_tokens=412
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_TOOL_CALL: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "c40237f9541a summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_$REQUEST_ID",
                                "name": "get_weather",
                                "input": {
                                    "city": "Berlin",
                                    "days": 7,
                                    "units": "metric",
                                    "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler ",
                                },
                            }
                        }
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.01287, input_cost=0.006072, output_cost=0.006798, prompt_tokens=1840, completion_tokens=412
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "1727b8128120 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockStart",
                payload={
                    "start": {"toolUse": {"toolUseId": "tooluse_$REQUEST_ID", "name": "get_weather"}},
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil'
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi"
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}'
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "tool_use"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
                    "metrics": {"latencyMs": 42},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.01287, input_cost=0.006072, output_cost=0.006798, prompt_tokens=1840, completion_tokens=412
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "649a7735f7cb summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer 649a7735f7cb"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {
                        "inputTokens": 1840,
                        "outputTokens": 412,
                        "totalTokens": 11468,
                        "cacheReadInputTokens": 6144,
                        "cacheWriteInputTokens": 3072,
                        "cacheDetails": [{"inputTokens": 2048, "ttl": "5m"}, {"inputTokens": 1024, "ttl": "1h"}],
                    },
                    "metrics": {"latencyMs": 42},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.03010392, input_cost=0.02330592, output_cost=0.006798, prompt_tokens=11056, completion_tokens=412
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_MESSAGES_CACHE_READ: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-messages_cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    endpoint="/v1/messages",
    request={
        "model": "$MODEL",
        "max_tokens": 412,
        "messages": [{"role": "user", "content": "summarize the attached material in one line"}],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted response"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 640, "outputTokens": 380, "totalTokens": 13308, "cacheReadInputTokens": 12288},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.01243704,
        input_cost=0.00616704,
        output_cost=0.00627,
        prompt_tokens=12928,
        completion_tokens=380,
        cache_read_cost=0.00405504,
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_PASSTHROUGH_CONVERSE: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-passthrough-converse",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    endpoint="/bedrock/model/$MODEL/converse",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "9aad4de0556c summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 9aad4de0556c"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.01287,
        input_cost=0.006072,
        output_cost=0.006798,
        prompt_tokens=1840,
        completion_tokens=412,
        cost_header=False,
    ),
)

ANTHROPIC_CLAUDE_SONNET_5_V1_0_PASSTHROUGH_CONVERSE_STREAM: Final = CostTrackingTestCase(
    name="anthropic.claude-sonnet-5-v1:0-passthrough-converse_stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    endpoint="/bedrock/model/$MODEL/converse-stream",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "a9257967d38a summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer a9257967d38a"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
                    "metrics": {"latencyMs": 42},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.01287,
        input_cost=0.006072,
        output_cost=0.006798,
        prompt_tokens=1840,
        completion_tokens=412,
        cost_header=False,
    ),
)

BEDROCK_CONVERSE_PROFILE_BASE_MODEL: Final = CostTrackingTestCase(
    name="bedrock-converse-profile-base-model",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    deployment=Deployment(
        model="bedrock/converse/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123",
        base_model="anthropic.claude-sonnet-5-v1:0",
    ),
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "bedrock-converse-profile-base-model"}],
        "stream": False,
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted response"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 1},
        },
    ),
    expected=ExactExpected(
        spend=0.01287, input_cost=0.006072, output_cost=0.006798, prompt_tokens=1840, completion_tokens=412
    ),
)

BEDROCK_CONVERSE_APAC_BARE_FALLBACK: Final = CostTrackingTestCase(
    name="bedrock-converse-apac-bare-fallback",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="anthropic.claude-sonnet-5-v1:0",
    deployment=Deployment(model="bedrock/converse/apac.anthropic.claude-sonnet-5-v1:0"),
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "bedrock-converse-apac-bare-fallback"}],
        "stream": False,
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted response"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 1},
        },
    ),
    expected=ExactExpected(
        spend=0.01287, input_cost=0.006072, output_cost=0.006798, prompt_tokens=1840, completion_tokens=412
    ),
)


# meta.llama4-maverick-17b-instruct-v1:0
META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_INPUT_TEXT: Final = CostTrackingTestCase(
    name="meta.llama4-maverick-17b-instruct-v1:0-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="meta.llama4-maverick-17b-instruct-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "54c4ce4d8096 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 54c4ce4d8096"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.00084124, input_cost=0.0004416, output_cost=0.00039964, prompt_tokens=1840, completion_tokens=412
    ),
)

META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_FALLBACK_CACHE_READ_AT_INPUT_RATE: Final = CostTrackingTestCase(
    name="meta.llama4-maverick-17b-instruct-v1:0-fallback_cache_read_at_input_rate",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="meta.llama4-maverick-17b-instruct-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "36b591711f22 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 36b591711f22"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 640, "outputTokens": 380, "totalTokens": 13308, "cacheReadInputTokens": 12288},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.00347132, input_cost=0.00310272, output_cost=0.0003686, prompt_tokens=12928, completion_tokens=380
    ),
)

META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_FALLBACK_CACHE_WRITE_AT_INPUT_RATE: Final = CostTrackingTestCase(
    name="meta.llama4-maverick-17b-instruct-v1:0-fallback_cache_write_at_input_rate",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="meta.llama4-maverick-17b-instruct-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "3bd1faf7cebd summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 3bd1faf7cebd"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 512,
                "outputTokens": 350,
                "totalTokens": 10078,
                "cacheWriteInputTokens": 9216,
                "cacheDetails": [{"inputTokens": 9216, "ttl": "5m"}],
            },
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.00267422, input_cost=0.00233472, output_cost=0.0003395, prompt_tokens=9728, completion_tokens=350
    ),
)

META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM: Final = CostTrackingTestCase(
    name="meta.llama4-maverick-17b-instruct-v1:0-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="meta.llama4-maverick-17b-instruct-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "f515db6db1e8 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer f515db6db1e8"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
                    "metrics": {"latencyMs": 42},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.00084124, input_cost=0.0004416, output_cost=0.00039964, prompt_tokens=1840, completion_tokens=412
    ),
)

META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="meta.llama4-maverick-17b-instruct-v1:0-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="meta.llama4-maverick-17b-instruct-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "c423409dd543 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer c423409dd543"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=2.4e-07, output_cost_per_token=9.7e-07)),
)

META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="meta.llama4-maverick-17b-instruct-v1:0-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="meta.llama4-maverick-17b-instruct-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "1b82e406f204 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockStart",
                payload={
                    "start": {"toolUse": {"toolUseId": "tooluse_$REQUEST_ID", "name": "get_weather"}},
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil'
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi"
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}'
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "tool_use"}),
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=2.4e-07, output_cost_per_token=9.7e-07)),
)

META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="meta.llama4-maverick-17b-instruct-v1:0-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="meta.llama4-maverick-17b-instruct-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "52555527573a summarize the attached material in one line and name the city weather",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,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",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer 52555527573a"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=2.4e-07, output_cost_per_token=9.7e-07)),
)

META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="meta.llama4-maverick-17b-instruct-v1:0-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="meta.llama4-maverick-17b-instruct-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "c47a40f71743 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer c47a40f71743"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.00084124, input_cost=0.0004416, output_cost=0.00039964, prompt_tokens=1840, completion_tokens=412
    ),
)

META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="meta.llama4-maverick-17b-instruct-v1:0-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="meta.llama4-maverick-17b-instruct-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "1aa422adaa97 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer 1aa422adaa97"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
                    "metrics": {"latencyMs": 42},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.00084124, input_cost=0.0004416, output_cost=0.00039964, prompt_tokens=1840, completion_tokens=412
    ),
)

META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_TOOL_CALL: Final = CostTrackingTestCase(
    name="meta.llama4-maverick-17b-instruct-v1:0-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="meta.llama4-maverick-17b-instruct-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "2e3593b273a4 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_$REQUEST_ID",
                                "name": "get_weather",
                                "input": {
                                    "city": "Berlin",
                                    "days": 7,
                                    "units": "metric",
                                    "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler ",
                                },
                            }
                        }
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.00084124, input_cost=0.0004416, output_cost=0.00039964, prompt_tokens=1840, completion_tokens=412
    ),
)

META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="meta.llama4-maverick-17b-instruct-v1:0-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="meta.llama4-maverick-17b-instruct-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "3f3df4cdd7d9 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockStart",
                payload={
                    "start": {"toolUse": {"toolUseId": "tooluse_$REQUEST_ID", "name": "get_weather"}},
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil'
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi"
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}'
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "tool_use"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
                    "metrics": {"latencyMs": 42},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.00084124, input_cost=0.0004416, output_cost=0.00039964, prompt_tokens=1840, completion_tokens=412
    ),
)

META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="meta.llama4-maverick-17b-instruct-v1:0-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="meta.llama4-maverick-17b-instruct-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "5eec826ded90 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer 5eec826ded90"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {
                        "inputTokens": 1840,
                        "outputTokens": 412,
                        "totalTokens": 11468,
                        "cacheReadInputTokens": 6144,
                        "cacheWriteInputTokens": 3072,
                        "cacheDetails": [{"inputTokens": 2048, "ttl": "5m"}, {"inputTokens": 1024, "ttl": "1h"}],
                    },
                    "metrics": {"latencyMs": 42},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.00305308, input_cost=0.00265344, output_cost=0.00039964, prompt_tokens=11056, completion_tokens=412
    ),
)


# us.anthropic.claude-opus-5-v1:0
US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_INPUT_TEXT: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "f6559891a89a summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer f6559891a89a"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.02145, input_cost=0.01012, output_cost=0.01133, prompt_tokens=1840, completion_tokens=412
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_CACHE_READ: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-cache_read",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "c054e1cd6b20 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer c054e1cd6b20"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 640, "outputTokens": 380, "totalTokens": 13308, "cacheReadInputTokens": 12288},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.0207284, input_cost=0.0102784, output_cost=0.01045, prompt_tokens=12928, completion_tokens=380
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_CACHE_WRITE_5M: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-cache_write_5m",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "87e62170eee7 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 87e62170eee7"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 512,
                "outputTokens": 350,
                "totalTokens": 10078,
                "cacheWriteInputTokens": 9216,
                "cacheDetails": [{"inputTokens": 9216, "ttl": "5m"}],
            },
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.075801, input_cost=0.066176, output_cost=0.009625, prompt_tokens=9728, completion_tokens=350
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_CACHE_WRITE_1H: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-cache_write_1h",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "4566b7a4b0d6 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 4566b7a4b0d6"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 512,
                "outputTokens": 350,
                "totalTokens": 10078,
                "cacheWriteInputTokens": 9216,
                "cacheDetails": [{"inputTokens": 2048, "ttl": "5m"}, {"inputTokens": 7168, "ttl": "1h"}],
            },
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.105369, input_cost=0.095744, output_cost=0.009625, prompt_tokens=9728, completion_tokens=350
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_TIERED_INPUT_ABOVE_200K: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-tiered_input_above_200k",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "e6889b23c228 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer e6889b23c228"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 204800, "outputTokens": 620, "totalTokens": 205420},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=2.278375, input_cost=2.2528, output_cost=0.025575, prompt_tokens=204800, completion_tokens=620
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_TIERED_CACHE_READ_ABOVE_200K: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-tiered_cache_read_above_200k",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "1d35f19047ff summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 1d35f19047ff"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 4096, "outputTokens": 480, "totalTokens": 206304, "cacheReadInputTokens": 201728},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.2867568, input_cost=0.2669568, output_cost=0.0198, prompt_tokens=205824, completion_tokens=480
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_TIERED_CACHE_WRITE_ABOVE_200K: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-tiered_cache_write_above_200k",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "14f144dc9bee summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 14f144dc9bee"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 4096,
                "outputTokens": 480,
                "totalTokens": 205280,
                "cacheWriteInputTokens": 200704,
                "cacheDetails": [{"inputTokens": 200704, "ttl": "5m"}],
            },
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=2.824536, input_cost=2.804736, output_cost=0.0198, prompt_tokens=204800, completion_tokens=480
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_SERVICE_TIER_FLEX: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-service_tier_flex",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "e984661f7bde summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "service_tier": "flex",
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer e984661f7bde"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
            "serviceTier": {"type": "flex"},
        },
    ),
    expected=ExactExpected(
        spend=0.010725, input_cost=0.00506, output_cost=0.005665, prompt_tokens=1840, completion_tokens=412
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_SERVICE_TIER_PRIORITY: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-service_tier_priority",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "419bc91d93ae summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "service_tier": "priority",
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 419bc91d93ae"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
            "serviceTier": {"type": "priority"},
        },
    ),
    expected=ExactExpected(
        spend=0.0268125, input_cost=0.01265, output_cost=0.0141625, prompt_tokens=1840, completion_tokens=412
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "4a3e5a729480 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer 4a3e5a729480"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
                    "metrics": {"latencyMs": 42},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.02145, input_cost=0.01012, output_cost=0.01133, prompt_tokens=1840, completion_tokens=412
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM_NO_USAGE: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-stream_no_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "45449a962a21 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer 45449a962a21"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=5.5e-06, output_cost_per_token=2.75e-05)),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM_NO_USAGE_TOOL_CALL: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-stream_no_usage_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "e0e1b17ca05f summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockStart",
                payload={
                    "start": {"toolUse": {"toolUseId": "tooluse_$REQUEST_ID", "name": "get_weather"}},
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil'
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi"
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}'
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "tool_use"}),
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=5.5e-06, output_cost_per_token=2.75e-05)),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM_NO_USAGE_IMAGE_INPUT: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-stream_no_usage_image_input",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "7d1ebbfd135c summarize the attached material in one line and name the city weather",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,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",
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer 7d1ebbfd135c"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
        ),
    ),
    expected=RecountExpected(recount=RecountRates(input_cost_per_token=5.5e-06, output_cost_per_token=2.75e-05)),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "80542567b1bb summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted answer 80542567b1bb"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.02145, input_cost=0.01012, output_cost=0.01133, prompt_tokens=1840, completion_tokens=412
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM_RESPONSE_MODEL_OVERRIDE: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-stream_response_model_override",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "ab06cda24199 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer ab06cda24199"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
                    "metrics": {"latencyMs": 42},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.02145, input_cost=0.01012, output_cost=0.01133, prompt_tokens=1840, completion_tokens=412
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_TOOL_CALL: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "7a6c5d71a8fe summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tooluse_$REQUEST_ID",
                                "name": "get_weather",
                                "input": {
                                    "city": "Berlin",
                                    "days": 7,
                                    "units": "metric",
                                    "notes": "filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler filler ",
                                },
                            }
                        }
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.02145, input_cost=0.01012, output_cost=0.01133, prompt_tokens=1840, completion_tokens=412
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM_TOOL_CALL: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-stream_tool_call",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "0ef1034f8717 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather and a short forecast for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "days": {"type": "integer", "description": "Forecast horizon in days"},
                            "units": {"type": "string", "enum": ["metric", "imperial"]},
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockStart",
                payload={
                    "start": {"toolUse": {"toolUseId": "tooluse_$REQUEST_ID", "name": "get_weather"}},
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": '{"city": "Berlin", "days": 7, "units": "metric", "notes": "filler filler filler filler fil'
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": "ler filler filler filler filler filler filler filler filler filler filler filler filler fi"
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={
                    "delta": {
                        "toolUse": {
                            "input": 'ller filler filler filler filler filler filler filler filler filler filler filler filler "}'
                        }
                    },
                    "contentBlockIndex": 0,
                },
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "tool_use"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
                    "metrics": {"latencyMs": 42},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.02145, input_cost=0.01012, output_cost=0.01133, prompt_tokens=1840, completion_tokens=412
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM_FULL_USAGE: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-stream_full_usage",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "8ba9788e0bd5 summarize the attached material in one line and name the city weather",
                    }
                ],
            },
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta",
                payload={"delta": {"text": "scripted answer 8ba9788e0bd5"}, "contentBlockIndex": 0},
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {
                        "inputTokens": 1840,
                        "outputTokens": 412,
                        "totalTokens": 11468,
                        "cacheReadInputTokens": 6144,
                        "cacheWriteInputTokens": 3072,
                        "cacheDetails": [{"inputTokens": 2048, "ttl": "5m"}, {"inputTokens": 1024, "ttl": "1h"}],
                    },
                    "metrics": {"latencyMs": 42},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.0501732, input_cost=0.0388432, output_cost=0.01133, prompt_tokens=11056, completion_tokens=412
    ),
)

US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_MESSAGES_INPUT_TEXT: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-v1:0-messages_input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.anthropic.claude-opus-5-v1:0",
    endpoint="/v1/messages",
    request={
        "model": "$MODEL",
        "max_tokens": 412,
        "messages": [{"role": "user", "content": "summarize the attached material in one line"}],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted response"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.02145, input_cost=0.01012, output_cost=0.01133, prompt_tokens=1840, completion_tokens=412
    ),
)


# eu.anthropic.claude-sonnet-5-v1:0
BEDROCK_CONVERSE_EU_REGIONAL_KEY: Final = CostTrackingTestCase(
    name="bedrock-converse-eu-regional-key",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="eu.anthropic.claude-sonnet-5-v1:0",
    deployment=Deployment(model="bedrock/converse/eu.anthropic.claude-sonnet-5-v1:0"),
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "bedrock-converse-eu-regional-key"}],
        "stream": False,
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted response"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 1},
        },
    ),
    expected=ExactExpected(
        spend=0.01326, input_cost=0.006256, output_cost=0.007004, prompt_tokens=1840, completion_tokens=412
    ),
)


# amazon.nova-2-pro-preview-20251202-v1:0
BEDROCK_CONVERSE_NOVA_2_PRO: Final = CostTrackingTestCase(
    name="bedrock-converse-nova-2-pro",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="amazon.nova-2-pro-preview-20251202-v1:0",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "bedrock-converse-nova-2-pro"}],
        "stream": False,
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "scripted response"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 1},
        },
    ),
    expected=ExactExpected(
        spend=0.011235, input_cost=0.004025, output_cost=0.00721, prompt_tokens=1840, completion_tokens=412
    ),
)


# mistral.mistral-large-3-675b-instruct
BEDROCK_CONVERSE_MISTRAL_LARGE_3_STREAM: Final = CostTrackingTestCase(
    name="bedrock-converse-mistral-large-3-stream",
    covers="quota_management.spend_tracking.scripted_wire.logs_cost",
    model="mistral.mistral-large-3-675b-instruct",
    request={
        "model": "$MODEL",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a deterministic pricing-harness assistant. Keep answers to a single short line.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "bedrock-converse-mistral-large-3-stream"}]},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "allowed_openai_params": [],
    },
    response=EventStreamResponse(
        content_type="application/vnd.amazon.eventstream",
        events=(
            EventStreamEvent(event_type="messageStart", payload={"role": "assistant"}),
            EventStreamEvent(
                event_type="contentBlockDelta", payload={"delta": {"text": "scripted response"}, "contentBlockIndex": 0}
            ),
            EventStreamEvent(event_type="contentBlockStop", payload={"contentBlockIndex": 0}),
            EventStreamEvent(event_type="messageStop", payload={"stopReason": "end_turn"}),
            EventStreamEvent(
                event_type="metadata",
                payload={
                    "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
                    "metrics": {"latencyMs": 1},
                },
            ),
        ),
    ),
    expected=ExactExpected(
        spend=0.00156052, input_cost=0.0009384, output_cost=0.00062212, prompt_tokens=1840, completion_tokens=412
    ),
)


# us.anthropic.claude-opus-5-5
US_ANTHROPIC_CLAUDE_OPUS_5_5_INPUT_TEXT: Final = CostTrackingTestCase(
    name="us.anthropic.claude-opus-5-5-input_text",
    covers="quota_management.spend_tracking.cost_matrix.logs_cost",
    model="us.anthropic.claude-opus-5-5",
    request={
        "model": "$MODEL",
        "messages": [{"role": "user", "content": "Say hello."}],
        "stream": False,
        "allowed_openai_params": [],
    },
    response=JsonResponse(
        content_type="application/json",
        body={
            "output": {"message": {"role": "assistant", "content": [{"text": "Hello."}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1840, "outputTokens": 412, "totalTokens": 2252},
            "metrics": {"latencyMs": 42},
        },
    ),
    expected=ExactExpected(
        spend=0.01716, input_cost=0.008096, output_cost=0.009064, prompt_tokens=1840, completion_tokens=412
    ),
)


CASES: Final[tuple[CostTrackingTestCase, ...]] = (
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_INPUT_TEXT,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_CACHE_READ,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_CACHE_WRITE_5M,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_CACHE_WRITE_1H,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_SERVICE_TIER_FLEX,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_SERVICE_TIER_PRIORITY,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM_NO_USAGE,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM_NO_USAGE_TOOL_CALL,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM_NO_USAGE_IMAGE_INPUT,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_RESPONSE_MODEL_OVERRIDE,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM_RESPONSE_MODEL_OVERRIDE,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_TOOL_CALL,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM_TOOL_CALL,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM_FULL_USAGE,
    META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_INPUT_TEXT,
    META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_FALLBACK_CACHE_READ_AT_INPUT_RATE,
    META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_FALLBACK_CACHE_WRITE_AT_INPUT_RATE,
    META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM,
    META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM_NO_USAGE,
    META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM_NO_USAGE_TOOL_CALL,
    META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM_NO_USAGE_IMAGE_INPUT,
    META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_RESPONSE_MODEL_OVERRIDE,
    META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM_RESPONSE_MODEL_OVERRIDE,
    META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_TOOL_CALL,
    META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM_TOOL_CALL,
    META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM_FULL_USAGE,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_INPUT_TEXT,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_CACHE_READ,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_CACHE_WRITE_5M,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_CACHE_WRITE_1H,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_TIERED_INPUT_ABOVE_200K,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_TIERED_CACHE_READ_ABOVE_200K,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_TIERED_CACHE_WRITE_ABOVE_200K,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_SERVICE_TIER_FLEX,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_SERVICE_TIER_PRIORITY,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM_NO_USAGE,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM_NO_USAGE_TOOL_CALL,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM_NO_USAGE_IMAGE_INPUT,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_RESPONSE_MODEL_OVERRIDE,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM_RESPONSE_MODEL_OVERRIDE,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_TOOL_CALL,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM_TOOL_CALL,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM_FULL_USAGE,
    US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_MESSAGES_INPUT_TEXT,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_MESSAGES_CACHE_READ,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_PASSTHROUGH_CONVERSE,
    ANTHROPIC_CLAUDE_SONNET_5_V1_0_PASSTHROUGH_CONVERSE_STREAM,
    BEDROCK_CONVERSE_PROFILE_BASE_MODEL,
    BEDROCK_CONVERSE_EU_REGIONAL_KEY,
    BEDROCK_CONVERSE_APAC_BARE_FALLBACK,
    BEDROCK_CONVERSE_NOVA_2_PRO,
    BEDROCK_CONVERSE_MISTRAL_LARGE_3_STREAM,
    US_ANTHROPIC_CLAUDE_OPUS_5_5_INPUT_TEXT,
)

PARITY: Final[tuple[StreamParityTestCase, ...]] = (
    StreamParityTestCase(
        plain=ANTHROPIC_CLAUDE_SONNET_5_V1_0_TOOL_CALL, streamed=ANTHROPIC_CLAUDE_SONNET_5_V1_0_STREAM_TOOL_CALL
    ),
    StreamParityTestCase(
        plain=META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_TOOL_CALL,
        streamed=META_LLAMA4_MAVERICK_17B_INSTRUCT_V1_0_STREAM_TOOL_CALL,
    ),
    StreamParityTestCase(
        plain=US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_TOOL_CALL, streamed=US_ANTHROPIC_CLAUDE_OPUS_5_V1_0_STREAM_TOOL_CALL
    ),
)
