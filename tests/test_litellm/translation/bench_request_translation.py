"""Request-translation CPU bench: v1 transform chain vs v2 pipeline.

Reproduces the pattern-auditor's method: Claude-Code-shaped tool-call
histories at 41/201/601 messages, comparing v1 (``map_openai_params`` +
``transform_request``) with v2 (``translate_chat_request``). Budget: v2 <= 1.5x
v1 at 601 messages.

Methodology: v1 and v2 are sampled INTERLEAVED (one v1 then one v2 per
iteration) so a transient scheduler hiccup lands on BOTH sides and cancels in
the ratio, and the estimator is the MINIMUM over many samples -- the cleanest
measure of a CPU-bound cost, since noise can only ADD time, never remove it. An
earlier version took the median of 5 separate-phase runs, which flapped FAIL on
a loaded machine even though the interleaved-min ratio was a stable ~1.42x (the
load-sensitivity the audit notes warned about); the min-of-many-interleaved
reading is what should ever gate, and only on a quiet runner.

Run:  python -m tests.test_litellm.translation.bench_request_translation
"""

import copy
import json
import time

from litellm.llms.anthropic.chat.transformation import AnthropicConfig

from litellm.translation import translate_chat_request

from .conftest import build_real_deps

MODEL = "claude-sonnet-4-5"

_TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}, "cwd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
}


def _history(turns: int) -> dict:
    messages = [{"role": "system", "content": "You are Claude Code."}]
    for index in range(turns):
        messages.extend(
            [
                {"role": "user", "content": f"step {index}: edit file {index}.py"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {
                                "name": "run_command",
                                "arguments": json.dumps(
                                    {"cmd": f"sed -i s/a/b/ {index}.py", "cwd": "/repo"}
                                ),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": f"call_{index}", "content": "ok" * 50},
                {"role": "assistant", "content": f"Done with step {index}."},
            ]
        )
    return {
        "model": MODEL,
        "max_tokens": 4096,
        "tools": [_TOOL],
        "messages": messages,
    }


def _v1(request: dict) -> dict:
    config = AnthropicConfig()
    params = {
        key: value for key, value in request.items() if key not in ("model", "messages")
    }
    optional = config.map_openai_params(
        copy.deepcopy(params), {}, request["model"], drop_params=False
    )
    return config.transform_request(
        request["model"], copy.deepcopy(request["messages"]), optional, {}, {}
    )


def _min_ms_interleaved(v1_fn, v2_fn, request: dict, samples: int = 60) -> tuple:
    """Sample v1 and v2 INTERLEAVED and return (min v1 ms, min v2 ms).

    Each side gets a fresh deepcopy per sample (translation is pure, but v1's
    transform mutates its message list), and the two are timed back-to-back so a
    scheduler hiccup hits both. The minimum over many samples is the CPU floor:
    noise only adds time, so the smallest sample is the least-polluted estimate.
    """
    for _ in range(10):  # warmup both
        v1_fn(copy.deepcopy(request))
        v2_fn(copy.deepcopy(request))
    v1_best = float("inf")
    v2_best = float("inf")
    for _ in range(samples):
        body = copy.deepcopy(request)
        start = time.perf_counter()
        v1_fn(body)
        v1_best = min(v1_best, (time.perf_counter() - start) * 1000)
        body = copy.deepcopy(request)
        start = time.perf_counter()
        v2_fn(body)
        v2_best = min(v2_best, (time.perf_counter() - start) * 1000)
    return v1_best, v2_best


def main() -> None:
    deps = build_real_deps()

    def v2(request: dict) -> None:
        result = translate_chat_request(request, "anthropic", deps)
        assert result.is_ok(), result.error.summary

    print(f"{'messages':>9} {'v1 ms':>9} {'v2 ms':>9} {'ratio':>7}")
    gated = 0.0
    for turns in (10, 50, 150):
        request = _history(turns)
        count = len(request["messages"])
        v1_ms, v2_ms = _min_ms_interleaved(_v1, v2, request)
        ratio = v2_ms / v1_ms
        if count >= 600:
            gated = ratio
        print(f"{count:>9} {v1_ms:>9.2f} {v2_ms:>9.2f} {ratio:>6.2f}x")
    budget = 1.5  # the audit budget is stated at 600-message histories
    status = "PASS" if gated <= budget else "FAIL"
    print(f"601-message ratio {gated:.2f}x vs {budget}x budget: {status}")


if __name__ == "__main__":
    main()
