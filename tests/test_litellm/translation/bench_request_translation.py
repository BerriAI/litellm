"""Request-translation CPU bench: v1 transform chain vs v2 pipeline.

Reproduces the pattern-auditor's method: Claude-Code-shaped tool-call
histories at 41/201/601 messages, median of 5 runs after warmup, comparing
v1 (``map_openai_params`` + ``transform_request``) with v2
(``translate_chat_request``). Budget: v2 <= 1.5x v1 at 601 messages.

Run:  python -m tests.test_litellm.translation.bench_request_translation
"""

import copy
import json
import statistics
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


def _median_ms(fn, request: dict, runs: int = 5) -> float:
    fn(copy.deepcopy(request))  # warmup
    samples = []
    for _ in range(runs):
        body = copy.deepcopy(request)
        start = time.perf_counter()
        fn(body)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


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
        v1_ms = _median_ms(_v1, request)
        v2_ms = _median_ms(v2, request)
        ratio = v2_ms / v1_ms
        if count >= 600:
            gated = ratio
        print(f"{count:>9} {v1_ms:>9.2f} {v2_ms:>9.2f} {ratio:>6.2f}x")
    budget = 1.5  # the audit budget is stated at 600-message histories
    status = "PASS" if gated <= budget else "FAIL"
    print(f"601-message ratio {gated:.2f}x vs {budget}x budget: {status}")


if __name__ == "__main__":
    main()
