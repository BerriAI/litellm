"""structured_outputs x Bedrock (Invoke).

Drive the real `claude` CLI in headless mode with the `--json-schema`
flag, route through a LiteLLM proxy aimed at Bedrock (Invoke), and assert that
the final stream-json `result` event surfaces a `structured_output`
object whose shape matches the schema.

The (feature, provider) for this cell is inferred from the file path by
`tests/e2e/claude_code/conftest.py`:

    tests/e2e/claude_code/structured_outputs/test_bedrock_invoke.py
                       ^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^
                       feature_id              provider

What this row actually exercises (and what it does not):

`--json-schema` is implemented client-side by Claude Code: the CLI
synthesizes a synthetic `StructuredOutput` tool whose `input_schema`
equals the user-supplied JSON Schema, forces the model toward it, and
finally extracts the tool_use input on the trailing `result` event as
`structured_output: {...}`. The proxy never sees `output_config.schema`
in this flow -- it sees a normal `tools` array with one synthetic
tool.

This makes the row a tool-use feature test in disguise. It's still a
distinct row from `tool_use` because:

  - The synthetic tool is generated per request from a user schema, not
    a developer-declared one. Provider-side bugs that special-case
    `Claude Code`-generated tool names (e.g. case-folding `tool_use`
    blocks back to lowercase, or stripping the StructuredOutput-only
    `additionalProperties: false`) only surface here.
  - The success signal lives on the *final* `result` event, not the
    intermediate `assistant` events the `tool_use` row checks. A proxy
    that drops trailing events (seen in early Bedrock Converse SSE
    plumbing) breaks this cell while leaving `tool_use` green.

It is NOT a test of Anthropic's server-side `output_config.schema`
parameter -- that's a different feature used internally by Claude Code
for session-title generation and is not reachable from any CLI flag.
LiteLLM's `output_config`-stripping fixes (2.1.122, 2.1.81) surface in
the `count_tokens` and other HTTP-probe rows, not here.

Three Claude tiers run in parallel; one `compat_result.add(...)` per
tier so the matrix's "all three must pass" rule applies.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping, Optional, Sequence, Tuple

import pytest

from claude_code.cli_driver import (
    ClaudeCLIError,
    failure_diagnostic,
    run_claude_models_parallel,
)

PROXY_BASE_URL_ENV = "LITELLM_PROXY_BASE_URL"
PROXY_API_KEY_ENV = "LITELLM_PROXY_API_KEY"

BEDROCK_INVOKE_MODELS = [
    "claude-haiku-4-5-bedrock-invoke",
    "claude-sonnet-4-6-bedrock-invoke",
    "claude-opus-4-7-bedrock-invoke",
]

# Minimal schema with one required integer field. Kept intentionally
# small -- the matrix tests the *plumbing*, not the model's ability to
# satisfy a complex schema. A trivial arithmetic prompt + a one-field
# integer schema gives every tier (including Haiku) enough headroom
# that schema satisfaction is essentially deterministic, isolating
# failures to the proxy / transport.
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}},
    "required": ["answer"],
    "additionalProperties": False,
}
SCHEMA_JSON = json.dumps(SCHEMA, separators=(",", ":"))

# A prompt the model has no reason to misanswer; we don't check the
# value, but a wrong answer would suggest the structured-output
# pathway is silently degrading reasoning, which is itself worth
# noticing.
PROMPT = "What is 2 + 2? Reply only via the structured output."


def _extract_structured_output(
    events: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Return the `structured_output` payload from the last `result` event.

    Claude Code emits its terminal stream-json line as
    `{"type":"result","structured_output":{...},...}` when a request
    used `--json-schema` and the model actually produced a valid tool
    call. If the model bailed or the proxy ate the trailing events,
    `structured_output` is missing -- which is exactly the failure
    mode we want this row to surface, so the caller treats `None` as
    "feature did not work end-to-end".
    """
    for event in reversed(list(events)):
        if event.get("type") != "result":
            continue
        so = event.get("structured_output")
        if isinstance(so, Mapping):
            return so
    return None


def _validate_against_schema(
    payload: Mapping[str, Any], schema: Mapping[str, Any]
) -> Optional[str]:
    """Tiny shape validator covering the subset we actually need.

    We deliberately do not pull in `jsonschema` as a test dep: the
    matrix's success signal is "does the proxy let the synthetic
    StructuredOutput tool round-trip end-to-end", and that's
    answerable with a presence + type check over `required` keys.
    Any malformed schema beyond that would be a Claude Code bug,
    not a LiteLLM-proxy bug, so a deeper check would only add false
    failures on the wrong axis.
    """
    type_map = {
        "integer": int,
        "number": (int, float),
        "string": str,
        "boolean": bool,
        "array": list,
        "object": Mapping,
    }
    required = schema.get("required") or []
    properties = schema.get("properties") or {}
    for key in required:
        if key not in payload:
            return f"missing required key {key!r}"
        expected = (properties.get(key) or {}).get("type")
        if expected and expected in type_map:
            if not isinstance(payload[key], type_map[expected]):
                return (
                    f"key {key!r} has wrong type: "
                    f"expected {expected}, got {type(payload[key]).__name__}"
                )
        # bool is a subclass of int in Python; reject `True`/`False`
        # when the schema asked for an integer/number.
        if expected in ("integer", "number") and isinstance(payload[key], bool):
            return f"key {key!r} is a bool but schema asked for {expected}"
    return None


@pytest.mark.covers("llm.messages.bedrock_invoke.structured_output.nonstream.works")
def test_structured_outputs_bedrock_invoke(compat_result):
    """Drive `claude --json-schema ...` against the LiteLLM proxy and
    assert the trailing `result` event contains a schema-conforming
    `structured_output`."""
    base_url = os.environ.get(PROXY_BASE_URL_ENV)
    api_key = os.environ.get(PROXY_API_KEY_ENV)
    if not base_url or not api_key:
        compat_result.set(
            {
                "status": "fail",
                "error": (
                    f"missing required env: set {PROXY_BASE_URL_ENV} and "
                    f"{PROXY_API_KEY_ENV} to point at a running LiteLLM proxy"
                ),
            }
        )
        pytest.fail(
            f"{PROXY_BASE_URL_ENV} / {PROXY_API_KEY_ENV} not configured",
            pytrace=False,
        )

    outcomes = run_claude_models_parallel(
        models=BEDROCK_INVOKE_MODELS,
        prompt=PROMPT,
        base_url=base_url,
        api_key=api_key,
        extra_args=["--json-schema", SCHEMA_JSON],
    )

    failures = []
    for model in BEDROCK_INVOKE_MODELS:
        outcome = outcomes[model]
        if isinstance(outcome, ClaudeCLIError):
            error = f"[{model}] {outcome}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        if outcome.exit_code != 0:
            error = f"[{model}] claude CLI failed: {failure_diagnostic(outcome)}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        payload = _extract_structured_output(outcome.events)
        if payload is None:
            error = (
                f"[{model}] no `structured_output` in trailing result event; "
                "Claude Code's StructuredOutput tool round-trip did not "
                "complete end-to-end through the proxy"
            )
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        shape_error = _validate_against_schema(payload, SCHEMA)
        if shape_error is not None:
            error = f"[{model}] structured_output shape error: {shape_error}; payload={payload}"
            compat_result.add({"status": "fail", "error": error})
            failures.append(error)
            continue

        compat_result.add({"status": "pass"})

    if failures:
        pytest.fail("; ".join(failures), pytrace=False)
