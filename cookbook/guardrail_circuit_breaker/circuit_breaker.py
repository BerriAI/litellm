"""A per-run circuit breaker guardrail for the LiteLLM proxy.

Monthly budgets and rate limits protect you across *many* requests. They do not stop a
single agent run stuck in a loop — calling the same broken tool over and over, paying full
price each time, until someone notices. Most of a runaway's spend happens *after* the model
already saw the first tool error, so you cannot rely on the model to stop itself.

This guardrail watches one run and cuts it, on the request path, when it trips:

  * an identical tool call (same name + args) is emitted twice in a row  -> stuck loop
  * the same tool result comes back an error twice in a row              -> broken tool
  * the run reaches its session budget (a per-run spend ceiling)         -> over budget
  * the run hits an absolute call backstop

A long, legitimate run moves through many different tool calls and is never cut. (Loop
detection applies only to tool calls — a text answer that happens to repeat is not a
runaway, so text turns don't drive it; they still count toward the budget and backstop.)

--- Run it -------------------------------------------------------------------------------

Register it (see config.yaml in this folder), then start the proxy:

    guardrails:
      - guardrail_name: "circuit-breaker"
        litellm_params:
          guardrail: circuit_breaker.CircuitBreaker   # module.ClassName (on PYTHONPATH)
          mode: [pre_call, post_call]
          default_on: true

    BREAKER_RUN_BUDGET_USD=0.50 PYTHONPATH=. litellm --config config.yaml   # :4000

Point any OpenAI-compatible client at it and pass a run id so the breaker can scope state to
one agent run:

    client.chat.completions.create(
        model="my-model", messages=[...], tools=[...],
        extra_body={"metadata": {"run_id": "agent-run-abc123"}},   # keys the breaker
    )

When a run trips, the next call is rejected with the reason, e.g.
`identical tool call ... repeated 2x in a row — stuck loop, cut run`.

--- Two conventions (the proxy sees chat calls, not tool executions) ----------------------

  1. Per-run identity is caller-supplied — pass it as metadata.run_id. (State is additionally
     namespaced by the authenticated key, and each caller's runs are LRU-bounded on their own,
     so one caller cannot read, trip, or evict another caller's run.)
  2. A tool error is a marked result — the agent sets is_error=True on the role:"tool"
     message (text starting with "error" is also treated as one).

Tune via env: BREAKER_RUN_BUDGET_USD (session budget, unset=off), BREAKER_RUN_MAX_CALLS,
BREAKER_MAX_TRACKED_RUNS (LRU cap on tracked runs *per caller*), BREAKER_MAX_TRACKED_CALLERS
(LRU cap on tracked callers). Worst-case tracked runs is the product of the two.

Extended version — windowed backstops, a pluggable Redis store, a standalone Node gateway,
and a full test suite: https://github.com/smallestbusiness/guardrail
"""
import hashlib
import json
import os
from collections import OrderedDict

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _signature(tool_calls) -> str:
    """Stable signature over ALL tool calls this turn (name + canonical args), so parallel
    tool calls are covered — not just the first."""
    parts = []
    for tc in tool_calls:
        fn = _get(tc, "function", tc)
        name = _get(fn, "name", "tool")
        args = _get(fn, "arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, ValueError):
                pass
        try:
            blob = json.dumps(args or {}, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            blob = str(args)
        parts.append(f"{name}({blob})")
    joined = "|".join(sorted(parts))
    return "tools#" + hashlib.sha1(joined.encode()).hexdigest()[:12]


class CircuitBreaker(CustomGuardrail):
    def __init__(self, **kwargs):
        # Two-level LRU: caller -> OrderedDict(run_id -> state). Each caller's runs are
        # evicted independently, so one caller flooding distinct run ids can only push out
        # its OWN runs — never another caller's tripped/cost/call state.
        self.runs = OrderedDict()
        self.budget = float(os.getenv("BREAKER_RUN_BUDGET_USD", "0") or 0) or None
        self.max_calls = int(os.getenv("BREAKER_RUN_MAX_CALLS", "500"))
        self.max_runs = int(os.getenv("BREAKER_MAX_TRACKED_RUNS", "1000"))       # per caller
        self.max_callers = int(os.getenv("BREAKER_MAX_TRACKED_CALLERS", "10000"))
        self.max_events = 50  # cap per-run history so one run can't grow without bound
        super().__init__(**kwargs)

    def _ids(self, user_api_key_dict, data):
        """Split identity into (caller, run_id). State is namespaced by the AUTHENTICATED
        caller, so a guessed run id from one caller can't read or trip another caller's run."""
        md = data.get("metadata") or {}
        run_id = md.get("run_id") or data.get("litellm_call_id") or "default"
        caller = _get(user_api_key_dict, "api_key") or _get(user_api_key_dict, "user_id") or "anon"
        caller = hashlib.sha1(str(caller).encode()).hexdigest()[:12]
        return caller, run_id

    def _state(self, caller, run_id):
        bucket = self.runs.get(caller)
        if bucket is None:
            if len(self.runs) >= self.max_callers:
                self.runs.popitem(last=False)  # evict least-recently-used caller bucket
            bucket = OrderedDict()
            self.runs[caller] = bucket
        else:
            self.runs.move_to_end(caller)

        r = bucket.get(run_id)
        if r is None:
            if len(bucket) >= self.max_runs:
                bucket.popitem(last=False)  # evict only THIS caller's least-recently-used run
            r = {"events": [], "cost": 0.0, "calls": 0, "tripped": None}
            bucket[run_id] = r
        else:
            bucket.move_to_end(run_id)
        return r

    # ---- pre-call: refuse a run that's already been cut, or has spent its budget --------
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        r = self._state(*self._ids(user_api_key_dict, data))
        if r["tripped"]:
            self._cut(r["tripped"])
        if self.budget is not None and r["cost"] >= self.budget:
            r["tripped"] = f"run hit its ${self.budget} session budget (spent ${r['cost']:.2f})"
            self._cut(r["tripped"])
        return data

    # ---- post-call: price the turn, feed the breaker its two signals --------------------
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        r = self._state(*self._ids(user_api_key_dict, data))
        r["calls"] += 1

        try:
            r["cost"] += litellm.completion_cost(completion_response=response) or 0.0
        except Exception:
            pass  # never let a pricing hiccup disable the breaker

        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        # LOOP / BROKEN-TOOL detection is only meaningful for tool calls. A repeated text
        # answer is not a runaway (and repeating text would false-positive), so text turns
        # are skipped here — they still count toward the budget and the call backstop.
        if tool_calls:
            sig = _signature(tool_calls)
            errored = False
            for m in reversed(data.get("messages", [])):
                if _get(m, "role") == "tool":
                    content = m.get("content") if isinstance(m, dict) else _get(m, "content")
                    errored = _get(m, "is_error") is True or str(content or "").strip().lower().startswith("error")
                    break

            ev = r["events"]
            ev.append((sig, not errored))
            del ev[:-self.max_events]  # bound per-run history

            if len(ev) >= 2 and ev[-1][0] == ev[-2][0]:
                if not ev[-1][1] and not ev[-2][1]:
                    r["tripped"] = f"tool call {sig} errored 2x in a row — broken tool, cut run"
                else:
                    r["tripped"] = f"identical tool call {sig} repeated 2x in a row — stuck loop, cut run"

        if r["tripped"] is None and r["calls"] >= self.max_calls:
            r["tripped"] = f"run hit {self.max_calls}-call backstop"
        return None

    # ---- how a run is stopped: raise so the proxy returns an error and the caller halts --
    def _cut(self, reason: str):
        # Raising an exception in a guardrail hook rejects the request and stops the run.
        # A plain exception keeps this example dependency-clean; inside the proxy you can
        # raise fastapi.HTTPException(status_code=429, detail=...) if you want a 429 status.
        raise Exception(f"circuit_breaker: {reason}")
