# Agent Threat Rules detection callback

A small cookbook example that screens user input against ATR-inspired
threat patterns before the request reaches the model. ATR is an open
detection standard for AI agent threats (prompt injection, tool
poisoning, MCP attacks, skill compromise) published under Apache-2.0:

https://github.com/Agent-Threat-Rule/agent-threat-rules

## What it does

`atr_detection_callback.py` defines `ATRDetectionGuardrail`, a
`CustomGuardrail` whose `async_pre_call_hook` runs each user message
through a handful of compiled regex patterns. On a match, it logs the
rule id and raises a `ValueError`, which LiteLLM surfaces to the caller
as a blocked request.

The patterns embedded in the file are illustrative copies covering
common categories: instruction override, system prompt exfiltration,
role-play jailbreak, base64-wrapped payloads, MCP tool override, and
`file://` SSRF references. The full ruleset lives in the ATR repository.

## Wire it up

Add a guardrail entry to your proxy config:

```yaml
guardrails:
  - guardrail_name: "atr-input-screen"
    litellm_params:
      guardrail: cookbook.atr_detection_callback.atr_detection_callback.ATRDetectionGuardrail
      mode: "pre_call"
      default_on: true
```

Then start the proxy as usual. Requests containing matching patterns
will be rejected before the LLM call, and the rule id will appear in
the proxy logs.

## Extending

To run against the live ATR YAML ruleset instead of embedded patterns,
load rule files at startup and compile their `detection.regex_patterns`
fields into the same list shape.
