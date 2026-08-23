"""
Lint a request payload for cross-provider portability *before* handing it to
LiteLLM.

LiteLLM standardizes the *call* -- one `completion()` signature that routes
to OpenAI, Anthropic, Gemini, Bedrock, etc. It does not, however, check
whether the *content* of your messages/params is actually safe to send to
every provider behind that call. Things like:

  - a system prompt that hardcodes "As an OpenAI language model..."
  - a stop_sequences list with 5 entries (OpenAI caps this at 4)
  - a temperature of 2.5 (outside Anthropic/Gemini's accepted range)
  - a chat-template special token leaked from a different model family

...will pass straight through LiteLLM's translation layer and fail (or
silently misbehave) only once they hit the target provider's API.
`prompt-portability` catches these before the call is made, so a provider
swap in your LiteLLM `model=` string doesn't surface a portability bug at
runtime.

Install:
    pip install prompt-portability litellm

Run:
    python prompt_portability_litellm_check.py
"""

from __future__ import annotations

import litellm
from llm_prompt_lint.linter import lint
from llm_prompt_lint.parsers import detect_and_parse

# A request written and tested against OpenAI, about to be pointed at a
# different provider via LiteLLM's `model=` switch.
request = {
    "model": "gpt-4o",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant. As an OpenAI assistant, "
            "always answer in a formal tone.",
        },
        {"role": "user", "content": "Summarize this quarter's RAG index drift report."},
    ],
    "temperature": 2.5,
    "stop": ["\n\n", "END", "###", "STOP", "<|end|>"],
}

doc = detect_and_parse(request)
report = lint(doc)

print(f"prompt-portability found {len(report.findings)} portability issue(s):\n")
for finding in report.findings:
    print(f"  [{finding.rule_id}] {finding.message}")

if report.findings:
    print(
        "\nFix these before swapping providers -- e.g. via LiteLLM's "
        "`model=\"anthropic/claude-...\"` -- to avoid a runtime surprise on "
        "the new provider."
    )

# LiteLLM's own call succeeds regardless -- it standardizes the transport,
# not the content, which is exactly the gap prompt-portability closes.
response = litellm.completion(
    model="gpt-4o",
    messages=request["messages"],
    temperature=request["temperature"],
    stop=request["stop"],
    mock_response="This call succeeds even though the payload above has "
    "portability issues LiteLLM doesn't check.",
)
print(f"\nLiteLLM call (mocked) still went through: {response.choices[0].message.content!r}")
