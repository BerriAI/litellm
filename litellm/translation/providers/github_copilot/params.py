"""Parameter gates for the github_copilot chat serializer.

``GithubCopilotConfig`` extends ``OpenAIConfig`` (NOT OpenAIGPTConfig), so
its supported-param surface is OpenAI's own model-shape fork over the BARE
model name. Copilot model names are bare ("gpt-4o", "gpt-5",
"claude-sonnet-4.5"), so OpenAI's gpt-5 / o-series gates leak in exactly
like provider "openai" names — REUSE the shared openai_compat gates. Probed
in-process at HEAD with a seeded fake token dir (no live OAuth):

- gpt-5 names: ``temperature != 1`` RAISES (OpenAIGPT5Config) — the shared
  ``unsupported_model_family`` gate falls back on the whole gpt-5 family;
- o-series names: their own param rewrites — same shared family gate;
- claude models: ``reasoning_effort`` and ``thinking`` RAISE
  UnsupportedParamsError (the config's claude thinking/reasoning_effort
  additions are DEAD at HEAD — ``supports_reasoning`` keys are absent, so
  the params never enter the supported list; researcher-5 correction 3);
- ``user`` is silently dropped (model-list gated upstream — v1 serves its
  own drop) for claude; it SERVES for gpt-4o. The shared gate falls back on
  ``user`` for every model (the model-list membership is ambient state v2
  does not read), so v1 keeps serving the served case and reproduces the
  drop on the other — a typed fallback either way;
- ``response_format`` on the two name-gated models raises (the base-list
  name gate).
"""

from __future__ import annotations

from ...ir import ChatRequest
from ..openai_compat import params as openai_params


def unsupported_model_family(model: str) -> str | None:
    return openai_params.unsupported_model_family(model)


def unsupported_params(request: ChatRequest) -> str | None:
    return openai_params.unsupported_params(request)


def unsupported_response_format(request: ChatRequest) -> str | None:
    return openai_params.unsupported_response_format(request)
