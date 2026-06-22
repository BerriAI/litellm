"""Realtime providers under e2e test.

Each entry's ``model`` is the proxy alias (the ``model_name`` in the proxy
config), not the upstream model id. The upstream model + credentials live in
the proxy config (see realtime_e2e_config.yaml). ``required_env`` is the set of
env vars that must be present for the proxy to actually reach that provider; a
test parametrized on a provider skips when any are missing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RealtimeProvider:
    id: str
    model: str
    required_env: tuple[str, ...]


PROVIDERS: tuple[RealtimeProvider, ...] = (
    RealtimeProvider("openai", "openai-realtime", ("OPENAI_API_KEY",)),
    RealtimeProvider("azure", "azure-realtime", ("AZURE_API_KEY", "AZURE_API_BASE")),
    RealtimeProvider("gemini", "gemini-realtime", ("GEMINI_API_KEY",)),
    RealtimeProvider(
        "vertex_ai", "vertex-realtime", ("GOOGLE_APPLICATION_CREDENTIALS",)
    ),
    RealtimeProvider(
        "bedrock",
        "bedrock-realtime",
        ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
    ),
    RealtimeProvider("xai", "xai-realtime", ("XAI_API_KEY",)),
)

PROVIDER_IDS: tuple[str, ...] = tuple(p.id for p in PROVIDERS)
