"""Single source of truth for every model the e2e suite drives.

Bump a model version here instead of editing individual tests. Constants are
named for the role a model plays, never its version, so a bump touches this
file (plus docker-compose.yml, which cannot import Python) and nothing else.
tests/code_coverage_tests/check_e2e_model_freshness.py fails CI when a pin
disappears from model_prices_and_context_window.json, approaches its
deprecation_date, drifts from the docker-compose gateway config, or when a
test hardcodes a model literal instead of importing a pin.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPin:
    provider: str
    model_id: str
    gateway_alias: str = ""
    pricing_key: str = ""

    @property
    def backend(self) -> str:
        return f"{self.provider}/{self.model_id}"

    @property
    def alias(self) -> str:
        return self.gateway_alias or self.model_id

    @property
    def canonical(self) -> str:
        return self.pricing_key or self.backend


GEMINI_CHAT = ModelPin("gemini", "gemini-3.5-flash")
OPENAI_CHAT = ModelPin("openai", "gpt-5.5")
OPENAI_CHAT_MINI = ModelPin("openai", "gpt-5.4-mini")
ANTHROPIC_CHAT = ModelPin("anthropic", "claude-haiku-4-5")
OPENAI_EMBEDDING = ModelPin(
    "openai", "text-embedding-3-small", gateway_alias="openai-text-embedding-3-small"
)
OPENAI_TTS = ModelPin("openai", "gpt-4o-mini-tts")
VERTEX_CHAT = ModelPin("vertex_ai", "gemini-3.5-flash")
AZURE_BATCH = ModelPin("azure", "gpt-4.1-mini-batch", pricing_key="azure/gpt-4.1-mini")
BEDROCK_ANTHROPIC_CHAT = ModelPin("bedrock", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
COHERE_RERANK = ModelPin("cohere", "rerank-v3.5")

GATEWAY_MODELS: tuple[ModelPin, ...] = (
    OPENAI_CHAT,
    ANTHROPIC_CHAT,
    GEMINI_CHAT,
    OPENAI_EMBEDDING,
)

ALL_PINS: tuple[ModelPin, ...] = (
    GEMINI_CHAT,
    OPENAI_CHAT,
    OPENAI_CHAT_MINI,
    ANTHROPIC_CHAT,
    OPENAI_EMBEDDING,
    OPENAI_TTS,
    VERTEX_CHAT,
    AZURE_BATCH,
    BEDROCK_ANTHROPIC_CHAT,
    COHERE_RERANK,
)
