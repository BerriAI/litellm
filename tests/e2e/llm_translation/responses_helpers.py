from __future__ import annotations

from typing import Final

from models import LiteLLMParamsBody

AZURE_OPENAI_BACKEND: Final = "azure/gpt-5.4-nano"
AZURE_OPENAI_API_VERSION: Final = "v1"


def azure_openai_params(api_version: str = AZURE_OPENAI_API_VERSION) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=AZURE_OPENAI_BACKEND,
        api_base="os.environ/AZURE_API_BASE",
        api_key="os.environ/AZURE_API_KEY",
        api_version=api_version,
    )
