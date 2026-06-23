from enum import Enum
from typing import Optional


class MistralRustOcrProvider(str, Enum):
    MISTRAL = "mistral"


def get_mistral_rust_ocr_provider(config_type: type) -> Optional[str]:
    if (
        config_type.__module__ != "litellm.llms.mistral.ocr.transformation"
        or config_type.__name__ != "MistralOCRConfig"
    ):
        return None
    return MistralRustOcrProvider.MISTRAL.value
