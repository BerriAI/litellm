import os
from functools import lru_cache
from typing import Final

from pydantic import BaseModel

OTEL_V2_ENV: Final = "LITELLM_OTEL_V2"


class _OTelV2Flag(BaseModel):
    enabled: bool = False


@lru_cache(maxsize=1)
def is_otel_v2_enabled() -> bool:
    environment: Final = {key.lower(): value for key, value in os.environ.items()}
    return _OTelV2Flag.model_validate({"enabled": environment.get(OTEL_V2_ENV.lower(), False)}).enabled
