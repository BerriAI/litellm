from typing import List, Optional, Union

from pydantic import ConfigDict, BaseModel, validator, VERSION


# Function to get Pydantic version
def is_pydantic_v2() -> int:
    return int(VERSION.split(".")[0])


def get_model_config(arbitrary_types_allowed: bool = False) -> ConfigDict:
    # Version-specific configuration
    if is_pydantic_v2() >= 2:
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=arbitrary_types_allowed, protected_namespaces=())  # type: ignore
    else:
        from pydantic import Extra

        model_config = ConfigDict(extra=Extra.allow, arbitrary_types_allowed=arbitrary_types_allowed)  # type: ignore

    return model_config


class EmbeddingRequest(BaseModel):
    model: str
    input: List[str] = []
    timeout: int = 600
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    api_key: Optional[str] = None
    api_type: Optional[str] = None
    caching: bool = False
    user: Optional[str] = None
    custom_llm_provider: Optional[Union[str, dict]] = None
    litellm_call_id: Optional[str] = None
    litellm_logging_obj: Optional[dict] = None
    logger_fn: Optional[str] = None
    model_config = get_model_config()
