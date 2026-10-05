from typing import Final

from litellm.llms.base_llm.decisions.transformation import JevCompatibleDecisionsEndpoint

TYPESAFE_DECISIONS_ENDPOINT: Final[JevCompatibleDecisionsEndpoint] = JevCompatibleDecisionsEndpoint(
    default_api_base_value="https://api.typesafe.ai",
    path="/v1/systemone",
    api_key_env=("TYPESAFE_API_KEY",),
    api_base_env="TYPESAFE_API_BASE",
)
