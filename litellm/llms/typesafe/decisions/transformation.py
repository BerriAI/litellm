from typing import Final

from litellm.llms.base_llm.decisions.transformation import DecisionsEndpoint

TYPESAFE_DECISIONS_ENDPOINT: Final[DecisionsEndpoint] = DecisionsEndpoint(
    default_api_base="https://api.typesafe.ai",
    path="/v1/systemone",
    api_key_env=("TYPESAFE_API_KEY",),
    api_base_env="TYPESAFE_API_BASE",
)
