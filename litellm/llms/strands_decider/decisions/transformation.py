from typing import Final

from litellm.llms.base_llm.decisions.transformation import JevCompatibleDecisionsEndpoint

STRANDS_DECIDER_DECISIONS_ENDPOINT: Final[JevCompatibleDecisionsEndpoint] = JevCompatibleDecisionsEndpoint(
    default_api_base_value=None,
    path="/v1/systemone",
    api_key_env=("STRANDS_DECIDER_API_KEY",),
    api_base_env="STRANDS_DECIDER_API_BASE",
    api_key_required=False,
)
