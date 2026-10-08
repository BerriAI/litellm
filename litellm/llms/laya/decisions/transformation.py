from typing import Final

from litellm.llms.base_llm.decisions.transformation import JevCompatibleDecisionsEndpoint

LAYA_DECISIONS_ENDPOINT: Final[JevCompatibleDecisionsEndpoint] = JevCompatibleDecisionsEndpoint(
    default_api_base_value=None,
    path="/v1/systemone",
    api_key_env=("LAYA_API_KEY",),
    api_base_env="LAYA_API_BASE",
    api_key_required=False,
)
