from typing import Final

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.llms.oss_decision import oss_connection, validate_oss_model


class LayaDecisionsConfig(BaseDecisionsConfig):
    api_key_required = False

    def canonical_model(self, model: str) -> str:
        return validate_oss_model("laya", model)

    def resolve_connection(self, api_base: str | None, api_key: str | None) -> tuple[str | None, str | None]:
        connection: Final = oss_connection("laya", api_base, api_key)
        return connection.api_base, connection.api_key
