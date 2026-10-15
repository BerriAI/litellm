from collections.abc import Mapping

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig, has_box_questions
from litellm.types.decisions import DecisionsIRRequest


class TypeSafeDecisionsConfig(BaseDecisionsConfig):
    api_key_env = ("TYPESAFE_API_KEY",)
    api_base_env = ("TYPESAFE_API_BASE",)
    supports_assets = True
    supports_box_questions = True

    def get_default_api_base(self) -> str | None:
        return "https://api.typesafe.ai"

    def request_headers(self, request: DecisionsIRRequest) -> Mapping[str, str]:
        if request.assets or has_box_questions(request):
            return {"TypeSafe-Danger-Zone-Images": "true"}
        return {}
