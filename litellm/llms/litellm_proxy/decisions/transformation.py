from collections.abc import Mapping
from dataclasses import replace
from typing import Final

from litellm.llms.base_llm.decisions.transformation import BaseDecisionsConfig
from litellm.llms.openai.decisions.transformation import OpenAIDecisionsConfig, ir_to_openai_request
from litellm.types.decisions import DecisionsIRRequest, UnsupportedDecisionsRequest


class LiteLLMProxyDecisionsConfig(OpenAIDecisionsConfig):
    """A LiteLLM proxy serves /v1/decisions in the OpenAI format, with its own base URL and key.

    Only the proxy knows the model behind the name it is given, so box questions, audio and videos are forwarded
    for the proxy to accept or refuse unless the deployment turns them off.
    """

    api_base_env = ("LITELLM_PROXY_API_BASE",)
    api_key_env = ("LITELLM_PROXY_API_KEY",)
    api_key_required = False

    def get_default_api_base(self) -> str | None:
        return None

    def missing_api_base_message(self, custom_llm_provider: str) -> str:
        return "api_base is required for the LiteLLM proxy, set api_base or LITELLM_PROXY_API_BASE"

    def resolve_api_base(self, api_base: str | None) -> str | None:
        return BaseDecisionsConfig.resolve_api_base(self, api_base)

    def resolve_api_key(self, api_key: str | None) -> str | None:
        return BaseDecisionsConfig.resolve_api_key(self, api_key)

    def resolve_credential(
        self, api_key: str | None, api_base: str, litellm_params: Mapping[str, object]
    ) -> str | None:
        return api_key

    def supports_box_questions(
        self, model: str, custom_llm_provider: str, headers: Mapping[str, str], deployment_flag: bool | None
    ) -> bool:
        return deployment_flag is not False

    def supports_audio_input(self, model: str, custom_llm_provider: str, deployment_flag: bool | None) -> bool:
        return deployment_flag is not False

    def supports_video_input(self, model: str, custom_llm_provider: str, deployment_flag: bool | None) -> bool:
        return deployment_flag is not False

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        body: Final = ir_to_openai_request(self.request_model(model), replace(request, audio=(), videos=()))
        if isinstance(body, UnsupportedDecisionsRequest):
            return body
        return {
            **body,
            **(
                {"audio": [clip if isinstance(clip, str) else clip.model_dump() for clip in request.audio]}
                if request.audio
                else {}
            ),
            **(
                {"videos": [clip if isinstance(clip, str) else clip.model_dump() for clip in request.videos]}
                if request.videos
                else {}
            ),
        }
