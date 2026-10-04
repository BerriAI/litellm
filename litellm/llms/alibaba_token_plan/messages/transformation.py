from collections.abc import Mapping

from litellm.llms.alibaba_token_plan.common_utils import get_messages_api_base, require_api_key
from litellm.llms.openai_like.messages.transformation import JSONProviderAnthropicMessagesConfig


class AlibabaTokenPlanAnthropicMessagesConfig(JSONProviderAnthropicMessagesConfig):
    def _resolve_api_key(self, api_key: str | None) -> str:
        return require_api_key(api_key)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return super().get_complete_url(
            api_base=get_messages_api_base(api_base),
            api_key=api_key,
            model=model,
            optional_params=dict(optional_params),
            litellm_params=dict(litellm_params),
            stream=stream,
        )
