from litellm.llms.alibaba_token_plan.common_utils import MESSAGES_PATH, get_api_url
from litellm.llms.openai_like.messages.transformation import JSONProviderAnthropicMessagesConfig


class AlibabaTokenPlanAnthropicMessagesConfig(JSONProviderAnthropicMessagesConfig):
    def _resolve_api_base(self, api_base: str | None) -> str:
        return get_api_url(api_base, MESSAGES_PATH)
