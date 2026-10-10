from litellm.llms.alibaba_token_plan.common_utils import get_api_base, get_api_key
from litellm.llms.dashscope.chat.transformation import DashScopeChatConfig


class AlibabaTokenPlanChatConfig(DashScopeChatConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        return self._resolve_chat_api_base(api_base), get_api_key(api_key)

    def _resolve_chat_api_base(self, api_base: str | None) -> str:
        return get_api_base(api_base)
