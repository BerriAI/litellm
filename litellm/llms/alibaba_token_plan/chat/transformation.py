from collections.abc import Mapping, Sequence

from litellm.llms.alibaba_token_plan.common_utils import get_api_base, get_api_key, validate_headers
from litellm.llms.dashscope.chat.transformation import DashScopeChatConfig
from litellm.types.llms.openai import AllMessageValues


class AlibabaTokenPlanChatConfig(DashScopeChatConfig):
    @property
    def custom_llm_provider(self) -> str:
        return "alibaba_token_plan"

    def _get_openai_compatible_provider_info(self, api_base: str | None, api_key: str | None) -> tuple[str, str | None]:
        return self._resolve_chat_api_base(api_base), get_api_key(api_key)

    def _resolve_chat_api_base(self, api_base: str | None) -> str:
        return get_api_base(api_base)

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: provider contract returns mutable headers
        return validate_headers(headers, api_key)
