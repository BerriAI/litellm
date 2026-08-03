import re
from types import MappingProxyType

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

ZAI_API_BASE = "https://api.z.ai/api/paas/v4"
GLM_VERSION_PATTERN = re.compile(r"glm-(\d+)(?:\.(\d+))?")
THINKING_ON = MappingProxyType({"type": "enabled"})
THINKING_OFF = MappingProxyType({"type": "disabled"})


class ZAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "zai"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        api_base = api_base or get_secret_str("ZAI_API_BASE") or ZAI_API_BASE
        dynamic_api_key = api_key or get_secret_str("ZAI_API_KEY")
        return api_base, dynamic_api_key

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: list[AllMessageValues],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> tuple[list[AllMessageValues], list[ChatCompletionToolParam] | None]:
        """
        Override to preserve cache_control for GLM/ZAI.
        GLM supports cache_control - don't strip it.
        """
        # GLM/ZAI supports cache_control, so return messages and tools unchanged
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list:
        base_params = [
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
            "response_format",
        ]

        if self._is_reasoning_model(model):
            base_params.extend(("thinking", "reasoning_effort"))

        return base_params

    @staticmethod
    def _glm_version(model: str) -> tuple[int, int] | None:
        """Parse the ``glm-<major>[.<minor>]`` version out of a Z.AI model id.

        Returns ``None`` for ids that don't follow that shape.
        """
        match = GLM_VERSION_PATTERN.match(model.split("/")[-1].lower())
        return (int(match[1]), int(match[2] or 0)) if match else None

    @classmethod
    def _is_reasoning_model(cls, model: str) -> bool:
        """GLM-4.5 and newer expose ``thinking``.

        Read from the id rather than the model map: the map marks GLM-4.6 and up
        but not GLM-4.5, and it lags every new release.
        """
        version = cls._glm_version(model)
        return version is not None and version >= (4, 5)

    @classmethod
    def _supports_reasoning_effort_param(cls, model: str) -> bool:
        """``reasoning_effort`` is native to GLM-5.2 and newer.

        Earlier reasoning models (GLM-4.5 through GLM-5.1) only expose the
        ``thinking`` object, so ``reasoning_effort`` is translated to that instead
        of being forwarded.
        """
        version = cls._glm_version(model)
        return version is not None and version >= (5, 2)

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: signature fixed by BaseConfig.map_openai_params
        optional_params: dict,  # mutable-ok: signature fixed by BaseConfig.map_openai_params
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: signature fixed by BaseConfig.map_openai_params
        """GLM takes ``max_tokens`` (not ``max_completion_tokens``), and only
        GLM-5.2+ takes ``reasoning_effort`` natively."""
        optional_params = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )

        max_completion_tokens = optional_params.pop("max_completion_tokens", None)
        if max_completion_tokens is not None:
            optional_params.setdefault("max_tokens", max_completion_tokens)

        if not self._supports_reasoning_effort_param(model):
            reasoning_effort = optional_params.pop("reasoning_effort", None)
            if isinstance(reasoning_effort, str):
                optional_params.setdefault("thinking", THINKING_OFF if reasoning_effort == "none" else THINKING_ON)

        return optional_params
