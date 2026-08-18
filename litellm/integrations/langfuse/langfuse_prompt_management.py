"""
Call Hook for LiteLLM Proxy which allows Langfuse prompt management.
"""

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, cast

from packaging.version import Version

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.prompt_management_base import PromptManagementClient
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.types.llms.openai import AllMessageValues, ChatCompletionSystemMessage
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import StandardCallbackDynamicParams, StandardLoggingPayload

from ...litellm_core_utils.specialty_caches.dynamic_logging_cache import (
    DynamicLoggingCache,
)
from ..prompt_management_base import PromptManagementBase
from .langfuse import LangFuseLogger, resolve_langfuse_credentials
from .langfuse_handler import LangFuseHandler

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.client import ChatPromptClient, TextPromptClient

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LangfuseClass: TypeAlias = Langfuse

    PROMPT_CLIENT = TextPromptClient | ChatPromptClient
else:
    PROMPT_CLIENT = Any
    LangfuseClass = Any
    LiteLLMLoggingObj = Any
in_memory_dynamic_logger_cache: Final = DynamicLoggingCache()


@lru_cache(maxsize=10)
def langfuse_client_init(
    langfuse_public_key=None,
    langfuse_secret=None,
    langfuse_secret_key=None,
    langfuse_host=None,
    flush_interval=1,
    allow_env_credentials: bool = True,
) -> LangfuseClass:
    """
    Initialize Langfuse client with caching to prevent multiple initializations.

    Args:
        langfuse_public_key (str, optional): Public key for Langfuse. Defaults to None.
        langfuse_secret (str, optional): Secret key for Langfuse. Defaults to None.
        langfuse_host (str, optional): Host URL for Langfuse. Defaults to None.
        flush_interval (int, optional): Flush interval in seconds. Defaults to 1.

    Returns:
        Langfuse: Initialized Langfuse client instance

    Raises:
        Exception: If langfuse package is not installed
    """
    try:
        import langfuse
        from langfuse import Langfuse
    except Exception as e:
        raise Exception(
            f"\033[91mLangfuse not installed, try running 'pip install langfuse' to fix this error: {e}\n\033[0m"
        )

    public_key, secret_key, langfuse_host = resolve_langfuse_credentials(
        langfuse_public_key=langfuse_public_key,
        langfuse_secret=langfuse_secret,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_host=langfuse_host,
        allow_env_credentials=allow_env_credentials,
    )

    if not (langfuse_host.startswith("http://") or langfuse_host.startswith("https://")):
        # add http:// if unset, assume communicating over private network - e.g. render
        langfuse_host = "http://" + langfuse_host

    langfuse_release: Final = os.getenv("LANGFUSE_RELEASE")
    langfuse_debug: Final = os.getenv("LANGFUSE_DEBUG")

    parameters: Final = {
        "public_key": public_key,
        "secret_key": secret_key,
        "host": langfuse_host,
        "release": langfuse_release,
        "debug": langfuse_debug,
        "flush_interval": LangFuseLogger._get_langfuse_flush_interval(flush_interval),  # flush interval in seconds
    }

    if Version(langfuse.version.__version__) >= Version("2.6.0"):
        parameters["sdk_integration"] = "litellm"

    if Version(langfuse.version.__version__) >= Version("2.7.3"):
        import httpx

        import litellm

        from ...llms.custom_httpx.http_handler import get_ssl_configuration

        parameters["httpx_client"] = httpx.Client(
            verify=get_ssl_configuration(),
            cert=os.getenv("SSL_CERTIFICATE", litellm.ssl_certificate),
        )

    client: Final = Langfuse(**parameters)

    return client


class LangfusePromptManagement(LangFuseLogger, PromptManagementBase, CustomLogger):
    def __init__(
        self,
        langfuse_public_key=None,
        langfuse_secret=None,
        langfuse_host=None,
        flush_interval=1,
    ):
        import langfuse

        self.langfuse_sdk_version = langfuse.version.__version__
        self.Langfuse = langfuse_client_init(
            langfuse_public_key=langfuse_public_key,
            langfuse_secret=langfuse_secret,
            langfuse_host=langfuse_host,
            flush_interval=flush_interval,
        )

    @property
    def integration_name(self):
        return "langfuse"

    def _get_prompt_from_id(
        self,
        langfuse_prompt_id: str,
        langfuse_client: LangfuseClass,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
    ) -> PROMPT_CLIENT:
        prompt_client = langfuse_client.get_prompt(langfuse_prompt_id, label=prompt_label, version=prompt_version)

        return prompt_client

    def _compile_prompt(
        self,
        langfuse_prompt_client: PROMPT_CLIENT,
        langfuse_prompt_variables: dict | None,
        call_type: Literal["completion", "text_completion"],
    ) -> list[AllMessageValues]:
        compiled_prompt: str | list | None = None

        if langfuse_prompt_variables is None:
            langfuse_prompt_variables = {}

        compiled_prompt = langfuse_prompt_client.compile(**langfuse_prompt_variables)

        if isinstance(compiled_prompt, str):
            compiled_prompt = [ChatCompletionSystemMessage(role="system", content=compiled_prompt)]
        else:
            compiled_prompt = cast(list[AllMessageValues], compiled_prompt)

        return compiled_prompt

    def _get_optional_params_from_langfuse(self, langfuse_prompt_client: PROMPT_CLIENT) -> dict:
        config: Final = langfuse_prompt_client.config
        optional_params: Final = {}
        for k, v in config.items():
            if k != "model":
                optional_params[k] = v
        return optional_params

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: list[AllMessageValues],
        non_default_params: dict,
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        litellm_logging_obj: LiteLLMLoggingObj,
        prompt_spec: PromptSpec | None = None,
        tools: list[dict] | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        ignore_prompt_manager_model: bool | None = False,
        ignore_prompt_manager_optional_params: bool | None = False,
    ) -> tuple[
        str,
        list[AllMessageValues],
        dict,
    ]:
        return self.get_chat_completion_prompt(
            model,
            messages,
            non_default_params,
            prompt_id,
            prompt_variables,
            dynamic_callback_params,
            prompt_spec=prompt_spec,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
            ignore_prompt_manager_model=ignore_prompt_manager_model,
            ignore_prompt_manager_optional_params=ignore_prompt_manager_optional_params,
        )

    def should_run_prompt_management(
        self,
        prompt_id: str | None,
        prompt_spec: PromptSpec | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        if prompt_id is None:
            return False
        langfuse_client: Final = langfuse_client_init(
            langfuse_public_key=dynamic_callback_params.get("langfuse_public_key"),
            langfuse_secret=dynamic_callback_params.get("langfuse_secret"),
            langfuse_secret_key=dynamic_callback_params.get("langfuse_secret_key"),
            langfuse_host=dynamic_callback_params.get("langfuse_host"),
            allow_env_credentials=dynamic_callback_params.get("langfuse_host") is None,
        )
        langfuse_prompt_client: Final = self._get_prompt_from_id(
            langfuse_prompt_id=prompt_id,
            langfuse_client=langfuse_client,
        )
        return langfuse_prompt_client is not None

    def _compile_prompt_helper(
        self,
        prompt_id: str | None,
        prompt_spec: PromptSpec | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
    ) -> PromptManagementClient:
        if prompt_id is None:
            raise ValueError("prompt_id is required for Langfuse prompt management")

        langfuse_client: Final = langfuse_client_init(
            langfuse_public_key=dynamic_callback_params.get("langfuse_public_key"),
            langfuse_secret=dynamic_callback_params.get("langfuse_secret"),
            langfuse_secret_key=dynamic_callback_params.get("langfuse_secret_key"),
            langfuse_host=dynamic_callback_params.get("langfuse_host"),
            allow_env_credentials=dynamic_callback_params.get("langfuse_host") is None,
        )
        langfuse_prompt_client: Final = self._get_prompt_from_id(
            langfuse_prompt_id=prompt_id,
            langfuse_client=langfuse_client,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
        )

        ## SET PROMPT
        compiled_prompt: Final = self._compile_prompt(
            langfuse_prompt_client=langfuse_prompt_client,
            langfuse_prompt_variables=prompt_variables,
            call_type="completion",
        )

        template_model: Final = langfuse_prompt_client.config.get("model")

        template_optional_params: Final = self._get_optional_params_from_langfuse(langfuse_prompt_client)

        return PromptManagementClient(
            prompt_id=prompt_id,
            prompt_template=compiled_prompt,
            prompt_template_model=template_model,
            prompt_template_optional_params=template_optional_params,
            completed_messages=None,
        )

    async def async_compile_prompt_helper(
        self,
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: PromptSpec | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
    ) -> PromptManagementClient:
        return self._compile_prompt_helper(
            prompt_id=prompt_id,
            prompt_variables=prompt_variables,
            dynamic_callback_params=dynamic_callback_params,
            prompt_spec=prompt_spec,
            prompt_label=prompt_label,
            prompt_version=prompt_version,
        )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        return run_async_function(self.async_log_success_event, kwargs, response_obj, start_time, end_time)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        return run_async_function(self.async_log_failure_event, kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            standard_callback_dynamic_params: Final = kwargs.get("standard_callback_dynamic_params")
            langfuse_logger_to_use: Final = LangFuseHandler.get_langfuse_logger_for_request(
                globalLangfuseLogger=self,
                standard_callback_dynamic_params=standard_callback_dynamic_params,
                in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
            )
            langfuse_logger_to_use.log_event_on_langfuse(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                user_id=kwargs.get("user", None),
            )
        except Exception as e:
            from litellm._logging import verbose_logger

            verbose_logger.exception("Langfuse Layer Error - Exception occurred while logging success event: %s", e)
            self.handle_callback_failure(callback_name="langfuse")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            standard_callback_dynamic_params: Final = kwargs.get("standard_callback_dynamic_params")
            langfuse_logger_to_use: Final = LangFuseHandler.get_langfuse_logger_for_request(
                globalLangfuseLogger=self,
                standard_callback_dynamic_params=standard_callback_dynamic_params,
                in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
            )
            standard_logging_object: Final = cast(
                StandardLoggingPayload | None,
                kwargs.get("standard_logging_object", None),
            )
            status_message = str(kwargs.get("exception", "Unknown error"))
            if standard_logging_object is not None:
                status_message = standard_logging_object.get("error_str", None) or status_message
            langfuse_logger_to_use.log_event_on_langfuse(
                start_time=start_time,
                end_time=end_time,
                response_obj=None,
                user_id=kwargs.get("user", None),
                status_message=status_message,
                level="ERROR",
                kwargs=kwargs,
            )
        except Exception as e:
            from litellm._logging import verbose_logger

            verbose_logger.exception("Langfuse Layer Error - Exception occurred while logging failure event: %s", e)
            self.handle_callback_failure(callback_name="langfuse")
