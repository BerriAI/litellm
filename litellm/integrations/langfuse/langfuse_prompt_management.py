"""
Call Hook for LiteLLM Proxy which allows Langfuse prompt management.
"""

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, Union, cast

from packaging.version import Version
from typing_extensions import TypeAlias

from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.prompt_management_base import PromptManagementClient
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.types.llms.openai import AllMessageValues, ChatCompletionSystemMessage
from litellm.types.utils import StandardCallbackDynamicParams, StandardLoggingPayload

from ...litellm_core_utils.specialty_caches.dynamic_logging_cache import (
    DynamicLoggingCache,
)
from ..prompt_management_base import PromptManagementBase
from .langfuse import LangFuseLogger
from .langfuse_handler import LangFuseHandler

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.client import ChatPromptClient, TextPromptClient

    LangfuseClass: TypeAlias = Langfuse

    PROMPT_CLIENT = Union[TextPromptClient, ChatPromptClient]
else:
    PROMPT_CLIENT = Any
    LangfuseClass = Any

in_memory_dynamic_logger_cache = DynamicLoggingCache()


@lru_cache(maxsize=10)
def langfuse_client_init(
    langfuse_public_key=None,
    langfuse_secret=None,
    langfuse_host=None,
    flush_interval=1,
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

    # Instance variables
    secret_key = langfuse_secret or os.getenv("LANGFUSE_SECRET_KEY")
    public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_host = langfuse_host or os.getenv(
        "LANGFUSE_HOST", "https://cloud.langfuse.com"
    )

    if not (
        langfuse_host.startswith("http://") or langfuse_host.startswith("https://")
    ):
        # add http:// if unset, assume communicating over private network - e.g. render
        langfuse_host = "http://" + langfuse_host

    langfuse_release = os.getenv("LANGFUSE_RELEASE")
    langfuse_debug = os.getenv("LANGFUSE_DEBUG")

    parameters = {
        "public_key": public_key,
        "secret_key": secret_key,
        "host": langfuse_host,
        "release": langfuse_release,
        "debug": langfuse_debug,
        "flush_interval": LangFuseLogger._get_langfuse_flush_interval(
            flush_interval
        ),  # flush interval in seconds
    }

    if Version(langfuse.version.__version__) >= Version("2.6.0"):
        parameters["sdk_integration"] = "litellm"

    client = Langfuse(**parameters)

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
        self, langfuse_prompt_id: str, langfuse_client: LangfuseClass
    ) -> PROMPT_CLIENT:
        return langfuse_client.get_prompt(langfuse_prompt_id)

    def _compile_prompt(
        self,
        langfuse_prompt_client: PROMPT_CLIENT,
        langfuse_prompt_variables: Optional[dict],
        call_type: Union[Literal["completion"], Literal["text_completion"]],
    ) -> List[AllMessageValues]:
        compiled_prompt: Optional[Union[str, list]] = None

        if langfuse_prompt_variables is None:
            langfuse_prompt_variables = {}

        compiled_prompt = langfuse_prompt_client.compile(**langfuse_prompt_variables)

        if isinstance(compiled_prompt, str):
            compiled_prompt = [
                ChatCompletionSystemMessage(role="system", content=compiled_prompt)
            ]
        else:
            compiled_prompt = cast(List[AllMessageValues], compiled_prompt)

        return compiled_prompt

    def _get_optional_params_from_langfuse(
        self, langfuse_prompt_client: PROMPT_CLIENT
    ) -> dict:
        config = langfuse_prompt_client.config
        optional_params = {}
        for k, v in config.items():
            if k != "model":
                optional_params[k] = v
        return optional_params

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: str,
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> Tuple[
        str,
        List[AllMessageValues],
        dict,
    ]:
        return self.get_chat_completion_prompt(
            model,
            messages,
            non_default_params,
            prompt_id,
            prompt_variables,
            dynamic_callback_params,
        )

    def should_run_prompt_management(
        self,
        prompt_id: str,
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> bool:
        langfuse_client = langfuse_client_init(
            langfuse_public_key=dynamic_callback_params.get("langfuse_public_key"),
            langfuse_secret=dynamic_callback_params.get("langfuse_secret"),
            langfuse_host=dynamic_callback_params.get("langfuse_host"),
        )
        langfuse_prompt_client = self._get_prompt_from_id(
            langfuse_prompt_id=prompt_id, langfuse_client=langfuse_client
        )
        return langfuse_prompt_client is not None

    def _compile_prompt_helper(
        self,
        prompt_id: str,
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> PromptManagementClient:
        langfuse_client = langfuse_client_init(
            langfuse_public_key=dynamic_callback_params.get("langfuse_public_key"),
            langfuse_secret=dynamic_callback_params.get("langfuse_secret"),
            langfuse_host=dynamic_callback_params.get("langfuse_host"),
        )
        langfuse_prompt_client = self._get_prompt_from_id(
            langfuse_prompt_id=prompt_id, langfuse_client=langfuse_client
        )

        ## SET PROMPT
        compiled_prompt = self._compile_prompt(
            langfuse_prompt_client=langfuse_prompt_client,
            langfuse_prompt_variables=prompt_variables,
            call_type="completion",
        )

        template_model = langfuse_prompt_client.config.get("model")

        template_optional_params = self._get_optional_params_from_langfuse(
            langfuse_prompt_client
        )

        return PromptManagementClient(
            prompt_id=prompt_id,
            prompt_template=compiled_prompt,
            prompt_template_model=template_model,
            prompt_template_optional_params=template_optional_params,
            completed_messages=None,
        )

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        return run_async_function(
            self.async_log_success_event, kwargs, response_obj, start_time, end_time
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        standard_callback_dynamic_params = kwargs.get(
            "standard_callback_dynamic_params"
        )
        langfuse_logger_to_use = LangFuseHandler.get_langfuse_logger_for_request(
            globalLangfuseLogger=self,
            standard_callback_dynamic_params=standard_callback_dynamic_params,
            in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
        )
        langfuse_logger_to_use._old_log_event(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            user_id=kwargs.get("user", None),
            print_verbose=None,
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        standard_callback_dynamic_params = kwargs.get(
            "standard_callback_dynamic_params"
        )
        langfuse_logger_to_use = LangFuseHandler.get_langfuse_logger_for_request(
            globalLangfuseLogger=self,
            standard_callback_dynamic_params=standard_callback_dynamic_params,
            in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
        )
        standard_logging_object = cast(
            Optional[StandardLoggingPayload],
            kwargs.get("standard_logging_object", None),
        )
        if standard_logging_object is None:
            return
        langfuse_logger_to_use._old_log_event(
            start_time=start_time,
            end_time=end_time,
            response_obj=None,
            user_id=kwargs.get("user", None),
            print_verbose=None,
            status_message=standard_logging_object["error_str"],
            level="ERROR",
            kwargs=kwargs,
        )
