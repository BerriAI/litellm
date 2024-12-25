"""
Call Hook for LiteLLM Proxy which allows Langfuse prompt management.
"""

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, Union, cast

from packaging.version import Version
from typing_extensions import TypeAlias

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardCallbackDynamicParams, StandardLoggingPayload

from .langfuse import LangFuseLogger

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.client import ChatPromptClient, TextPromptClient

    LangfuseClass: TypeAlias = Langfuse

    PROMPT_CLIENT = Union[TextPromptClient, ChatPromptClient]
else:
    PROMPT_CLIENT = Any
    LangfuseClass = Any


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
    langfuse_flush_interval = os.getenv("LANGFUSE_FLUSH_INTERVAL") or flush_interval

    parameters = {
        "public_key": public_key,
        "secret_key": secret_key,
        "host": langfuse_host,
        "release": langfuse_release,
        "debug": langfuse_debug,
        "flush_interval": langfuse_flush_interval,  # flush interval in seconds
    }

    if Version(langfuse.version.__version__) >= Version("2.6.0"):
        parameters["sdk_integration"] = "litellm"

    client = Langfuse(**parameters)

    return client


class LangfusePromptManagement(LangFuseLogger, CustomLogger):
    def __init__(
        self,
        langfuse_public_key=None,
        langfuse_secret=None,
        langfuse_host=None,
        flush_interval=1,
    ):
        self.Langfuse = langfuse_client_init(
            langfuse_public_key=langfuse_public_key,
            langfuse_secret=langfuse_secret,
            langfuse_host=langfuse_host,
            flush_interval=flush_interval,
        )

    def _get_prompt_from_id(
        self, langfuse_prompt_id: str, langfuse_client: LangfuseClass
    ) -> PROMPT_CLIENT:
        return langfuse_client.get_prompt(langfuse_prompt_id)

    def _compile_prompt(
        self,
        langfuse_prompt_client: PROMPT_CLIENT,
        langfuse_prompt_variables: Optional[dict],
        call_type: Union[Literal["completion"], Literal["text_completion"]],
    ) -> Optional[Union[str, list]]:
        compiled_prompt: Optional[Union[str, list]] = None

        if langfuse_prompt_variables is None:
            langfuse_prompt_variables = {}

        compiled_prompt = langfuse_prompt_client.compile(**langfuse_prompt_variables)

        return compiled_prompt

    def _get_model_from_prompt(
        self, langfuse_prompt_client: PROMPT_CLIENT, model: str
    ) -> str:
        config = langfuse_prompt_client.config
        if "model" in config:
            return config["model"]
        else:
            return model.replace("langfuse/", "")

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Union[
            Literal["completion"],
            Literal["text_completion"],
            Literal["embeddings"],
            Literal["image_generation"],
            Literal["moderation"],
            Literal["audio_transcription"],
            Literal["pass_through_endpoint"],
            Literal["rerank"],
        ],
    ) -> Union[Exception, str, dict, None]:

        metadata = data.get("metadata") or {}

        if isinstance(metadata, dict):
            langfuse_prompt_id = cast(Optional[str], metadata.get("langfuse_prompt_id"))

            langfuse_prompt_variables = cast(
                Optional[dict], metadata.get("langfuse_prompt_variables") or {}
            )
        else:
            return None

        if langfuse_prompt_id is None:
            return None

        prompt_client = self._get_prompt_from_id(
            langfuse_prompt_id=langfuse_prompt_id, langfuse_client=self.Langfuse
        )
        compiled_prompt: Optional[Union[str, list]] = None
        if call_type == "completion" or call_type == "text_completion":
            compiled_prompt = self._compile_prompt(
                langfuse_prompt_client=prompt_client,
                langfuse_prompt_variables=langfuse_prompt_variables,
                call_type=call_type,
            )
        if compiled_prompt is None:
            return await super().async_pre_call_hook(
                user_api_key_dict, cache, data, call_type
            )
        if call_type == "completion":
            if isinstance(compiled_prompt, list):
                data["messages"] = compiled_prompt + data["messages"]
            else:
                data["messages"] = [
                    {"role": "system", "content": compiled_prompt}
                ] + data["messages"]
        elif call_type == "text_completion" and isinstance(compiled_prompt, str):
            data["prompt"] = compiled_prompt + "\n" + data["prompt"]

        verbose_proxy_logger.debug(
            f"LangfusePromptManagement.async_pre_call_hook compiled_prompt: {compiled_prompt}, type: {type(compiled_prompt)}"
        )

        return await super().async_pre_call_hook(
            user_api_key_dict, cache, data, call_type
        )

    def get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        headers: dict,
        prompt_id: str,
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> Tuple[
        str,
        List[AllMessageValues],
        dict,
    ]:
        if prompt_id is None:
            raise ValueError(
                "Langfuse prompt id is required. Pass in as parameter 'langfuse_prompt_id'"
            )
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

        if compiled_prompt is None:
            raise ValueError(f"Langfuse prompt not found. Prompt id={prompt_id}")
        if isinstance(compiled_prompt, list):
            messages = compiled_prompt
        elif isinstance(compiled_prompt, str):
            messages = [{"role": "user", "content": compiled_prompt}]
        else:
            raise ValueError(
                f"Langfuse prompt is not a list or string. Prompt id={prompt_id}, compiled_prompt type={type(compiled_prompt)}"
            )

        ## SET MODEL
        model = self._get_model_from_prompt(langfuse_prompt_client, model)

        return model, messages, non_default_params

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._old_log_event(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=start_time,
            end_time=end_time,
            user_id=kwargs.get("user", None),
            print_verbose=None,
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        standard_logging_object = cast(
            Optional[StandardLoggingPayload],
            kwargs.get("standard_logging_object", None),
        )
        if standard_logging_object is None:
            return
        self._old_log_event(
            start_time=start_time,
            end_time=end_time,
            response_obj=None,
            user_id=kwargs.get("user", None),
            print_verbose=None,
            status_message=standard_logging_object["error_str"],
            level="ERROR",
            kwargs=kwargs,
        )
