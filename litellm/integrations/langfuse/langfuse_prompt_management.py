"""
Call Hook for LiteLLM Proxy which allows Langfuse prompt management.
"""

import os
import traceback
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple, Union, cast

from packaging.version import Version

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.secret_managers.main import str_to_bool
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from langfuse.client import ChatPromptClient, TextPromptClient

    PROMPT_CLIENT = Union[TextPromptClient, ChatPromptClient]
else:
    PROMPT_CLIENT = Any


class LangfusePromptManagement(CustomLogger):
    def __init__(
        self,
        langfuse_public_key=None,
        langfuse_secret=None,
        langfuse_host=None,
        flush_interval=1,
    ):
        try:
            import langfuse
            from langfuse import Langfuse
        except Exception as e:
            raise Exception(
                f"\033[91mLangfuse not installed, try running 'pip install langfuse' to fix this error: {e}\n{traceback.format_exc()}\033[0m"
            )
        # Instance variables
        self.secret_key = langfuse_secret or os.getenv("LANGFUSE_SECRET_KEY")
        self.public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_host = langfuse_host or os.getenv(
            "LANGFUSE_HOST", "https://cloud.langfuse.com"
        )
        if not (
            self.langfuse_host.startswith("http://")
            or self.langfuse_host.startswith("https://")
        ):
            # add http:// if unset, assume communicating over private network - e.g. render
            self.langfuse_host = "http://" + self.langfuse_host
        self.langfuse_release = os.getenv("LANGFUSE_RELEASE")
        self.langfuse_debug = os.getenv("LANGFUSE_DEBUG")
        self.langfuse_flush_interval = (
            os.getenv("LANGFUSE_FLUSH_INTERVAL") or flush_interval
        )

        parameters = {
            "public_key": self.public_key,
            "secret_key": self.secret_key,
            "host": self.langfuse_host,
            "release": self.langfuse_release,
            "debug": self.langfuse_debug,
            "flush_interval": self.langfuse_flush_interval,  # flush interval in seconds
        }

        if Version(langfuse.version.__version__) >= Version("2.6.0"):
            parameters["sdk_integration"] = "litellm"

        self.Langfuse = Langfuse(**parameters)

    def _get_prompt_from_id(self, langfuse_prompt_id: str) -> PROMPT_CLIENT:
        return self.Langfuse.get_prompt(langfuse_prompt_id)

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
            return

        if langfuse_prompt_id is None:
            return

        prompt_client = self._get_prompt_from_id(langfuse_prompt_id=langfuse_prompt_id)
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
        langfuse_prompt_client = self._get_prompt_from_id(prompt_id)

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
