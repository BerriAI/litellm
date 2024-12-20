"""
Call Hook for LiteLLM Proxy which allows Langfuse prompt management.
"""

import os
import traceback
from typing import Literal, Optional, Union

from packaging.version import Version

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.secret_managers.main import str_to_bool


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

        # set the current langfuse project id in the environ
        # this is used by Alerting to link to the correct project
        try:
            project_id = self.Langfuse.client.projects.get().data[0].id
            os.environ["LANGFUSE_PROJECT_ID"] = project_id
        except Exception:
            project_id = None

        if os.getenv("UPSTREAM_LANGFUSE_SECRET_KEY") is not None:
            upstream_langfuse_debug = (
                str_to_bool(self.upstream_langfuse_debug)
                if self.upstream_langfuse_debug is not None
                else None
            )
            self.upstream_langfuse_secret_key = os.getenv(
                "UPSTREAM_LANGFUSE_SECRET_KEY"
            )
            self.upstream_langfuse_public_key = os.getenv(
                "UPSTREAM_LANGFUSE_PUBLIC_KEY"
            )
            self.upstream_langfuse_host = os.getenv("UPSTREAM_LANGFUSE_HOST")
            self.upstream_langfuse_release = os.getenv("UPSTREAM_LANGFUSE_RELEASE")
            self.upstream_langfuse_debug = os.getenv("UPSTREAM_LANGFUSE_DEBUG")
            self.upstream_langfuse = Langfuse(
                public_key=self.upstream_langfuse_public_key,
                secret_key=self.upstream_langfuse_secret_key,
                host=self.upstream_langfuse_host,
                release=self.upstream_langfuse_release,
                debug=(
                    upstream_langfuse_debug
                    if upstream_langfuse_debug is not None
                    else False
                ),
            )
        else:
            self.upstream_langfuse = None

    def _compile_prompt(
        self,
        metadata: dict,
        call_type: Union[Literal["completion"], Literal["text_completion"]],
    ) -> Optional[Union[str, list]]:
        compiled_prompt: Optional[Union[str, list]] = None
        if isinstance(metadata, dict):
            langfuse_prompt_id = metadata.get("langfuse_prompt_id")

            langfuse_prompt_variables = metadata.get("langfuse_prompt_variables") or {}
            if (
                langfuse_prompt_id
                and isinstance(langfuse_prompt_id, str)
                and isinstance(langfuse_prompt_variables, dict)
            ):
                langfuse_prompt = self.Langfuse.get_prompt(langfuse_prompt_id)
                compiled_prompt = langfuse_prompt.compile(**langfuse_prompt_variables)

        return compiled_prompt

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
        compiled_prompt: Optional[Union[str, list]] = None
        if call_type == "completion" or call_type == "text_completion":
            compiled_prompt = self._compile_prompt(metadata, call_type)
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
