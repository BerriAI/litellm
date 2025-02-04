# +------------------------+
#
#            LLM Guard
#   https://llm-guard.com/
#
# +------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan
## This provides an LLM Guard Integration for content moderation on the proxy

from typing import Optional, Literal
import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
import aiohttp
from litellm.utils import get_formatted_prompt
from litellm.secret_managers.main import get_secret_str

litellm.set_verbose = True


class _ENTERPRISE_LLMGuard(CustomLogger):
    # Class variables or attributes
    def __init__(
        self,
        mock_testing: bool = False,
        mock_redacted_text: Optional[dict] = None,
    ):
        self.mock_redacted_text = mock_redacted_text
        self.llm_guard_mode = litellm.llm_guard_mode
        if mock_testing == True:  # for testing purposes only
            return
        self.llm_guard_api_base = get_secret_str("LLM_GUARD_API_BASE", None)
        if self.llm_guard_api_base is None:
            raise Exception("Missing `LLM_GUARD_API_BASE` from environment")
        elif not self.llm_guard_api_base.endswith("/"):
            self.llm_guard_api_base += "/"

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except Exception:
            pass

    async def moderation_check(self, text: str):
        """
        [TODO] make this more performant for high-throughput scenario
        """
        try:
            async with aiohttp.ClientSession() as session:
                if self.mock_redacted_text is not None:
                    redacted_text = self.mock_redacted_text
                else:
                    # Make the first request to /analyze
                    analyze_url = f"{self.llm_guard_api_base}analyze/prompt"
                    verbose_proxy_logger.debug("Making request to: %s", analyze_url)
                    analyze_payload = {"prompt": text}
                    redacted_text = None
                    async with session.post(
                        analyze_url, json=analyze_payload
                    ) as response:
                        redacted_text = await response.json()
                verbose_proxy_logger.info(
                    f"LLM Guard: Received response - {redacted_text}"
                )
                if redacted_text is not None:
                    if (
                        redacted_text.get("is_valid", None) is not None
                        and redacted_text["is_valid"] != True
                    ):
                        raise HTTPException(
                            status_code=400,
                            detail={"error": "Violated content safety policy"},
                        )
                    else:
                        pass
                else:
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "error": f"Invalid content moderation response: {redacted_text}"
                        },
                    )
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.enterprise.enterprise_hooks.llm_guard::moderation_check - Exception occurred - {}".format(
                    str(e)
                )
            )
            raise e

    def should_proceed(self, user_api_key_dict: UserAPIKeyAuth, data: dict) -> bool:
        if self.llm_guard_mode == "key-specific":
            # check if llm guard enabled for specific keys only
            self.print_verbose(
                f"user_api_key_dict.permissions: {user_api_key_dict.permissions}"
            )
            if (
                user_api_key_dict.permissions.get("enable_llm_guard_check", False)
                == True
            ):
                return True
        elif self.llm_guard_mode == "all":
            return True
        elif self.llm_guard_mode == "request-specific":
            self.print_verbose(f"received metadata: {data.get('metadata', {})}")
            metadata = data.get("metadata", {})
            permissions = metadata.get("permissions", {})
            if (
                "enable_llm_guard_check" in permissions
                and permissions["enable_llm_guard_check"] == True
            ):
                return True
        return False

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ):
        """
        - Calls the LLM Guard Endpoint
        - Rejects request if it fails safety check
        - Use the sanitized prompt returned
            - LLM Guard can handle things like PII Masking, etc.
        """
        self.print_verbose(
            f"Inside LLM Guard Pre-Call Hook - llm_guard_mode={self.llm_guard_mode}"
        )

        _proceed = self.should_proceed(user_api_key_dict=user_api_key_dict, data=data)
        if _proceed == False:
            return

        self.print_verbose("Makes LLM Guard Check")
        try:
            assert call_type in [
                "completion",
                "embeddings",
                "image_generation",
                "moderation",
                "audio_transcription",
            ]
        except Exception:
            self.print_verbose(
                f"Call Type - {call_type}, not in accepted list - ['completion','embeddings','image_generation','moderation','audio_transcription']"
            )
            return data

        formatted_prompt = get_formatted_prompt(data=data, call_type=call_type)  # type: ignore
        self.print_verbose(f"LLM Guard, formatted_prompt: {formatted_prompt}")
        return await self.moderation_check(text=formatted_prompt)

    async def async_post_call_streaming_hook(
        self, user_api_key_dict: UserAPIKeyAuth, response: str
    ):
        if response is not None:
            await self.moderation_check(text=response)

        return response


# llm_guard = _ENTERPRISE_LLMGuard()

# asyncio.run(
#     llm_guard.async_moderation_hook(
#         data={"messages": [{"role": "user", "content": "Hey how's it going?"}]}
#     )
# )
