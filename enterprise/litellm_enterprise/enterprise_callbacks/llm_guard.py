# +------------------------+
#
#            LLM Guard
#   https://llm-guard.com/
#
# +------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan
## This provides an LLM Guard Integration for content moderation on the proxy

import asyncio
from typing import Optional

import aiohttp
from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import CallTypesLiteral


class _ENTERPRISE_LLMGuard(CustomLogger):
    # Class variables or attributes
    def __init__(
        self,
        mock_testing: bool = False,
        mock_redacted_text: Optional[dict] = None,
    ):
        self.mock_redacted_text = mock_redacted_text
        self.llm_guard_mode = litellm.llm_guard_mode
        if mock_testing is True:  # for testing purposes only
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

    async def moderation_check(self, text: str) -> str:
        """
        Runs the LLM Guard moderation check on ``text``.

        Raises an HTTPException when the content violates the safety policy;
        otherwise returns the sanitized prompt from LLM Guard, falling back to
        the original text when the API does not provide one.

        [TODO] make this more performant for high-throughput scenario
        """
        try:
            if self.mock_redacted_text is not None:
                redacted_text = self.mock_redacted_text
            else:
                analyze_url = f"{self.llm_guard_api_base}analyze/prompt"
                verbose_proxy_logger.debug("Making request to: %s", analyze_url)
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        analyze_url, json={"prompt": text}
                    ) as response:
                        redacted_text = await response.json()
            verbose_proxy_logger.debug(
                f"LLM Guard: Received response - {redacted_text}"
            )
            if redacted_text is None:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": f"Invalid content moderation response: {redacted_text}"
                    },
                )
            if redacted_text.get("is_valid", None) is False:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "Violated content safety policy"},
                )
            sanitized_prompt = redacted_text.get("sanitized_prompt")
            return sanitized_prompt if isinstance(sanitized_prompt, str) else text
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
                is True
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
                and permissions["enable_llm_guard_check"] is True
            ):
                return True
        return False

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: CallTypesLiteral,
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
        if _proceed is False:
            return

        self.print_verbose("Makes LLM Guard Check")
        if call_type not in [
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ]:
            self.print_verbose(
                f"Call Type - {call_type}, not in accepted list - ['completion','embeddings','image_generation','moderation','audio_transcription']"
            )
            return data

        return await self._moderate_request(data=data)

    async def _moderate_request(self, data: dict) -> dict:
        """
        Sanitizes the request in place using the prompt returned by LLM Guard so
        the provider-bound request carries the redacted content, then returns it.
        """
        messages = data.get("messages")
        if messages is not None:
            data["messages"] = list(
                await asyncio.gather(
                    *(self._moderate_message(message) for message in messages)
                )
            )
            return data

        input_ = data.get("input")
        if input_ is not None:
            data["input"] = await self._moderate_input(input_)
            return data

        prompt = data.get("prompt")
        if isinstance(prompt, str):
            data["prompt"] = await self.moderation_check(text=prompt)
        return data

    async def _moderate_message(self, message: dict) -> dict:
        content = message.get("content")
        if isinstance(content, str):
            return {**message, "content": await self.moderation_check(text=content)}
        if isinstance(content, list):
            return {
                **message,
                "content": list(
                    await asyncio.gather(
                        *(self._moderate_content_part(part) for part in content)
                    )
                ),
            }
        return message

    async def _moderate_content_part(self, part: dict) -> dict:
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            return {**part, "text": await self.moderation_check(text=part["text"])}
        return part

    async def _moderate_input(self, input_: object) -> object:
        if isinstance(input_, str):
            return await self.moderation_check(text=input_)
        if isinstance(input_, list):
            return [
                await self.moderation_check(text=item)
                if isinstance(item, str)
                else item
                for item in input_
            ]
        return input_

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
