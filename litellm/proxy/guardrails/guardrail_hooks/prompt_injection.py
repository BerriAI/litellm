# +------------------------------------+
#
#        Prompt Injection Detection
#
# +------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan
# ## Reject a call if it contains a prompt injection attack.


from difflib import SequenceMatcher
from typing import List, Literal, Optional, Union

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.constants import DEFAULT_PROMPT_INJECTION_SIMILARITY_THRESHOLD
from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
from litellm.litellm_core_utils.prompt_templates.factory import (
    prompt_injection_detection_default_pt,
)
from litellm.proxy._types import LiteLLMPromptInjectionParams, UserAPIKeyAuth
from litellm.router import Router
from litellm.utils import get_formatted_prompt
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams

class PromptInjectionGuardrail(CustomGuardrail):
    def __init__(
        self,
        guardrail_name: str = "prompt_injection",
        default_on: bool = False,
        event_hook: Optional[Union[GuardrailEventHooks, List[GuardrailEventHooks]]] = None,
        litellm_params: Optional[LitellmParams] = None,
        **kwargs,
    ):
        super().__init__(
            guardrail_name=guardrail_name,
            default_on=default_on,
            event_hook=event_hook,
            **kwargs,
        )
        self.prompt_injection_params = None
        if litellm_params and litellm_params.guardrail == "prompt_injection":
            # Map LitellmParams to LiteLLMPromptInjectionParams
            # Filter out standard LitellmParams fields to get the injection-specific ones
            params_dict = litellm_params.dict(exclude={"guardrail", "mode", "default_on"}, exclude_none=True)
            self.prompt_injection_params = LiteLLMPromptInjectionParams(**params_dict)
            
        self.llm_router: Optional[Router] = None

        self.verbs = [
            "Ignore",
            "Disregard",
            "Skip",
            "Forget",
            "Neglect",
            "Overlook",
            "Omit",
            "Bypass",
            "Pay no attention to",
            "Do not follow",
            "Do not obey",
        ]
        self.adjectives = [
            "",
            "prior",
            "previous",
            "preceding",
            "above",
            "foregoing",
            "earlier",
            "initial",
        ]
        self.prepositions = [
            "",
            "and start over",
            "and start anew",
            "and begin afresh",
            "and start from scratch",
        ]

    def update_environment(self, router: Optional[Router] = None):
        self.llm_router = router

        if (
            self.prompt_injection_params is not None
            and self.prompt_injection_params.llm_api_check is True
        ):
            if self.llm_router is None:
                raise Exception(
                    "PromptInjectionDetection: Model List not set. Required for Prompt Injection detection."
                )

            verbose_proxy_logger.debug(
                f"model_names: {self.llm_router.model_names}; self.prompt_injection_params.llm_api_name: {self.prompt_injection_params.llm_api_name}"
            )
            if (
                self.prompt_injection_params.llm_api_name is None
                or self.prompt_injection_params.llm_api_name
                not in self.llm_router.model_names
            ):
                raise Exception(
                    "PromptInjectionDetection: Invalid LLM API Name. LLM API Name must be a 'model_name' in 'model_list'."
                )

    def generate_injection_keywords(self) -> List[str]:
        combinations = []
        for verb in self.verbs:
            for adj in self.adjectives:
                for prep in self.prepositions:
                    phrase = " ".join(filter(None, [verb, adj, prep])).strip()
                    if (
                        len(phrase.split()) > 2
                    ):  # additional check to ensure more than 2 words
                        combinations.append(phrase.lower())
        return combinations

    def check_user_input_similarity(
        self,
        user_input: str,
        similarity_threshold: float = DEFAULT_PROMPT_INJECTION_SIMILARITY_THRESHOLD,
    ) -> bool:
        user_input_lower = user_input.lower()
        keywords = self.generate_injection_keywords()

        for keyword in keywords:
            # Calculate the length of the keyword to extract substrings of the same length from user input
            keyword_length = len(keyword)

            for i in range(len(user_input_lower) - keyword_length + 1):
                # Extract a substring of the same length as the keyword
                substring = user_input_lower[i : i + keyword_length]

                # Calculate similarity
                match_ratio = SequenceMatcher(None, substring, keyword).ratio()
                if match_ratio > similarity_threshold:
                    verbose_proxy_logger.info(
                        f"Rejected user input - {user_input}. {match_ratio} similar to {keyword}"
                    )
                    return True  # Found a highly similar substring
        return False  # No substring crossed the threshold

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,  # "completion", "embeddings", "image_generation", "moderation"
    ):
        try:
            """
            - check if user id part of call
            - check if user id part of blocked list
            """
            verbose_proxy_logger.debug("Inside Prompt Injection Detection Pre-Call Hook")
            try:
                assert call_type in [
                    "acompletion",
                    "completion",
                    "text_completion",
                    "embeddings",
                    "image_generation",
                    "moderation",
                    "audio_transcription",
                ]
            except Exception:
                verbose_proxy_logger.debug(
                    f"Call Type - {call_type}, not in accepted list - ['completion','embeddings','image_generation','moderation','audio_transcription']"
                )
                return data
            formatted_prompt = get_formatted_prompt(data=data, call_type=call_type)  # type: ignore

            is_prompt_attack = False

            if self.prompt_injection_params is not None:
                # 1. check if heuristics check turned on
                if self.prompt_injection_params.heuristics_check is True:
                    is_prompt_attack = self.check_user_input_similarity(
                        user_input=formatted_prompt
                    )
                    if is_prompt_attack is True:
                        self.raise_passthrough_exception(
                            violation_message="Rejected message. This is a prompt injection attack.",
                            request_data=data,
                        )
                # 2. check if vector db similarity check turned on [TODO] Not Implemented yet
                if self.prompt_injection_params.vector_db_check is True:
                    pass
            else:
                is_prompt_attack = self.check_user_input_similarity(
                    user_input=formatted_prompt
                )

            if is_prompt_attack is True:
                                        self.raise_passthrough_exception(
                            violation_message="Rejected message. This is a prompt injection attack.",
                            request_data=data,
                        )
            return data

        except HTTPException as e:
            if (
                e.status_code == 400
                and isinstance(e.detail, dict)
                and "error" in e.detail  # type: ignore
                and self.prompt_injection_params is not None
                and self.prompt_injection_params.reject_as_response
            ):
                return e.detail.get("error")
            raise e
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.guardrails.guardrail_hooks.prompt_injection.py::async_pre_call_hook(): Exception occured - {}".format(
                    str(e)
                )
            )
            raise e

    @log_guardrail_information
    async def async_moderation_hook(  # type: ignore
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "acompletion",
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> Optional[bool]:
        verbose_proxy_logger.debug(
            f"IN ASYNC MODERATION HOOK - self.prompt_injection_params = {self.prompt_injection_params}"
        )

        if self.prompt_injection_params is None:
            return None

        formatted_prompt = get_formatted_prompt(data=data, call_type=call_type)  # type: ignore
        is_prompt_attack = False

        prompt_injection_system_prompt = getattr(
            self.prompt_injection_params,
            "llm_api_system_prompt",
            prompt_injection_detection_default_pt(),
        )

        # 3. check if llm api check turned on
        if (
            self.prompt_injection_params.llm_api_check is True
            and self.prompt_injection_params.llm_api_name is not None
            and self.llm_router is not None
        ):
            # make a call to the llm api
            response = await self.llm_router.acompletion(
                model=self.prompt_injection_params.llm_api_name,
                messages=[
                    {
                        "role": "system",
                        "content": prompt_injection_system_prompt,
                    },
                    {"role": "user", "content": formatted_prompt},
                ],
            )

            verbose_proxy_logger.debug(f"Received LLM Moderation response: {response}")
            verbose_proxy_logger.debug(
                f"llm_api_fail_call_string: {self.prompt_injection_params.llm_api_fail_call_string}"
            )
            if isinstance(response, litellm.ModelResponse) and isinstance(
                response.choices[0], litellm.Choices
            ):
                if self.prompt_injection_params.llm_api_fail_call_string in response.choices[0].message.content:  # type: ignore
                    is_prompt_attack = True

        if is_prompt_attack is True:
            self.raise_passthrough_exception(
                            violation_message="Rejected message. This is a prompt injection attack.",
                            request_data=data,
                        )

        return is_prompt_attack

def initialize_prompt_injection(
    litellm_params: LitellmParams,
    guardrail: dict,
    llm_router: Optional[Router] = None,
) -> PromptInjectionGuardrail:
    return PromptInjectionGuardrail(
        guardrail_name=guardrail.get("guardrail_name", "prompt_injection"),
        default_on=guardrail.get("default_on", False),
        event_hook=litellm_params.mode,
        litellm_params=litellm_params,
    )
