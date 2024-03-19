# +------------------------------------+
#
#        Prompt Injection Detection
#
# +------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan
## Reject a call if it contains a prompt injection attack.


from typing import Optional, Literal
import litellm
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from litellm._logging import verbose_proxy_logger
from litellm.utils import get_formatted_prompt
from fastapi import HTTPException
import json, traceback, re
from difflib import SequenceMatcher
from typing import List


class _ENTERPRISE_PromptInjectionDetection(CustomLogger):
    # Class variables or attributes
    def __init__(self):
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

    def print_verbose(self, print_statement, level: Literal["INFO", "DEBUG"] = "DEBUG"):
        if level == "INFO":
            verbose_proxy_logger.info(print_statement)
        elif level == "DEBUG":
            verbose_proxy_logger.debug(print_statement)

        if litellm.set_verbose is True:
            print(print_statement)  # noqa

    def generate_injection_keywords(self) -> List[str]:
        combinations = []
        for verb in self.verbs:
            for adj in self.adjectives:
                for prep in self.prepositions:
                    phrase = " ".join(filter(None, [verb, adj, prep])).strip()
                    combinations.append(phrase.lower())
        return combinations

    def check_user_input_similarity(
        self, user_input: str, similarity_threshold: float = 0.7
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
                    self.print_verbose(
                        print_statement=f"Rejected user input - {user_input}. {match_ratio} similar to {keyword}",
                        level="INFO",
                    )
                    return True  # Found a highly similar substring
        return False  # No substring crossed the threshold

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
            self.print_verbose(f"Inside Prompt Injection Detection Pre-Call Hook")
            try:
                assert call_type in [
                    "completion",
                    "embeddings",
                    "image_generation",
                    "moderation",
                    "audio_transcription",
                ]
            except Exception as e:
                self.print_verbose(
                    f"Call Type - {call_type}, not in accepted list - ['completion','embeddings','image_generation','moderation','audio_transcription']"
                )
                return data
            formatted_prompt = get_formatted_prompt(data=data, call_type=call_type)  # type: ignore

            is_prompt_attack = self.check_user_input_similarity(
                user_input=formatted_prompt
            )

            if is_prompt_attack == True:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Rejected message. This is a prompt injection attack."
                    },
                )

            return data

        except HTTPException as e:
            raise e
        except Exception as e:
            traceback.print_exc()
