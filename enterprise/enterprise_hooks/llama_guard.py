# +-------------------------------------------------------------+
#
#                   Llama Guard
#   https://huggingface.co/meta-llama/LlamaGuard-7b/tree/main
#
#           LLM for Content Moderation
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import sys
import os
from collections.abc import Iterable

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Optional, Literal
import litellm
import sys
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm.types.utils import (
    ModelResponse,
    Choices,
)

litellm.set_verbose = True


class _ENTERPRISE_LlamaGuard(CustomLogger):
    # Class variables or attributes
    def __init__(self, model_name: Optional[str] = None):
        _model = model_name or litellm.llamaguard_model_name
        if _model is None:
            raise ValueError("model_name not set for LlamaGuard")
        self.model = _model
        file_path = litellm.llamaguard_unsafe_content_categories
        data = None

        if file_path is not None:
            try:
                with open(file_path, "r") as file:
                    data = file.read()
            except FileNotFoundError:
                raise Exception(f"File not found. file_path={file_path}")
            except Exception as e:
                raise Exception(f"An error occurred: {str(e)}, file_path={file_path}")

        self.unsafe_content_categories = data

        verbose_proxy_logger.debug(
            f"self.unsafe_content_categories: {self.unsafe_content_categories}"
        )

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except Exception:
            pass

    def set_custom_prompt_template(self, messages: list):
        if self.unsafe_content_categories is not None and self.model is not None:
            role = "Agent" if len(messages) % 2 == 0 else "User"
            litellm.register_prompt_template(
                model=self.model,
                initial_prompt_value=f"""[INST] Task: Check if there is unsafe content in '{role}' messages in conversations according our safety policy with the below categories.

                <BEGIN UNSAFE CONTENT CATEGORIES>
                {self.unsafe_content_categories}
                <END UNSAFE CONTENT CATEGORIES>

                <BEGIN CONVERSATION>

                """,
                roles={
                    "system": {
                        "pre_message": "User: ",  # [OPTIONAL]
                        "post_message": "\n\n",  # [OPTIONAL]
                    },
                    "user": {
                        "pre_message": "User: ",  # [OPTIONAL]
                        "post_message": "\n\n",  # [OPTIONAL]
                    },
                    "assistant": {
                        "pre_message": "Agent: ",  # [OPTIONAL]
                        "post_message": "\n\n",  # [OPTIONAL]
                    },
                },
                final_prompt_value="""<END CONVERSATION>

        Provide your safety assessment for {role} in the above conversation:
        - First line must read 'safe' or 'unsafe'.
        - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]""",
            )

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
        - Calls the Llama Guard Endpoint
        - Rejects request if it fails safety check

        The llama guard prompt template is applied automatically in factory.py
        """
        if "messages" in data:
            safety_check_messages = data["messages"][
                -1
            ]  # get the last response - llama guard has a 4k token limit
            response = await litellm.acompletion(
                model=self.model,
                messages=[safety_check_messages],
                hf_model_name="meta-llama/LlamaGuard-7b",
            )

            if (
                isinstance(response, ModelResponse)
                and isinstance(response.choices[0], Choices)
                and response.choices[0].message.content is not None
                and isinstance(response.choices[0].message.content, Iterable)
                and "unsafe" in response.choices[0].message.content
            ):
                raise HTTPException(
                    status_code=400, detail={"error": "Violated content safety policy"}
                )

        return data
