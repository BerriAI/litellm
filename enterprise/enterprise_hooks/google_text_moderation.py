# +-----------------------------------------------+
#
#            Google Text Moderation
#   https://cloud.google.com/natural-language/docs/moderating-text
#
# +-----------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan


from typing import Optional, Literal, Union
import litellm, traceback, sys, uuid
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.integrations.custom_logger import CustomLogger
from fastapi import HTTPException
from litellm._logging import verbose_proxy_logger
from litellm.utils import (
    ModelResponse,
    EmbeddingResponse,
    ImageResponse,
    StreamingChoices,
)
from datetime import datetime
import aiohttp, asyncio


class _ENTERPRISE_GoogleTextModeration(CustomLogger):
    user_api_key_cache = None
    confidence_categories = [
        "toxic",
        "insult",
        "profanity",
        "derogatory",
        "sexual",
        "death_harm_and_tragedy",
        "violent",
        "firearms_and_weapons",
        "public_safety",
        "health",
        "religion_and_belief",
        "illicit_drugs",
        "war_and_conflict",
        "politics",
        "finance",
        "legal",
    ]  # https://cloud.google.com/natural-language/docs/moderating-text#safety_attribute_confidence_scores

    # Class variables or attributes
    def __init__(self):
        try:
            from google.cloud import language_v1
        except:
            raise Exception(
                "Missing google.cloud package. Run `pip install --upgrade google-cloud-language`"
            )

        # Instantiates a client
        self.client = language_v1.LanguageServiceClient()
        self.moderate_text_request = language_v1.ModerateTextRequest
        self.language_document = language_v1.types.Document
        self.document_type = language_v1.types.Document.Type.PLAIN_TEXT

        default_confidence_threshold = (
            litellm.google_moderation_confidence_threshold or 0.8
        )  # by default require a high confidence (80%) to fail

        for category in self.confidence_categories:
            if hasattr(litellm, f"{category}_confidence_threshold"):
                setattr(
                    self,
                    f"{category}_confidence_threshold",
                    getattr(litellm, f"{category}_confidence_threshold"),
                )
            else:
                setattr(
                    self,
                    f"{category}_confidence_threshold",
                    default_confidence_threshold,
                )
            set_confidence_value = getattr(
                self,
                f"{category}_confidence_threshold",
            )
            verbose_proxy_logger.info(
                f"Google Text Moderation: {category}_confidence_threshold: {set_confidence_value}"
            )

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except:
            pass

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "embeddings", "image_generation"],
    ):
        """
        - Calls Google's Text Moderation API
        - Rejects request if it fails safety check
        """
        if "messages" in data and isinstance(data["messages"], list):
            text = ""
            for m in data["messages"]:  # assume messages is a list
                if "content" in m and isinstance(m["content"], str):
                    text += m["content"]
            document = self.language_document(content=text, type_=self.document_type)

            request = self.moderate_text_request(
                document=document,
            )

            # Make the request
            response = self.client.moderate_text(request=request)
            for category in response.moderation_categories:
                category_name = category.name
                category_name = category_name.lower()
                category_name = category_name.replace("&", "and")
                category_name = category_name.replace(",", "")
                category_name = category_name.replace(
                    " ", "_"
                )  # e.g. go from 'Firearms & Weapons' to 'firearms_and_weapons'
                if category.confidence > getattr(
                    self, f"{category_name}_confidence_threshold"
                ):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": f"Violated content safety policy. Category={category}"
                        },
                    )
            # Handle the response
            return data


# google_text_moderation_obj = _ENTERPRISE_GoogleTextModeration()
# asyncio.run(
#     google_text_moderation_obj.async_moderation_hook(
#         data={"messages": [{"role": "user", "content": "Hey, how's it going?"}]}
#     )
# )
