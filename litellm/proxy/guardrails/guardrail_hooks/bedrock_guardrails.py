# +-------------------------------------------------------------+
#
#           Use Bedrock Guardrails for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
import json
import sys
import traceback
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import aiohttp
import httpx
from fastapi import HTTPException

import litellm
from litellm import get_secret
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.logging_utils import (
    convert_litellm_response_object_to_str,
)
from litellm.llms.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    _get_async_httpx_client,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_helpers import should_proceed_based_on_metadata
from litellm.types.guardrails import (
    BedrockContentItem,
    BedrockRequest,
    BedrockTextContent,
    GuardrailEventHooks,
)

GUARDRAIL_NAME = "bedrock"


class BedrockGuardrail(CustomGuardrail, BaseAWSLLM):
    def __init__(
        self,
        guardrailIdentifier: Optional[str] = None,
        guardrailVersion: Optional[str] = None,
        **kwargs,
    ):
        self.async_handler = _get_async_httpx_client()
        self.guardrailIdentifier = guardrailIdentifier
        self.guardrailVersion = guardrailVersion

        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)

    def convert_to_bedrock_format(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        response: Optional[Union[Any, litellm.ModelResponse]] = None,
    ) -> BedrockRequest:
        bedrock_request: BedrockRequest = BedrockRequest(source="INPUT")
        bedrock_request_content: List[BedrockContentItem] = []

        if messages:
            for message in messages:
                content = message.get("content")
                if isinstance(content, str):
                    bedrock_content_item = BedrockContentItem(
                        text=BedrockTextContent(text=content)
                    )
                    bedrock_request_content.append(bedrock_content_item)

            bedrock_request["content"] = bedrock_request_content
        if response:
            bedrock_request["source"] = "OUTPUT"
            if isinstance(response, litellm.ModelResponse):
                for choice in response.choices:
                    if isinstance(choice, litellm.Choices):
                        if choice.message.content and isinstance(
                            choice.message.content, str
                        ):
                            bedrock_content_item = BedrockContentItem(
                                text=BedrockTextContent(text=choice.message.content)
                            )
                            bedrock_request_content.append(bedrock_content_item)
                bedrock_request["content"] = bedrock_request_content
        return bedrock_request

    #### CALL HOOKS - proxy only ####
    def _load_credentials(
        self,
    ):
        try:
            from botocore.credentials import Credentials
        except ImportError as e:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
        ## CREDENTIALS ##
        # pop aws_secret_access_key, aws_access_key_id, aws_session_token, aws_region_name from kwargs, since completion calls fail with them
        aws_secret_access_key = self.optional_params.pop("aws_secret_access_key", None)
        aws_access_key_id = self.optional_params.pop("aws_access_key_id", None)
        aws_session_token = self.optional_params.pop("aws_session_token", None)
        aws_region_name = self.optional_params.pop("aws_region_name", None)
        aws_role_name = self.optional_params.pop("aws_role_name", None)
        aws_session_name = self.optional_params.pop("aws_session_name", None)
        aws_profile_name = self.optional_params.pop("aws_profile_name", None)
        aws_bedrock_runtime_endpoint = self.optional_params.pop(
            "aws_bedrock_runtime_endpoint", None
        )  # https://bedrock-runtime.{region_name}.amazonaws.com
        aws_web_identity_token = self.optional_params.pop(
            "aws_web_identity_token", None
        )
        aws_sts_endpoint = self.optional_params.pop("aws_sts_endpoint", None)

        ### SET REGION NAME ###
        if aws_region_name is None:
            # check env #
            litellm_aws_region_name = get_secret("AWS_REGION_NAME", None)

            if litellm_aws_region_name is not None and isinstance(
                litellm_aws_region_name, str
            ):
                aws_region_name = litellm_aws_region_name

            standard_aws_region_name = get_secret("AWS_REGION", None)
            if standard_aws_region_name is not None and isinstance(
                standard_aws_region_name, str
            ):
                aws_region_name = standard_aws_region_name

            if aws_region_name is None:
                aws_region_name = "us-west-2"

        credentials: Credentials = self.get_credentials(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region_name=aws_region_name,
            aws_session_name=aws_session_name,
            aws_profile_name=aws_profile_name,
            aws_role_name=aws_role_name,
            aws_web_identity_token=aws_web_identity_token,
            aws_sts_endpoint=aws_sts_endpoint,
        )
        return credentials, aws_region_name

    def _prepare_request(
        self,
        credentials,
        data: BedrockRequest,
        optional_params: dict,
        aws_region_name: str,
        extra_headers: Optional[dict] = None,
    ):
        try:
            import boto3
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError as e:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)
        api_base = f"https://bedrock-runtime.{aws_region_name}.amazonaws.com/guardrail/{self.guardrailIdentifier}/version/{self.guardrailVersion}/apply"

        encoded_data = json.dumps(data).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers = {"Content-Type": "application/json", **extra_headers}

        request = AWSRequest(
            method="POST", url=api_base, data=encoded_data, headers=headers
        )
        sigv4.add_auth(request)
        if (
            extra_headers is not None and "Authorization" in extra_headers
        ):  # prevent sigv4 from overwriting the auth header
            request.headers["Authorization"] = extra_headers["Authorization"]

        prepped_request = request.prepare()

        return prepped_request

    async def make_bedrock_api_request(
        self, kwargs: dict, response: Optional[Union[Any, litellm.ModelResponse]] = None
    ):

        credentials, aws_region_name = self._load_credentials()
        request_data: BedrockRequest = self.convert_to_bedrock_format(
            messages=kwargs.get("messages"), response=response
        )
        prepared_request = self._prepare_request(
            credentials=credentials,
            data=request_data,
            optional_params=self.optional_params,
            aws_region_name=aws_region_name,
        )
        verbose_proxy_logger.debug(
            "Bedrock AI request body: %s, url %s, headers: %s",
            request_data,
            prepared_request.url,
            prepared_request.headers,
        )
        _json_data = json.dumps(request_data)  # type: ignore
        response = await self.async_handler.post(
            url=prepared_request.url,
            json=request_data,  # type: ignore
            headers=prepared_request.headers,
        )
        verbose_proxy_logger.debug("Bedrock AI response: %s", response.text)
        if response.status_code == 200:
            # check if the response was flagged
            _json_response = response.json()
            if _json_response.get("action") == "GUARDRAIL_INTERVENED":
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Violated guardrail policy",
                        "bedrock_guardrail_response": _json_response,
                    },
                )
        else:
            verbose_proxy_logger.error(
                "Bedrock AI: error in response. Status code: %s, response: %s",
                response.status_code,
                response.text,
            )

    async def async_moderation_hook(  ### 👈 KEY CHANGE ###
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal["completion", "embeddings", "image_generation"],
    ):
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )
        from litellm.types.guardrails import GuardrailEventHooks

        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return

        new_messages: Optional[List[dict]] = data.get("messages")
        if new_messages is not None:
            await self.make_bedrock_api_request(kwargs=data)
            add_guardrail_to_applied_guardrails_header(
                request_data=data, guardrail_name=self.guardrail_name
            )
        else:
            verbose_proxy_logger.warning(
                "Bedrock AI: not running guardrail. No messages in data"
            )
            pass

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )
        from litellm.types.guardrails import GuardrailEventHooks

        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            return

        new_messages: Optional[List[dict]] = data.get("messages")
        if new_messages is not None:
            await self.make_bedrock_api_request(kwargs=data, response=response)
            add_guardrail_to_applied_guardrails_header(
                request_data=data, guardrail_name=self.guardrail_name
            )
        else:
            verbose_proxy_logger.warning(
                "Bedrock AI: not running guardrail. No messages in data"
            )
