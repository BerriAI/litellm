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
import json
import sys
from typing import Any, List, Literal, Optional, Tuple, Union

from fastapi import HTTPException

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.secret_managers.main import get_secret
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockContentItem,
    BedrockGuardrailOutput,
    BedrockGuardrailResponse,
    BedrockRequest,
    BedrockTextContent,
)
from litellm.types.utils import ModelResponse

GUARDRAIL_NAME = "bedrock"


class BedrockGuardrail(CustomGuardrail, BaseAWSLLM):
    def __init__(
        self,
        guardrailIdentifier: Optional[str] = None,
        guardrailVersion: Optional[str] = None,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.guardrailIdentifier = guardrailIdentifier
        self.guardrailVersion = guardrailVersion

        # store kwargs as optional_params
        self.optional_params = kwargs

        super().__init__(**kwargs)
        BaseAWSLLM.__init__(self)

        verbose_proxy_logger.debug(
            "Bedrock Guardrail initialized with guardrailIdentifier: %s, guardrailVersion: %s",
            self.guardrailIdentifier,
            self.guardrailVersion,
        )

    def convert_to_bedrock_format(
        self,
        messages: Optional[List[AllMessageValues]] = None,
        response: Optional[Union[Any, ModelResponse]] = None,
    ) -> BedrockRequest:
        bedrock_request: BedrockRequest = BedrockRequest(source="INPUT")
        bedrock_request_content: List[BedrockContentItem] = []

        if messages:
            for message in messages:
                message_text_content: Optional[
                    List[str]
                ] = self.get_content_for_message(message=message)
                if message_text_content is None:
                    continue
                for text_content in message_text_content:
                    bedrock_content_item = BedrockContentItem(
                        text=BedrockTextContent(text=text_content)
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
        except ImportError:
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
        self.optional_params.pop(
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
        data: dict,
        optional_params: dict,
        aws_region_name: str,
        extra_headers: Optional[dict] = None,
    ):
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
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
    ) -> BedrockGuardrailResponse:
        credentials, aws_region_name = self._load_credentials()
        bedrock_request_data: dict = dict(
            self.convert_to_bedrock_format(
                messages=kwargs.get("messages"), response=response
            )
        )
        bedrock_guardrail_response: BedrockGuardrailResponse = (
            BedrockGuardrailResponse()
        )
        bedrock_request_data.update(
            self.get_guardrail_dynamic_request_body_params(request_data=kwargs)
        )
        prepared_request = self._prepare_request(
            credentials=credentials,
            data=bedrock_request_data,
            optional_params=self.optional_params,
            aws_region_name=aws_region_name,
        )
        verbose_proxy_logger.debug(
            "Bedrock AI request body: %s, url %s, headers: %s",
            bedrock_request_data,
            prepared_request.url,
            prepared_request.headers,
        )

        response = await self.async_handler.post(
            url=prepared_request.url,
            data=prepared_request.body,  # type: ignore
            headers=prepared_request.headers,  # type: ignore
        )
        verbose_proxy_logger.debug("Bedrock AI response: %s", response.text)
        if response.status_code == 200:
            # check if the response was flagged
            _json_response = response.json()
            bedrock_guardrail_response = BedrockGuardrailResponse(**_json_response)
            if self._should_raise_guardrail_blocked_exception(
                bedrock_guardrail_response
            ):
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

        return bedrock_guardrail_response

    def _should_raise_guardrail_blocked_exception(
        self, response: BedrockGuardrailResponse
    ) -> bool:
        """
        By default always raise an exception when a guardrail intervention is detected.

        If `self.mask_request_content` or `self.mask_response_content` is set to `True`, then use the output from the guardrail to mask the request or response content.
        """

        # if user opted into masking, return False. since we'll use the masked output from the guardrail
        if self.mask_request_content or self.mask_response_content:
            return False

        # if intervention, return True
        if response.get("action") == "GUARDRAIL_INTERVENED":
            return True

        # if no intervention, return False
        return False

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Union[Exception, str, dict, None]:
        verbose_proxy_logger.debug("Inside AIM Pre-Call Hook")

        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        new_messages: Optional[List[AllMessageValues]] = data.get("messages")
        if new_messages is None:
            verbose_proxy_logger.warning(
                "Bedrock AI: not running guardrail. No messages in data"
            )
            return data

        #########################################################
        ########## 1. Make the Bedrock API request ##########
        #########################################################
        bedrock_guardrail_response = await self.make_bedrock_api_request(kwargs=data)
        #########################################################

        #########################################################
        ########## 2. Update the messages with the guardrail response ##########
        #########################################################
        data[
            "messages"
        ] = self._update_messages_with_updated_bedrock_guardrail_response(
            messages=new_messages,
            bedrock_guardrail_response=bedrock_guardrail_response,
        )

        #########################################################
        ########## 3. Add the guardrail to the applied guardrails header ##########
        #########################################################
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return data

    @log_guardrail_information
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
            "responses",
        ],
    ):
        from litellm.proxy.common_utils.callback_utils import (
            add_guardrail_to_applied_guardrails_header,
        )

        event_type: GuardrailEventHooks = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return

        new_messages: Optional[List[AllMessageValues]] = data.get("messages")
        if new_messages is None:
            verbose_proxy_logger.warning(
                "Bedrock AI: not running guardrail. No messages in data"
            )
            return

        #########################################################
        ########## 1. Make the Bedrock API request ##########
        #########################################################
        bedrock_guardrail_response = await self.make_bedrock_api_request(kwargs=data)
        #########################################################

        #########################################################
        ########## 2. Update the messages with the guardrail response ##########
        #########################################################
        data[
            "messages"
        ] = self._update_messages_with_updated_bedrock_guardrail_response(
            messages=new_messages,
            bedrock_guardrail_response=bedrock_guardrail_response,
        )

        #########################################################
        ########## 3. Add the guardrail to the applied guardrails header ##########
        #########################################################
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return data

    @log_guardrail_information
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

        new_messages: Optional[List[AllMessageValues]] = data.get("messages")
        if new_messages is None:
            verbose_proxy_logger.warning(
                "Bedrock AI: not running guardrail. No messages in data"
            )
            return

        #########################################################
        ########## 1. Make the Bedrock API request ##########
        #########################################################
        bedrock_guardrail_response = await self.make_bedrock_api_request(
            kwargs=data, response=response
        )
        #########################################################

        #########################################################
        ########## 2. Update the messages with the guardrail response ##########
        #########################################################
        data[
            "messages"
        ] = self._update_messages_with_updated_bedrock_guardrail_response(
            messages=new_messages,
            bedrock_guardrail_response=bedrock_guardrail_response,
        )

        #########################################################
        ########## 3. Add the guardrail to the applied guardrails header ##########
        #########################################################
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

    ###########  HELPER FUNCTIONS for bedrock guardrails ############################
    ##############################################################################
    ##############################################################################
    def _update_messages_with_updated_bedrock_guardrail_response(
        self,
        messages: List[AllMessageValues],
        bedrock_guardrail_response: BedrockGuardrailResponse,
    ) -> List[AllMessageValues]:
        """
        Use the output from the bedrock guardrail to mask sensitive content in messages.

        Args:
            messages: Original list of messages
            bedrock_guardrail_response: Response from Bedrock guardrail containing masked content

        Returns:
            List of messages with content masked according to guardrail response
        """
        # Skip processing if masking is not enabled
        if not (self.mask_request_content or self.mask_response_content):
            return messages

        # Get masked texts from guardrail response
        masked_texts = self._extract_masked_texts_from_response(
            bedrock_guardrail_response
        )
        if not masked_texts:
            return messages

        # Apply masking to messages using index tracking
        return self._apply_masking_to_messages(
            messages=messages, masked_texts=masked_texts
        )

    def _extract_masked_texts_from_response(
        self, bedrock_guardrail_response: BedrockGuardrailResponse
    ) -> List[str]:
        """
        Extract all masked text outputs from the guardrail response.

        Args:
            bedrock_guardrail_response: Response from Bedrock guardrail

        Returns:
            List of masked text strings
        """
        masked_output_text: List[str] = []
        masked_outputs: Optional[List[BedrockGuardrailOutput]] = (
            bedrock_guardrail_response.get("outputs", []) or []
        )
        if not masked_outputs:
            verbose_proxy_logger.debug("No masked outputs found in guardrail response")
            return []

        for output in masked_outputs:
            text_content: Optional[str] = output.get("text")
            if text_content is not None:
                masked_output_text.append(text_content)

        return masked_output_text

    def _apply_masking_to_messages(
        self, messages: List[AllMessageValues], masked_texts: List[str]
    ) -> List[AllMessageValues]:
        """
        Apply masked texts to message content using index tracking.

        Args:
            messages: Original messages
            masked_texts: List of masked text strings from guardrail

        Returns:
            Updated messages with masked content
        """
        updated_messages = []
        masking_index = 0

        for message in messages:
            new_message = message.copy()
            content = new_message.get("content")

            # Skip messages with no content
            if content is None:
                updated_messages.append(new_message)
                continue

            # Handle string content
            if isinstance(content, str):
                if masking_index < len(masked_texts):
                    new_message["content"] = masked_texts[masking_index]
                    masking_index += 1
            # Handle list content
            elif isinstance(content, list):
                new_message["content"], masking_index = self._mask_content_list(
                    content_list=content,
                    masked_texts=masked_texts,
                    masking_index=masking_index,
                )

            updated_messages.append(new_message)

        return updated_messages

    def _mask_content_list(
        self, content_list: List[Any], masked_texts: List[str], masking_index: int
    ) -> Tuple[List[Any], int]:
        """
        Apply masking to a list of content items.

        Args:
            content_list: List of content items
            masked_texts: List of masked text strings
            starting_index: Starting index in the masked_texts list

        Returns:
            Updated content list with masked items
        """
        new_content: List[Union[dict, str]] = []
        for item in content_list:
            if isinstance(item, dict) and "text" in item:
                new_item = item.copy()
                if masking_index < len(masked_texts):
                    new_item["text"] = masked_texts[masking_index]
                    masking_index += 1
                new_content.append(new_item)
            elif isinstance(item, str):
                if masking_index < len(masked_texts):
                    item = masked_texts[masking_index]
                    masking_index += 1
                if item is not None:
                    new_content.append(item)

        return new_content, masking_index

    def get_content_for_message(self, message: AllMessageValues) -> Optional[List[str]]:
        """
        Get the content for a message.

        For bedrock guardrails we create a list of all the text content in the message.

        If a message has a list of content items, we flatten the list and return a list of text content.
        """
        message_text_content = []
        content = message.get("content")
        if content is None:
            return None
        if isinstance(content, str):
            message_text_content.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    message_text_content.append(item["text"])
                elif isinstance(item, str):
                    message_text_content.append(item)
        return message_text_content
