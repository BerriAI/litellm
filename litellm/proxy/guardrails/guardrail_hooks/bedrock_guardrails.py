# +-------------------------------------------------------------+
#
#           Use Bedrock Guardrails for your LLM calls
#
# +-------------------------------------------------------------+
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import copy
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import json
import sys
from typing import Any, AsyncGenerator, List, Literal, Optional, Tuple, Union

import httpx
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
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import AllMessageValues
from litellm.types.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
    BedrockContentItem,
    BedrockGuardrailOutput,
    BedrockGuardrailResponse,
    BedrockRequest,
    BedrockTextContent,
)
from litellm.types.utils import (
    Choices,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    TextChoices,
)

GUARDRAIL_NAME = "bedrock"


def _redact_pii_matches(response_json: dict) -> dict:
    try:
        # Create a deep copy to avoid modifying the original response
        redacted_response = copy.deepcopy(response_json)

        # Get assessments from the response
        assessments = redacted_response.get("assessments", [])
        if not assessments:
            return redacted_response

        for assessment in assessments:
            # Redact PII entities in sensitive information policy
            sensitive_info_policy = assessment.get("sensitiveInformationPolicy")
            if sensitive_info_policy:
                pii_entities = sensitive_info_policy.get("piiEntities", [])
                for pii_entity in pii_entities:
                    if "match" in pii_entity:
                        pii_entity["match"] = "[REDACTED]"

                # Redact regex matches
                regexes = sensitive_info_policy.get("regexes", [])
                for regex_match in regexes:
                    if "match" in regex_match:
                        regex_match["match"] = "[REDACTED]"

            # Redact custom word matches in word policy
            word_policy = assessment.get("wordPolicy")
            if word_policy:
                custom_words = word_policy.get("customWords", [])
                for custom_word in custom_words:
                    if "match" in custom_word:
                        custom_word["match"] = "[REDACTED]"

                managed_words = word_policy.get("managedWordLists", [])
                for managed_word in managed_words:
                    if "match" in managed_word:
                        managed_word["match"] = "[REDACTED]"

        return redacted_response
    except Exception as e:
        # We do not want to fail in any case so this is just a warning
        verbose_proxy_logger.warning("Guardrail log redaction failed: %s", str(e))
        return response_json


class BedrockGuardrail(CustomGuardrail, BaseAWSLLM):
    def __init__(
        self,
        guardrailIdentifier: Optional[str] = None,
        guardrailVersion: Optional[str] = None,
        disable_exception_on_block: Optional[bool] = False,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.guardrailIdentifier = guardrailIdentifier
        self.guardrailVersion = guardrailVersion

        # store kwargs as optional_params
        self.optional_params = kwargs

        self.disable_exception_on_block: bool = disable_exception_on_block or False
        """
        If True, will not raise an exception when the guardrail is blocked.
        """
        

        # Set supported event hooks to include MCP hooks
        if 'supported_event_hooks' not in kwargs:
            kwargs['supported_event_hooks'] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
                GuardrailEventHooks.pre_mcp_call,
                GuardrailEventHooks.during_mcp_call,
            ]
        
        super().__init__(**kwargs)
        BaseAWSLLM.__init__(self)

        verbose_proxy_logger.debug(
            "Bedrock Guardrail initialized with guardrailIdentifier: %s, guardrailVersion: %s",
            self.guardrailIdentifier,
            self.guardrailVersion,
        )
    

    def _create_bedrock_input_content_request(self, messages: Optional[List[AllMessageValues]]) -> BedrockRequest:
        """
        Create a bedrock request for the input content - the LLM request.
        """
        bedrock_request: BedrockRequest = BedrockRequest(source="INPUT")
        bedrock_request_content: List[BedrockContentItem] = []
        if messages is None:
            return bedrock_request
        for message in messages:
            message_text_content: Optional[List[str]] = (
                self.get_content_for_message(message=message)
            )
            if message_text_content is None:
                continue
            for text_content in message_text_content:
                bedrock_content_item = BedrockContentItem(
                    text=BedrockTextContent(text=text_content)
                )
                bedrock_request_content.append(bedrock_content_item)

        bedrock_request["content"] = bedrock_request_content
        return bedrock_request

    def _create_bedrock_output_content_request(self, response: Union[Any, ModelResponse]) -> BedrockRequest:
        """
        Create a bedrock request for the output content - the LLM response.
        """
        bedrock_request: BedrockRequest = BedrockRequest(source="OUTPUT")
        bedrock_request_content: List[BedrockContentItem] = []
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

    def convert_to_bedrock_format(
        self,
        source: Literal["INPUT", "OUTPUT"],
        messages: Optional[List[AllMessageValues]] = None,
        response: Optional[Union[Any, ModelResponse]] = None,
    ) -> BedrockRequest:
        """
        Convert the litellm messages/response to the bedrock request format.

        If source is "INPUT", then messages is required.
        If source is "OUTPUT", then response is required.

        Returns:
            BedrockRequest: The bedrock request object.
        """
        bedrock_request: BedrockRequest = BedrockRequest(source=source)
        if source == "INPUT":
            bedrock_request = self._create_bedrock_input_content_request(messages=messages)
        elif source == "OUTPUT":
            bedrock_request = self._create_bedrock_output_content_request(response=response)
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
        aws_secret_access_key = self.optional_params.get("aws_secret_access_key", None)
        aws_access_key_id = self.optional_params.get("aws_access_key_id", None)
        aws_session_token = self.optional_params.get("aws_session_token", None)
        aws_region_name = self.optional_params.get("aws_region_name", None)
        aws_role_name = self.optional_params.get("aws_role_name", None)
        aws_session_name = self.optional_params.get("aws_session_name", None)
        aws_profile_name = self.optional_params.get("aws_profile_name", None)
        aws_web_identity_token = self.optional_params.get(
            "aws_web_identity_token", None
        )
        aws_sts_endpoint = self.optional_params.get("aws_sts_endpoint", None)

        ### SET REGION NAME ###
        aws_region_name = self.get_aws_region_name_for_non_llm_api_calls(
            aws_region_name=aws_region_name,
        )

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
        self, 
        source: Literal["INPUT", "OUTPUT"],
        messages: Optional[List[AllMessageValues]] = None,
        response: Optional[Union[Any, litellm.ModelResponse]] = None,
        request_data: Optional[dict] = None
    ) -> BedrockGuardrailResponse:
        from datetime import datetime
        start_time = datetime.now()
        credentials, aws_region_name = self._load_credentials()
        bedrock_request_data: dict = dict(
            self.convert_to_bedrock_format(
                source=source, 
                messages=messages, 
                response=response
            )
        )
        bedrock_guardrail_response: BedrockGuardrailResponse = (
            BedrockGuardrailResponse()
        )
        if request_data:
            bedrock_request_data.update(
                self.get_guardrail_dynamic_request_body_params(request_data=request_data)
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
        #########################################################
        # Add guardrail information to request trace
        #########################################################
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=response.json(),
            request_data=request_data or {},
            guardrail_status=self._get_bedrock_guardrail_response_status(response=response),
            start_time=start_time.timestamp(),
            end_time=datetime.now().timestamp(),
            duration=(datetime.now() - start_time).total_seconds(),
        )
        #########################################################
        if response.status_code == 200:
            # check if the response was flagged
            _json_response = response.json()
            redacted_response = _redact_pii_matches(_json_response)
            verbose_proxy_logger.debug("Bedrock AI response : %s", redacted_response)
            bedrock_guardrail_response = BedrockGuardrailResponse(**_json_response)
            if self._should_raise_guardrail_blocked_exception(
                bedrock_guardrail_response
            ):
                raise self._get_http_exception_for_blocked_guardrail(
                    bedrock_guardrail_response
                )
        else:
            verbose_proxy_logger.error(
                "Bedrock AI: error in response. Status code: %s, response: %s",
                response.status_code,
                response.text,
            )

        return bedrock_guardrail_response
    
    def _get_bedrock_guardrail_response_status(self, response: httpx.Response) -> Literal["success", "failure"]:
        """
        Get the status of the bedrock guardrail response.
        """
        if response.status_code == 200:
            return "success"
        return "failure"

    def _get_http_exception_for_blocked_guardrail(self, response: BedrockGuardrailResponse) -> HTTPException:
        """
        Get the HTTP exception for a blocked guardrail.
        """
        bedrock_guardrail_output_text: str = ""
        outputs: Optional[List[BedrockGuardrailOutput]] = (
            response.get("outputs", []) or []
        )
        if outputs:
            for output in outputs:
                if output.get("text"):
                    bedrock_guardrail_output_text += output.get("text") or ""
                
        
        return HTTPException(
            status_code=400,
            detail={
                "error": "Violated guardrail policy", 
                "bedrock_guardrail_response": bedrock_guardrail_output_text,
            }
        )


    def _should_raise_guardrail_blocked_exception(
        self, response: BedrockGuardrailResponse
    ) -> bool:
        """
        Only raise exception for "BLOCKED" actions, not for "ANONYMIZED" actions.

        If `self.mask_request_content` or `self.mask_response_content` is set to `True`, then use the output from the guardrail to mask the request or response content.
        """

        # if user opted into masking, return False. since we'll use the masked output from the guardrail
        if self.mask_request_content or self.mask_response_content:
            return False
        
        if self.disable_exception_on_block is True:
            return False

        # if no intervention, return False
        if response.get("action") != "GUARDRAIL_INTERVENED":
            return False

        # Check assessments to determine if any actions were BLOCKED (vs ANONYMIZED)
        assessments = response.get("assessments", [])
        if not assessments:
            return False

        for assessment in assessments:
            # Check topic policy
            topic_policy = assessment.get("topicPolicy")
            if topic_policy:
                topics = topic_policy.get("topics", [])
                for topic in topics:
                    if topic.get("action") == "BLOCKED":
                        return True

            # Check content policy
            content_policy = assessment.get("contentPolicy")
            if content_policy:
                filters = content_policy.get("filters", [])
                for filter_item in filters:
                    if filter_item.get("action") == "BLOCKED":
                        return True

            # Check word policy
            word_policy = assessment.get("wordPolicy")
            if word_policy:
                custom_words = word_policy.get("customWords", [])
                for custom_word in custom_words:
                    if custom_word.get("action") == "BLOCKED":
                        return True
                managed_words = word_policy.get("managedWordLists", [])
                for managed_word in managed_words:
                    if managed_word.get("action") == "BLOCKED":
                        return True

            # Check sensitive information policy
            sensitive_info_policy = assessment.get("sensitiveInformationPolicy")
            if sensitive_info_policy:
                pii_entities = sensitive_info_policy.get("piiEntities", [])
                if pii_entities:
                    for pii_entity in pii_entities:
                        if pii_entity.get("action") == "BLOCKED":
                            return True
                regexes = sensitive_info_policy.get("regexes", [])
                if regexes:
                    for regex in regexes:
                        if regex.get("action") == "BLOCKED":
                            return True

            # Check contextual grounding policy
            contextual_grounding_policy = assessment.get("contextualGroundingPolicy")
            if contextual_grounding_policy:
                grounding_filters = contextual_grounding_policy.get("filters", [])
                for grounding_filter in grounding_filters:
                    if grounding_filter.get("action") == "BLOCKED":
                        return True

        # If we got here, intervention occurred but no BLOCKED actions found
        # This means all actions were ANONYMIZED or NONE, so don't raise exception
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
            "mcp_call",
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
        bedrock_guardrail_response = await self.make_bedrock_api_request(
            source="INPUT", messages=new_messages, request_data=data
        )
        #########################################################

        #########################################################
        ########## 2. Update the messages with the guardrail response ##########
        #########################################################
        data["messages"] = (
            self._update_messages_with_updated_bedrock_guardrail_response(
                messages=new_messages,
                bedrock_guardrail_response=bedrock_guardrail_response,
            )
        )

        #########################################################
        ########## 3. Add the guardrail to the applied guardrails header ##########
        #########################################################
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

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
            "mcp_call",
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
        bedrock_guardrail_response = await self.make_bedrock_api_request(
            source="INPUT", messages=new_messages, request_data=data
        )
        #########################################################

        #########################################################
        ########## 2. Update the messages with the guardrail response ##########
        #########################################################
        data["messages"] = (
            self._update_messages_with_updated_bedrock_guardrail_response(
                messages=new_messages,
                bedrock_guardrail_response=bedrock_guardrail_response,
            )
        )

        #########################################################
        ########## 3. Add the guardrail to the applied guardrails header ##########
        #########################################################
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return data

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
        ########## 1. Make parallel Bedrock API requests ##########
        #########################################################
        output_content_bedrock = await self.make_bedrock_api_request(
            source="OUTPUT", 
            response=response,
            request_data=data
        )  # Only response

        #########################################################
        ########## 2. Apply masking to response with output guardrail response ##########
        #########################################################
        self._apply_masking_to_response(
            response=response,
            bedrock_guardrail_response=output_content_bedrock,
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
        # Get masked texts from guardrail response
        masked_texts = self._extract_masked_texts_from_response(
            bedrock_guardrail_response
        )

        # If guardrail provided masked output, use it regardless of masking flags
        # because the guardrail has already determined this content needs anonymization
        if masked_texts:
            verbose_proxy_logger.debug(
                "Bedrock guardrail provided masked output, applying to messages"
            )
            return self._apply_masking_to_messages(
                messages=messages, masked_texts=masked_texts
            )

        # If masking is enabled but no masked texts available, still try to apply
        # (this maintains backward compatibility for edge cases)
        if self.mask_request_content or self.mask_response_content:
            verbose_proxy_logger.debug(
                "Masking enabled but no masked output from guardrail, returning original messages"
            )

        return messages

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Process streaming response chunks.

        Collect content from the stream and make parallel bedrock api requests to get the guardrail responses.
        """
        # Import here to avoid circular imports
        import asyncio

        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder
        from litellm.types.utils import TextCompletionResponse

        # Collect all chunks to process them together
        all_chunks: List[ModelResponseStream] = []
        async for chunk in response:
            all_chunks.append(chunk)

        assembled_model_response: Optional[
            Union[ModelResponse, TextCompletionResponse]
        ] = stream_chunk_builder(
            chunks=all_chunks,
        )
        if isinstance(assembled_model_response, ModelResponse):
            ####################################################################
            ########## 1. Make parallel Bedrock Apply Guardrail API requests ##########

            # Bedrock will raise an exception if this violates the guardrail policy
            ###################################################################
            # Create tasks for parallel execution
            input_task = self.make_bedrock_api_request(
                source="INPUT", messages=request_data.get("messages"), request_data=request_data
            )  # Only input messages
            output_task = self.make_bedrock_api_request(
                source="OUTPUT", response=assembled_model_response
            )  # Only response

            # Execute both requests in parallel
            _, output_guardrail_response = await asyncio.gather(
                input_task, output_task
            )

            #########################################################################
            ########## 2. Apply masking to response with output guardrail response ##########
            #########################################################################
            self._apply_masking_to_response(
                response=assembled_model_response,
                bedrock_guardrail_response=output_guardrail_response,
            )

            #########################################################################
            ########## 3. Return the (potentially masked) chunks ##########
            #########################################################################
            mock_response = MockResponseIterator(
                model_response=assembled_model_response
            )

            # Return the reconstructed stream
            async for chunk in mock_response:
                yield chunk
        else:
            for chunk in all_chunks:
                yield chunk

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

    def _apply_masking_to_response(
        self,
        response: Union[ModelResponse, Any],
        bedrock_guardrail_response: BedrockGuardrailResponse,
    ) -> None:
        """
        Apply masked content from bedrock guardrail to the response object.
        
        Args:
            response: The response object to modify
            bedrock_guardrail_response: Response from Bedrock guardrail containing masked content
        """
        # Get masked texts from guardrail response
        masked_texts = self._extract_masked_texts_from_response(
            bedrock_guardrail_response
        )

        if not masked_texts:
            verbose_proxy_logger.debug("No masked outputs found, skipping response masking")
            return

        verbose_proxy_logger.debug(
            "Applying masking to response with %d masked texts", len(masked_texts)
        )

        # Apply masking to ModelResponse
        if isinstance(response, litellm.ModelResponse):
            self._apply_masking_to_model_response(response, masked_texts)
        else:
            verbose_proxy_logger.warning(
                "Unsupported response type for masking: %s", type(response)
            )

    def _apply_masking_to_model_response(
        self, response: litellm.ModelResponse, masked_texts: List[str]
    ) -> None:
        """
        Apply masked texts to a ModelResponse object.
        
        Args:
            response: The ModelResponse object to modify in-place
            masked_texts: List of masked text strings from guardrail
        """
        masking_index = 0
        
        for choice in response.choices:
            if isinstance(choice, Choices):
                # For chat completions
                if choice.message.content and isinstance(choice.message.content, str):
                    if masking_index < len(masked_texts):
                        choice.message.content = masked_texts[masking_index]
                        masking_index += 1
                        verbose_proxy_logger.debug(
                            "Applied masking to choice message content"
                        )
            elif isinstance(choice, StreamingChoices):
                # For streaming responses, modify delta content
                if choice.delta.content and isinstance(choice.delta.content, str):
                    if masking_index < len(masked_texts):
                        choice.delta.content = masked_texts[masking_index]
                        masking_index += 1
                        verbose_proxy_logger.debug(
                            "Applied masking to choice delta content"
                        )
            elif isinstance(choice, TextChoices):
                # For text completions
                if choice.text and isinstance(choice.text, str):
                    if masking_index < len(masked_texts):
                        choice.text = masked_texts[masking_index]
                        masking_index += 1
                        verbose_proxy_logger.debug(
                            "Applied masking to choice text content"
                        )
