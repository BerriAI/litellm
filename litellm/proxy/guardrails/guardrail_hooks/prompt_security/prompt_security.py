import os
import re
import asyncio
import base64
from copy import deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Type, Union, cast
from fastapi import HTTPException
from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client, httpxSpecialProvider
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.responses.main import GenericResponseOutputItem, OutputText
from litellm.types.utils import (
    CallTypes,
    Choices,
    Delta,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
)

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel


# Exception classes
class PromptSecurityGuardrailMissingSecrets(Exception):
    """Exception raised when Prompt Security API credentials are missing."""

    pass


class PromptSecurityGuardrailAPIError(Exception):
    """Exception raised when there's an error calling the Prompt Security API."""

    pass


class PromptSecurityBlockedMessage(HTTPException):
    """Exception raised when content is blocked by Prompt Security."""

    def __init__(self, violations: List[str], message_type: str = "request"):
        super().__init__(
            status_code=400,
            detail={
                "error": "Blocked by Prompt Security Guardrail",
                "message_type": message_type,
                "violations": violations,
            },
        )

class PromptSecurityGuardrail(CustomGuardrail):
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, user: Optional[str] = None, system_prompt: Optional[str] = None, **kwargs):
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.api_key = api_key or os.environ.get("PROMPT_SECURITY_API_KEY")
        self.api_base = api_base or os.environ.get("PROMPT_SECURITY_API_BASE")
        self.user = user or os.environ.get("PROMPT_SECURITY_USER")
        self.system_prompt = system_prompt or os.environ.get("PROMPT_SECURITY_SYSTEM_PROMPT")
        if not self.api_key or not self.api_base:
            msg = (
                "Couldn't get Prompt Security api base or key, "
                "either set the `PROMPT_SECURITY_API_BASE` and `PROMPT_SECURITY_API_KEY` in the environment "
                "or pass them as parameters to the guardrail in the config file"
            )
            raise PromptSecurityGuardrailMissingSecrets(msg)
        
        # Configuration for file sanitization
        self.max_poll_attempts = 30  # Maximum number of polling attempts
        self.poll_interval = 2  # Seconds between polling attempts
        
        super().__init__(**kwargs)

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Union[Exception, str, dict, None]:
        # Ensure metadata exists for logging
        data.setdefault("metadata", {})
        
        event_type = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        transformed_data = data
        sanitized_messages: Optional[List[Any]] = None

        try:
            call_type_enum = CallTypes(call_type)
        except ValueError:
            call_type_enum = None

        if call_type_enum is not None and call_type_enum in {CallTypes.responses, CallTypes.aresponses}:
            extracted_messages = self.get_guardrails_messages_for_call_type(
                call_type=call_type_enum,
                data=data,
            )
            if extracted_messages is not None:
                sanitized_messages = extracted_messages
                transformed_data = deepcopy(data)
                transformed_data["messages"] = extracted_messages

        result = await self.call_prompt_security_guardrail(transformed_data)

        if (
            call_type_enum in {CallTypes.responses, CallTypes.aresponses}
            and isinstance(result, dict)
        ):
            result_messages = result.get("messages") or sanitized_messages
            result = self._update_responses_request_with_guardrail_messages(
                original_request=data,
                guardrail_result=result,
                sanitized_messages=result_messages,
            )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        return result

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: str,
    ) -> Union[Exception, str, dict, None]:
        # Ensure metadata exists for logging
        data.setdefault("metadata", {})
        
        event_type = GuardrailEventHooks.during_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        await self.call_prompt_security_guardrail(data)
        
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        
        return data

    async def sanitize_file_content(self, file_data: bytes, filename: str) -> dict:
        """
        Sanitize file content using Prompt Security API
        Returns: dict with keys 'action', 'content', 'metadata'
        """
        headers = {'APP-ID': self.api_key}
        
        # Step 1: Upload file for sanitization
        files = {'file': (filename, file_data)}
        upload_response = await self.async_handler.post(
            f"{self.api_base}/api/sanitizeFile",
            headers=headers,
            files=files,
        )
        upload_response.raise_for_status()
        upload_result = upload_response.json()
        job_id = upload_result.get("jobId")
        
        if not job_id:
            raise HTTPException(status_code=500, detail="Failed to get jobId from Prompt Security")
        
        verbose_proxy_logger.debug(f"File sanitization started with jobId: {job_id}")
        
        # Step 2: Poll for results
        for attempt in range(self.max_poll_attempts):
            await asyncio.sleep(self.poll_interval)
            
            poll_response = await self.async_handler.get(
                f"{self.api_base}/api/sanitizeFile",
                headers=headers,
                params={"jobId": job_id},
            )
            poll_response.raise_for_status()
            result = poll_response.json()
            
            status = result.get("status")
            
            if status == "done":
                verbose_proxy_logger.debug(f"File sanitization completed: {result}")
                return {
                    "action": result.get("metadata", {}).get("action", "allow"),
                    "content": result.get("content"),
                    "metadata": result.get("metadata", {}),
                    "violations": result.get("metadata", {}).get("violations", []),
                }
            elif status == "in progress":
                verbose_proxy_logger.debug(f"File sanitization in progress (attempt {attempt + 1}/{self.max_poll_attempts})")
                continue
            else:
                raise HTTPException(status_code=500, detail=f"Unexpected sanitization status: {status}")
        
        raise HTTPException(status_code=408, detail="File sanitization timeout")

    async def _process_image_url_item(self, item: dict) -> dict:
        """Process and sanitize image_url items."""
        image_url_data = item.get("image_url", {})
        url = image_url_data.get("url", "") if isinstance(image_url_data, dict) else image_url_data
        
        if not url.startswith("data:"):
            return item
        
        try:
            header, encoded = url.split(",", 1)
            file_data = base64.b64decode(encoded)
            mime_type = header.split(";")[0].split(":")[1]
            extension = mime_type.split("/")[-1]
            filename = f"image.{extension}"
            
            sanitization_result = await self.sanitize_file_content(file_data, filename)
            action = sanitization_result.get("action")
            
            if action == "block":
                violations = sanitization_result.get("violations", [])
                raise HTTPException(
                    status_code=400,
                    detail=f"File blocked by Prompt Security. Violations: {', '.join(violations)}"
                )
            
            if action == "modify":
                sanitized_content = sanitization_result.get("content", "")
                if sanitized_content:
                    sanitized_encoded = base64.b64encode(sanitized_content.encode()).decode()
                    sanitized_url = f"{header},{sanitized_encoded}"
                    if isinstance(image_url_data, dict):
                        image_url_data["url"] = sanitized_url
                    else:
                        item["image_url"] = sanitized_url
                    verbose_proxy_logger.info("File content modified by Prompt Security")
            
            return item
        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"Error sanitizing image file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"File sanitization failed: {str(e)}")

    async def _process_document_item(self, item: dict) -> dict:
        """Process and sanitize document/file items."""
        doc_data = item.get("document") or item.get("file") or item
        
        if isinstance(doc_data, dict):
            url = doc_data.get("url", "")
            doc_content = doc_data.get("data", "")
        else:
            url = doc_data if isinstance(doc_data, str) else ""
            doc_content = ""
        
        if not (url.startswith("data:") or doc_content):
            return item
        
        try:
            header = ""
            if url.startswith("data:"):
                header, encoded = url.split(",", 1)
                file_data = base64.b64decode(encoded)
                mime_type = header.split(";")[0].split(":")[1]
            else:
                file_data = base64.b64decode(doc_content)
                mime_type = doc_data.get("mime_type", "application/pdf") if isinstance(doc_data, dict) else "application/pdf"
            
            if "pdf" in mime_type:
                filename = "document.pdf"
            elif "word" in mime_type or "docx" in mime_type:
                filename = "document.docx"
            elif "excel" in mime_type or "xlsx" in mime_type:
                filename = "document.xlsx"
            else:
                extension = mime_type.split("/")[-1]
                filename = f"document.{extension}"
            
            verbose_proxy_logger.info(f"Sanitizing document: {filename}")
            
            sanitization_result = await self.sanitize_file_content(file_data, filename)
            action = sanitization_result.get("action")
            
            if action == "block":
                violations = sanitization_result.get("violations", [])
                raise HTTPException(
                    status_code=400,
                    detail=f"Document blocked by Prompt Security. Violations: {', '.join(violations)}"
                )
            
            if action == "modify":
                sanitized_content = sanitization_result.get("content", "")
                if sanitized_content:
                    sanitized_encoded = base64.b64encode(
                        sanitized_content if isinstance(sanitized_content, bytes) else sanitized_content.encode()
                    ).decode()
                    
                    if url.startswith("data:") and header:
                        sanitized_url = f"{header},{sanitized_encoded}"
                        if isinstance(doc_data, dict):
                            doc_data["url"] = sanitized_url
                    elif isinstance(doc_data, dict):
                        doc_data["data"] = sanitized_encoded
                    
                    verbose_proxy_logger.info("Document content modified by Prompt Security")
            
            return item
        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"Error sanitizing document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Document sanitization failed: {str(e)}")

    async def process_message_files(self, messages: list) -> list:
        """Process messages and sanitize any file content (images, documents, PDFs, etc.)."""
        processed_messages = []
        
        for message in messages:
            content = message.get("content")
            
            if not isinstance(content, list):
                processed_messages.append(message)
                continue
            
            processed_content = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "image_url":
                        item = await self._process_image_url_item(item)
                    elif item_type in ["document", "file"]:
                        item = await self._process_document_item(item)
                
                processed_content.append(item)
            
            processed_message = message.copy()
            processed_message["content"] = processed_content
            processed_messages.append(processed_message)
        
        return processed_messages

    async def call_prompt_security_guardrail(self, data: dict) -> dict:
        from datetime import datetime
        
        # Ensure metadata exists for logging
        data.setdefault("metadata", {})
        
        start_time = datetime.now()
        
        messages = data.get("messages", [])
        
        try:
            # First, sanitize any files in the messages
            messages = await self.process_message_files(messages)

            # TODO: Message filtering workaround - filters out system-generated metadata before sending to Prompt Security API
            # Removes:
            #   1. Messages starting with '### ' (markdown headers, likely system-generated metadata)
            #   2. Messages containing '"follow_ups": [' (JSON-structured follow-up suggestions)
            # Note: This is a brittle pattern-matching approach unlike other guardrails (Bedrock, GuardrailsAI)
            # which use role-based filtering or extract specific message types. Consider refactoring to:
            #   - Use message role/metadata instead of content patterns
            #   - Add logging to track what's being filtered
            #   - Investigate where these messages originate (LiteLLM internals?)
            def good_msg(msg):
                content = msg.get('content', '')
                # Handle both string and list content types
                if isinstance(content, str):
                    if content.startswith('### '): return False
                    if '"follow_ups": [' in content: return False
                return True

            messages = list(filter(lambda msg: good_msg(msg), messages))

            data["messages"] = messages

            # Then, run the regular prompt security check
            headers = { 'APP-ID': self.api_key, 'Content-Type': 'application/json' }
            response = await self.async_handler.post(
                f"{self.api_base}/api/protect",
                headers=headers,
                json={"messages": messages, "user": self.user, "system_prompt": self.system_prompt},
            )
            response.raise_for_status()
            res = response.json()
            result = res.get("result", {}).get("prompt", {})
            
            if result is None: # prompt can exist but be with value None!
                # Log successful scan with no action needed
                self.add_standard_logging_guardrail_information_to_request_data(
                    guardrail_provider="prompt_security",
                    guardrail_json_response=res,
                    request_data=data,
                    guardrail_status="success",
                    start_time=start_time.timestamp(),
                    end_time=datetime.now().timestamp(),
                    duration=(datetime.now() - start_time).total_seconds(),
                )
                return data
                
            action = result.get("action")
            violations = result.get("violations", [])
            
            # Determine guardrail status based on action
            if action == "block":
                guardrail_status = "guardrail_intervened"
            elif action == "modify":
                guardrail_status = "guardrail_intervened"
            else:
                guardrail_status = "success"
            
            # Log guardrail information
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="prompt_security",
                guardrail_json_response=res,
                request_data=data,
                guardrail_status=guardrail_status,
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
            )
            
            # Handle actions
            if action == "block":
                raise PromptSecurityBlockedMessage(violations=violations, message_type="request")
            elif action == "modify":
                data["messages"] = result.get("modified_messages", [])
            
            return data
            
        except PromptSecurityBlockedMessage:
            raise
        except HTTPException:
            raise
        except Exception as e:
            # Log failures
            verbose_proxy_logger.error(
                f"Prompt Security guardrail error: {str(e)}",
                exc_info=True,
            )
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider="prompt_security",
                guardrail_json_response=str(e),
                request_data=data,
                guardrail_status="guardrail_failed_to_respond",
                start_time=start_time.timestamp(),
                end_time=datetime.now().timestamp(),
                duration=(datetime.now() - start_time).total_seconds(),
            )
            raise PromptSecurityGuardrailAPIError(
                f"Failed to call Prompt Security API: {str(e)}"
            ) from e
    

    async def call_prompt_security_guardrail_on_output(self, output: str, request_data: Optional[dict] = None) -> dict:
        from datetime import datetime
        
        # Ensure metadata exists for logging if request_data is provided
        if request_data is not None:
            request_data.setdefault("metadata", {})
        
        start_time = datetime.now()
        
        try:
            response = await self.async_handler.post(
                f"{self.api_base}/api/protect",
                headers = { 'APP-ID': self.api_key, 'Content-Type': 'application/json' },
                json = { "response": output, "user": self.user, "system_prompt": self.system_prompt }
            )
            response.raise_for_status()
            res = response.json()
            result = res.get("result", {}).get("response", {})
            
            if result is None: # prompt can exist but be with value None!
                if request_data:
                    self.add_standard_logging_guardrail_information_to_request_data(
                        guardrail_provider="prompt_security",
                        guardrail_json_response=res,
                        request_data=request_data,
                        guardrail_status="success",
                        start_time=start_time.timestamp(),
                        end_time=datetime.now().timestamp(),
                        duration=(datetime.now() - start_time).total_seconds(),
                    )
                return {}
                
            violations = result.get("violations", [])
            action = result.get("action")
            
            # Determine guardrail status
            if action == "block":
                guardrail_status = "guardrail_intervened"
            elif action == "modify":
                guardrail_status = "guardrail_intervened"
            else:
                guardrail_status = "success"
            
            # Log guardrail information if request_data is available
            if request_data:
                self.add_standard_logging_guardrail_information_to_request_data(
                    guardrail_provider="prompt_security",
                    guardrail_json_response=res,
                    request_data=request_data,
                    guardrail_status=guardrail_status,
                    start_time=start_time.timestamp(),
                    end_time=datetime.now().timestamp(),
                    duration=(datetime.now() - start_time).total_seconds(),
                )
            
            return { "action": action, "modified_text": result.get("modified_text"), "violations": violations }
            
        except Exception as e:
            verbose_proxy_logger.error(
                f"Prompt Security output scan error: {str(e)}",
                exc_info=True,
            )
            if request_data:
                self.add_standard_logging_guardrail_information_to_request_data(
                    guardrail_provider="prompt_security",
                    guardrail_json_response=str(e),
                    request_data=request_data,
                    guardrail_status="guardrail_failed_to_respond",
                    start_time=start_time.timestamp(),
                    end_time=datetime.now().timestamp(),
                    duration=(datetime.now() - start_time).total_seconds(),
                )
            raise

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[Any, ModelResponse, EmbeddingResponse, ImageResponse],
    ) -> Any:
        # Ensure metadata exists for logging
        data.setdefault("metadata", {})
        
        event_type = GuardrailEventHooks.post_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return response

        if (isinstance(response, ModelResponse) and response.choices and isinstance(response.choices[0], Choices)):
            content = response.choices[0].message.content or ""
            ret = await self.call_prompt_security_guardrail_on_output(content, request_data=data)
            violations = ret.get("violations", [])
            if ret.get("action") == "block":
                raise PromptSecurityBlockedMessage(violations=violations, message_type="response")
            elif ret.get("action") == "modify":
                response.choices[0].message.content = ret.get("modified_text")
            
            add_guardrail_to_applied_guardrails_header(
                request_data=data, guardrail_name=self.guardrail_name
            )
            return response

        if isinstance(response, ResponsesAPIResponse) or (
            hasattr(response, "output") and hasattr(response, "model")
        ):
            result = await self._scan_responses_api_output(response, request_data=data)
            add_guardrail_to_applied_guardrails_header(
                request_data=data, guardrail_name=self.guardrail_name
            )
            return result
        
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        buffer: str = ""
        WINDOW_SIZE = 250  # Adjust window size as needed

        async for item in response:
            if not isinstance(item, ModelResponseStream) or not item.choices or len(item.choices) == 0:
                yield item
                continue

            choice = item.choices[0]
            if choice.delta and choice.delta.content:
                buffer += choice.delta.content

            if choice.finish_reason or len(buffer) >= WINDOW_SIZE:
                if buffer:
                    if not choice.finish_reason and re.search(r'\s', buffer):
                        chunk, buffer = re.split(r'(?=\s\S*$)', buffer, 1)
                    else:
                        chunk, buffer = buffer,''

                    ret = await self.call_prompt_security_guardrail_on_output(chunk, request_data=request_data)
                    violations = ret.get("violations", [])
                    if ret.get("action") == "block":
                        raise PromptSecurityBlockedMessage(violations=violations, message_type="streaming_response")
                    elif ret.get("action") == "modify":
                        chunk = ret.get("modified_text")
                    
                    if choice.delta:
                        choice.delta.content = chunk
                    else:
                        choice.delta = Delta(content=chunk)
                yield item

    
    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.prompt_security import (
            PromptSecurityGuardrailConfigModel,
        )
        return PromptSecurityGuardrailConfigModel

    def _normalize_message(self, message: Any) -> Dict[str, Any]:
        if hasattr(message, "model_dump"):
            return message.model_dump()
        if isinstance(message, dict):
            return message
        return dict(message)

    def _update_responses_request_with_guardrail_messages(
        self,
        original_request: dict,
        guardrail_result: dict,
        sanitized_messages: Optional[List[Any]],
    ) -> dict:
        if not sanitized_messages:
            return guardrail_result

        normalized_messages = [self._normalize_message(msg) for msg in sanitized_messages]

        instructions = guardrail_result.get("instructions", original_request.get("instructions"))
        non_system_messages: List[Dict[str, Any]] = []

        for msg in normalized_messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system" and isinstance(content, str):
                instructions = content
                continue
            non_system_messages.append(msg)

        original_input = original_request.get("input")

        if isinstance(original_input, str):
            if non_system_messages:
                first_content = non_system_messages[0].get("content")
                if isinstance(first_content, str):
                    guardrail_result["input"] = first_content
        elif isinstance(original_input, list) and len(non_system_messages) == len(original_input):
            updated_input = deepcopy(original_input)
            for idx, item in enumerate(updated_input):
                sanitized_message = non_system_messages[idx]
                if isinstance(item, dict):
                    if "content" in sanitized_message:
                        item["content"] = sanitized_message.get("content")
                    if sanitized_message.get("role") is not None:
                        item["role"] = sanitized_message.get("role")
                    if sanitized_message.get("type") is not None:
                        item["type"] = sanitized_message.get("type")
            guardrail_result["input"] = updated_input
        else:
            fallback_input: List[Dict[str, Any]] = []
            for msg in non_system_messages:
                fallback_msg = {
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                }
                if msg.get("type") is not None:
                    fallback_msg["type"] = msg.get("type")
                fallback_input.append(fallback_msg)
            if len(fallback_input) == 1 and isinstance(fallback_input[0].get("content"), str):
                guardrail_result["input"] = fallback_input[0]["content"]
            else:
                guardrail_result["input"] = fallback_input

        if instructions is not None:
            guardrail_result["instructions"] = instructions

        return guardrail_result

    async def _scan_responses_api_output(
        self,
        responses: Union[ResponsesAPIResponse, Any],
        request_data: Optional[dict] = None,
    ) -> Union[ResponsesAPIResponse, Any]:
        """
        Scan Responses API output for guardrail violations.
        
        Returns the modified response or the original response if no violations.
        """
        if not hasattr(responses, "output") or responses.output is None:
            return responses

        for output_item in responses.output:
            if isinstance(output_item, GenericResponseOutputItem):
                content_items = output_item.content or []
            elif isinstance(output_item, dict):
                content_items = output_item.get("content") or []
            else:
                continue

            for content in content_items:
                if isinstance(content, OutputText):
                    text_value = content.text
                elif isinstance(content, dict):
                    text_value = content.get("text")
                else:
                    text_value = None

                if not text_value:
                    continue

                guardrail_result = await self.call_prompt_security_guardrail_on_output(
                    cast(str, text_value),
                    request_data=request_data
                )
                violations = guardrail_result.get("violations", [])
                action = guardrail_result.get("action")

                if action == "block":
                    # Raise exception to ensure consistent blocking behavior with popup/alert
                    raise PromptSecurityBlockedMessage(violations=violations, message_type="responses_api_output")
                elif action == "modify":
                    modified_text = guardrail_result.get("modified_text")
                    if isinstance(content, OutputText):
                        content.text = modified_text
                    elif isinstance(content, dict):
                        content["text"] = modified_text

        return responses