import os
import re
import asyncio
import base64
from typing import TYPE_CHECKING, AsyncGenerator, List, Optional, Type, Union
from fastapi import HTTPException
from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail, log_guardrail_information
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client, httpxSpecialProvider
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import add_guardrail_to_applied_guardrails_header
from litellm.types.guardrails import GuardrailEventHooks, PiiEntityType
from litellm.types.utils import (
    CallTypes,
    Delta,
    GuardrailStatus,
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
    """
    Hybrid Prompt Security Guardrail for LiteLLM.

    Uses apply_guardrail for automatic text checking across all endpoints.
    Uses async_pre_call_hook for file sanitization (images, PDFs, documents).
    Uses async_post_call_streaming_iterator_hook for streaming responses.
    """

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

    async def apply_guardrail(
        self,
        text: str,
        language: Optional[str] = None,
        entities: Optional[List[PiiEntityType]] = None,
        request_data: Optional[dict] = None,
    ) -> str:
        """
        Apply Prompt Security guardrail to text content.

        This method is called automatically by LiteLLM for both input and output text
        across all LLM call types (chat completions, anthropic messages, responses API, etc.).

        LiteLLM automatically:
        - Extracts text from all message formats
        - Handles multimodal content (extracts text portions)
        - Maps responses back to original structure
        - Works for both input (pre_call/during_call) and output (post_call)

        Args:
            text: The text to check against Prompt Security rules (extracted by LiteLLM)
            language: Optional language parameter (unused)
            entities: Optional entities parameter (unused)
            request_data: Optional request data for logging

        Returns:
            The original or modified text if allowed

        Raises:
            PromptSecurityBlockedMessage: If content is blocked by Prompt Security
            PromptSecurityGuardrailAPIError: If API call fails
        """
        from datetime import datetime

        start_time = datetime.now()

        # Ensure metadata exists for logging if request_data is provided
        if request_data is not None:
            request_data.setdefault("metadata", {})

        try:
            # Call Prompt Security API
            headers = {
                'APP-ID': self.api_key,
                'Content-Type': 'application/json'
            }

            # Create a simple message structure for the API
            payload = {
                "messages": [{"role": "user", "content": text}],
                "user": self.user,
                "system_prompt": self.system_prompt
            }

            response = await self.async_handler.post(
                f"{self.api_base}/api/protect",
                headers=headers,
                json=payload,
                timeout=5,
            )
            response.raise_for_status()
            res = response.json()

            # Parse response - check both prompt and response fields
            result = res.get("result", {})
            prompt_result = result.get("prompt", {})
            response_result = result.get("response", {})

            # Use whichever result is present (prompt for input, response for output)
            guardrail_result = prompt_result if prompt_result else response_result

            if guardrail_result is None or not guardrail_result:
                # No violations found
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
                return text

            action = guardrail_result.get("action")
            violations = guardrail_result.get("violations", [])

            # Determine guardrail status
            guardrail_status: GuardrailStatus
            if action == "block":
                guardrail_status = "guardrail_intervened"
            elif action == "modify":
                guardrail_status = "guardrail_intervened"
            else:
                guardrail_status = "success"

            # Log guardrail information
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

            # Handle blocking
            if action == "block":
                raise PromptSecurityBlockedMessage(violations=violations, message_type="text")

            # Handle modification
            if action == "modify":
                # Extract modified text from the response
                modified_messages = guardrail_result.get("modified_messages", [])
                modified_text = guardrail_result.get("modified_text")

                # Extract modified text from modified_messages if present
                if not modified_text and modified_messages:
                    for msg in modified_messages:
                        if isinstance(msg, dict) and msg.get("content"):
                            modified_text = msg.get("content")
                            break

                if modified_text:
                    verbose_proxy_logger.debug(
                        f"Prompt Security modified text: {len(text)} -> {len(modified_text)} chars"
                    )
                    return modified_text

            return text

        except PromptSecurityBlockedMessage:
            raise
        except Exception as e:
            verbose_proxy_logger.error(
                f"Prompt Security API error: {str(e)}",
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
            raise PromptSecurityGuardrailAPIError(
                f"Failed to call Prompt Security API: {str(e)}"
            ) from e

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Union[Exception, str, dict, None]:
        """
        Pre-call hook for file sanitization and message filtering.

        Handles:
        1. Sanitizing files (images, documents, PDFs) in messages
        2. Filtering out system-generated metadata messages

        Text checking is handled automatically by apply_guardrail method
        which LiteLLM calls for all endpoints.
        """
        # Ensure metadata exists for logging
        data.setdefault("metadata", {})

        event_type = GuardrailEventHooks.pre_call
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        # Get messages from request
        messages = data.get("messages")
        if not messages:
            # Try to extract messages from responses API format
            try:
                call_type_enum = CallTypes(call_type)
                if call_type_enum in {CallTypes.responses, CallTypes.aresponses}:
                    messages = self.get_guardrails_messages_for_call_type(
                        call_type=call_type_enum,
                        data=data,
                    )
            except ValueError:
                pass

        if messages:
            # Step 1: Sanitize any files in the messages
            messages = await self.process_message_files(messages)

            # Step 2: Filter out system-generated metadata messages
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
                    if content.startswith('### '):
                        return False
                    if '"follow_ups": [' in content:
                        return False
                return True

            messages = list(filter(lambda msg: good_msg(msg), messages))

            # Update messages in data
            data["messages"] = messages

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )

        # Return data - LiteLLM will then call apply_guardrail for text checking
        return data

    async def sanitize_file_content(self, file_data: bytes, filename: str) -> dict:
        """
        Sanitize file content using Prompt Security API.

        Returns: dict with keys 'action', 'content', 'metadata', 'violations'
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

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict,
        response,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Handle streaming responses with buffering and word-boundary splitting.

        This hook is still needed because apply_guardrail doesn't handle streaming.
        """
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
                        chunk, buffer = buffer, ''

                    # Use apply_guardrail for the chunk
                    try:
                        processed_chunk = await self.apply_guardrail(chunk, request_data=request_data)
                        chunk = processed_chunk
                    except Exception as e:
                        # If guardrail fails, re-raise (will block stream)
                        raise e

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
