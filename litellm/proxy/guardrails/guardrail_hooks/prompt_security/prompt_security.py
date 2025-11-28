import asyncio
import base64
import os
import re
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, Type, Union

from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import (
    Choices,
    Delta,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
)

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

class PromptSecurityGuardrailMissingSecrets(Exception):
    pass

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

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Union[Exception, str, dict, None]:
        return await self.call_prompt_security_guardrail(data)

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: str,
    ) -> Union[Exception, str, dict, None]:
        await self.call_prompt_security_guardrail(data)
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

        messages = data.get("messages", [])
        
        # First, sanitize any files in the messages
        messages = await self.process_message_files(messages)

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
            return data
        action = result.get("action")
        violations = result.get("violations", [])
        if action == "block":
            raise HTTPException(status_code=400, detail="Blocked by Prompt Security, Violations: " + ", ".join(violations))
        elif action == "modify":
            data["messages"] = result.get("modified_messages", [])
        return data
    

    async def call_prompt_security_guardrail_on_output(self, output: str) -> dict:
        response = await self.async_handler.post(
            f"{self.api_base}/api/protect",
            headers = { 'APP-ID': self.api_key, 'Content-Type': 'application/json' },
            json = { "response": output, "user": self.user, "system_prompt": self.system_prompt }
        )
        response.raise_for_status()
        res = response.json()
        result = res.get("result", {}).get("response", {})
        if result is None: # prompt can exist but be with value None!
            return {}
        violations = result.get("violations", [])
        return { "action": result.get("action"), "modified_text": result.get("modified_text"), "violations": violations }

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[Any, ModelResponse, EmbeddingResponse, ImageResponse],
    ) -> Any:
        if (isinstance(response, ModelResponse) and response.choices and isinstance(response.choices[0], Choices)):
            content = response.choices[0].message.content or ""
            ret = await self.call_prompt_security_guardrail_on_output(content)
            violations = ret.get("violations", [])
            if ret.get("action") == "block":
                raise HTTPException(status_code=400, detail="Blocked by Prompt Security, Violations: " + ", ".join(violations))
            elif ret.get("action") == "modify":
                response.choices[0].message.content = ret.get("modified_text")
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

                    ret = await self.call_prompt_security_guardrail_on_output(chunk)
                    violations = ret.get("violations", [])
                    if ret.get("action") == "block":
                        from litellm.proxy.proxy_server import StreamingCallbackError
                        raise StreamingCallbackError("Blocked by Prompt Security, Violations: " + ", ".join(violations))
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