import os
import re
import asyncio
import base64
from typing import TYPE_CHECKING, Any, AsyncGenerator, Literal, Optional, Type, Union
from fastapi import HTTPException
from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client, httpxSpecialProvider
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import (
    Choices,
    Delta,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream
)

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

class PromptSecurityGuardrailMissingSecrets(Exception):
    pass

class PromptSecurityGuardrail(CustomGuardrail):
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, **kwargs):
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.api_key = api_key or os.environ.get("PROMPT_SECURITY_API_KEY")
        self.api_base = api_base or os.environ.get("PROMPT_SECURITY_API_BASE")
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
            "anthropic_messages",
        ],
    ) -> Union[Exception, str, dict, None]:
        return await self.call_prompt_security_guardrail(data)

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
            "anthropic_messages",
        ],
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

    async def process_message_files(self, messages: list) -> list:
        """
        Process messages and sanitize any file content (images, documents, etc.)
        """
        processed_messages = []
        
        for message in messages:
            content = message.get("content")
            
            # Handle messages with content list (multimodal messages)
            if isinstance(content, list):
                processed_content = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type")
                        
                        # Handle image_url type
                        if item_type == "image_url":
                            image_url_data = item.get("image_url", {})
                            if isinstance(image_url_data, dict):
                                url = image_url_data.get("url", "")
                            else:
                                url = image_url_data
                            
                            # Check if it's a base64 encoded image
                            if url.startswith("data:"):
                                try:
                                    # Extract base64 data
                                    header, encoded = url.split(",", 1)
                                    file_data = base64.b64decode(encoded)
                                    
                                    # Determine filename from mime type
                                    mime_type = header.split(";")[0].split(":")[1]
                                    extension = mime_type.split("/")[-1]
                                    filename = f"image.{extension}"
                                    
                                    # Sanitize the file
                                    sanitization_result = await self.sanitize_file_content(file_data, filename)
                                    
                                    action = sanitization_result.get("action")
                                    
                                    if action == "block":
                                        violations = sanitization_result.get("violations", [])
                                        raise HTTPException(
                                            status_code=400,
                                            detail=f"File blocked by Prompt Security. Violations: {', '.join(violations)}"
                                        )
                                    elif action == "modify":
                                        # Replace with sanitized content
                                        sanitized_content = sanitization_result.get("content", "")
                                        if sanitized_content:
                                            # Convert back to base64
                                            sanitized_encoded = base64.b64encode(sanitized_content.encode()).decode()
                                            sanitized_url = f"{header},{sanitized_encoded}"
                                            if isinstance(image_url_data, dict):
                                                image_url_data["url"] = sanitized_url
                                            else:
                                                item["image_url"] = sanitized_url
                                            verbose_proxy_logger.info("File content modified by Prompt Security")
                                    
                                except Exception as e:
                                    verbose_proxy_logger.error(f"Error sanitizing file: {str(e)}")
                                    raise HTTPException(status_code=500, detail=f"File sanitization failed: {str(e)}")
                        
                        processed_content.append(item)
                    else:
                        processed_content.append(item)
                
                processed_message = message.copy()
                processed_message["content"] = processed_content
                processed_messages.append(processed_message)
            else:
                processed_messages.append(message)
        
        return processed_messages

    async def call_prompt_security_guardrail(self, data: dict) -> dict:
        messages = data.get("messages", [])
        
        # First, sanitize any files in the messages
        messages = await self.process_message_files(messages)
        data["messages"] = messages
        
        # Then, run the regular prompt security check
        headers = { 'APP-ID': self.api_key, 'Content-Type': 'application/json' }
        response = await self.async_handler.post(
            f"{self.api_base}/api/protect",
            headers=headers,
            json={"messages": messages},
        )
        response.raise_for_status()
        res = response.json()
        result = res.get("result", {}).get("prompt", {})
        action = result.get("action")
        if action == "block":
            raise HTTPException(status_code=400, detail="Blocked by Prompt Security")
        elif action == "modify":
            data["messages"] = result.get("modified_messages", [])
        return data
    

    async def call_prompt_security_guardrail_on_output(self, output: str) -> dict:
        response = await self.async_handler.post(
            f"{self.api_base}/api/protect",
            headers = { 'APP-ID': self.api_key, 'Content-Type': 'application/json' },
            json = { "response": output }
        )
        response.raise_for_status()
        res = response.json()
        result = res.get("result", {}).get("response", {})
        return { "action": result.get("action"), "modified_text": result.get("modified_text") }

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[Any, ModelResponse, EmbeddingResponse, ImageResponse],
    ) -> Any:
        if (isinstance(response, ModelResponse) and response.choices and isinstance(response.choices[0], Choices)):
            content = response.choices[0].message.content or ""
            ret = await self.call_prompt_security_guardrail_on_output(content)
            if ret.get("action") == "block":
                raise HTTPException(status_code=400, detail="Blocked by Prompt Security")
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
                    if ret.get("action") == "block":
                        from litellm.proxy.proxy_server import StreamingCallbackError
                        raise StreamingCallbackError("Blocked by Prompt Security")
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