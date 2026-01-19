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


class PromptSecurityGuardrailMissingSecrets(Exception):
    pass


class PromptSecurityGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        user: Optional[str] = None,
        system_prompt: Optional[str] = None,
        check_tool_results: Optional[bool] = None,
        **kwargs,
    ):
        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self.api_key = api_key or os.environ.get("PROMPT_SECURITY_API_KEY")
        self.api_base = api_base or os.environ.get("PROMPT_SECURITY_API_BASE")
        self.user = user or os.environ.get("PROMPT_SECURITY_USER")
        self.system_prompt = system_prompt or os.environ.get(
            "PROMPT_SECURITY_SYSTEM_PROMPT"
        )

        # Configure whether to check tool/function results for indirect prompt injection
        # Default: False (Filter out tool/function messages)
        # True: Transform to "other" role and send to API
        if check_tool_results is None:
            check_tool_results_env = os.environ.get(
                "PROMPT_SECURITY_CHECK_TOOL_RESULTS", "false"
            ).lower()
            self.check_tool_results = check_tool_results_env in ("true", "1", "yes")
        else:
            self.check_tool_results = check_tool_results

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
        alias = self._resolve_key_alias(user_api_key_dict, data)
        return await self.call_prompt_security_guardrail(
            data, call_type=call_type, user_api_key_alias=alias
        )

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: str,
    ) -> Union[Exception, str, dict, None]:
        alias = self._resolve_key_alias(user_api_key_dict, data)
        await self.call_prompt_security_guardrail(
            data, call_type=call_type, user_api_key_alias=alias
        )
        return data

    async def sanitize_file_content(
        self,
        file_data: bytes,
        filename: str,
        user_api_key_alias: Optional[str] = None,
    ) -> dict:
        """
        Sanitize file content using Prompt Security API
        Returns: dict with keys 'action', 'content', 'metadata'
        """
        # For file upload, don't set Content-Type header - let httpx set multipart/form-data
        headers = {"APP-ID": self.api_key}
        if user_api_key_alias:
            headers["X-LiteLLM-Key-Alias"] = user_api_key_alias

        self._log_api_request(
            method="POST",
            url=f"{self.api_base}/api/sanitizeFile",
            headers=headers,
            payload=f"file upload: {filename}",
        )

        # Step 1: Upload file for sanitization
        files = {"file": (filename, file_data)}
        upload_response = await self.async_handler.post(
            f"{self.api_base}/api/sanitizeFile",
            headers=headers,
            files=files,
        )
        upload_response.raise_for_status()
        upload_result = upload_response.json()
        job_id = upload_result.get("jobId")
        self._log_api_response(
            url=f"{self.api_base}/api/sanitizeFile",
            status_code=upload_response.status_code,
            payload={"jobId": job_id},
        )

        if not job_id:
            raise HTTPException(
                status_code=500, detail="Failed to get jobId from Prompt Security"
            )

        verbose_proxy_logger.debug(
            "Prompt Security Guardrail: File sanitization started with jobId=%s", job_id
        )

        # Step 2: Poll for results
        for attempt in range(self.max_poll_attempts):
            await asyncio.sleep(self.poll_interval)

            self._log_api_request(
                method="GET",
                url=f"{self.api_base}/api/sanitizeFile",
                headers=headers,
                payload={"jobId": job_id},
            )
            poll_response = await self.async_handler.get(
                f"{self.api_base}/api/sanitizeFile",
                headers=headers,
                params={"jobId": job_id},
            )
            poll_response.raise_for_status()
            result = poll_response.json()
            self._log_api_response(
                url=f"{self.api_base}/api/sanitizeFile",
                status_code=poll_response.status_code,
                payload={"jobId": job_id, "status": result.get("status")},
            )

            status = result.get("status")

            if status == "done":
                verbose_proxy_logger.debug(
                    "Prompt Security Guardrail: File sanitization completed for jobId=%s",
                    job_id,
                )
                return {
                    "action": result.get("metadata", {}).get("action", "allow"),
                    "content": result.get("content"),
                    "metadata": result.get("metadata", {}),
                    "violations": result.get("metadata", {}).get("violations", []),
                }
            elif status == "in progress":
                verbose_proxy_logger.debug(
                    "Prompt Security Guardrail: File sanitization in progress (attempt %d/%d)",
                    attempt + 1,
                    self.max_poll_attempts,
                )
                continue
            else:
                raise HTTPException(
                    status_code=500, detail=f"Unexpected sanitization status: {status}"
                )

        raise HTTPException(status_code=408, detail="File sanitization timeout")

    async def _process_image_url_item(
        self, item: dict, user_api_key_alias: Optional[str]
    ) -> dict:
        """Process and sanitize image_url items."""
        image_url_data = item.get("image_url", {})
        url = (
            image_url_data.get("url", "")
            if isinstance(image_url_data, dict)
            else image_url_data
        )

        if not url.startswith("data:"):
            return item

        try:
            header, encoded = url.split(",", 1)
            file_data = base64.b64decode(encoded)
            mime_type = header.split(";")[0].split(":")[1]
            extension = mime_type.split("/")[-1]
            filename = f"image.{extension}"

            sanitization_result = await self.sanitize_file_content(
                file_data, filename, user_api_key_alias=user_api_key_alias
            )
            action = sanitization_result.get("action")

            if action == "block":
                violations = sanitization_result.get("violations", [])
                raise HTTPException(
                    status_code=400,
                    detail=f"File blocked by Prompt Security. Violations: {', '.join(violations)}",
                )

            if action == "modify":
                sanitized_content = sanitization_result.get("content", "")
                if sanitized_content:
                    sanitized_encoded = base64.b64encode(
                        sanitized_content.encode()
                    ).decode()
                    sanitized_url = f"{header},{sanitized_encoded}"
                    if isinstance(image_url_data, dict):
                        image_url_data["url"] = sanitized_url
                    else:
                        item["image_url"] = sanitized_url
                    verbose_proxy_logger.info(
                        "File content modified by Prompt Security"
                    )

            return item
        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"Error sanitizing image file: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"File sanitization failed: {str(e)}"
            )

    async def _process_document_item(
        self, item: dict, user_api_key_alias: Optional[str]
    ) -> dict:
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
                mime_type = (
                    doc_data.get("mime_type", "application/pdf")
                    if isinstance(doc_data, dict)
                    else "application/pdf"
                )

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

            sanitization_result = await self.sanitize_file_content(
                file_data, filename, user_api_key_alias=user_api_key_alias
            )
            action = sanitization_result.get("action")

            if action == "block":
                violations = sanitization_result.get("violations", [])
                raise HTTPException(
                    status_code=400,
                    detail=f"Document blocked by Prompt Security. Violations: {', '.join(violations)}",
                )

            if action == "modify":
                sanitized_content = sanitization_result.get("content", "")
                if sanitized_content:
                    sanitized_encoded = base64.b64encode(
                        sanitized_content
                        if isinstance(sanitized_content, bytes)
                        else sanitized_content.encode()
                    ).decode()

                    if url.startswith("data:") and header:
                        sanitized_url = f"{header},{sanitized_encoded}"
                        if isinstance(doc_data, dict):
                            doc_data["url"] = sanitized_url
                    elif isinstance(doc_data, dict):
                        doc_data["data"] = sanitized_encoded

                    verbose_proxy_logger.info(
                        "Document content modified by Prompt Security"
                    )

            return item
        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.error(f"Error sanitizing document: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Document sanitization failed: {str(e)}"
            )

    async def process_message_files(
        self, messages: list, user_api_key_alias: Optional[str] = None
    ) -> list:
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
                        item = await self._process_image_url_item(
                            item, user_api_key_alias
                        )
                    elif item_type in ["document", "file"]:
                        item = await self._process_document_item(
                            item, user_api_key_alias
                        )

                processed_content.append(item)

            processed_message = message.copy()
            processed_message["content"] = processed_content
            processed_messages.append(processed_message)

        return processed_messages

    @staticmethod
    def _resolve_key_alias(
        user_api_key_dict: Optional[UserAPIKeyAuth], data: Optional[dict]
    ) -> Optional[str]:
        if user_api_key_dict:
            alias = getattr(user_api_key_dict, "key_alias", None)
            if alias:
                return alias

        if data:
            metadata = data.get("metadata", {})
            alias = metadata.get("user_api_key_alias")
            if alias:
                return alias

        return None

    def filter_messages_by_role(self, messages: list) -> list:
        """Filter messages to only include standard OpenAI/Anthropic roles.

        Behavior depends on check_tool_results flag:
        - False (default): Filters out tool/function roles completely
        - True : Transforms tool/function to "other" role and includes them

        This allows checking tool results for indirect prompt injection when enabled.
        """
        supported_roles = ["system", "user", "assistant"]
        filtered_messages = []
        transformed_count = 0
        filtered_count = 0

        for message in messages:
            role = message.get("role", "")
            if role in supported_roles:
                filtered_messages.append(message)
            else:
                if self.check_tool_results:
                    transformed_message = {
                        "role": "other",
                        **{
                            key: value
                            for key, value in message.items()
                            if key != "role"
                        },
                    }
                    filtered_messages.append(transformed_message)
                    transformed_count += 1
                    verbose_proxy_logger.debug(
                        "Prompt Security Guardrail: Transformed message from role '%s' to 'other'",
                        role,
                    )
                else:
                    filtered_count += 1
                    verbose_proxy_logger.debug(
                        "Prompt Security Guardrail: Filtered message with role '%s'",
                        role,
                    )

        if transformed_count > 0:
            verbose_proxy_logger.debug(
                "Prompt Security Guardrail: Transformed %d tool/function messages to 'other' role",
                transformed_count,
            )

        if filtered_count > 0:
            verbose_proxy_logger.debug(
                "Prompt Security Guardrail: Filtered %d messages (%d -> %d messages)",
                filtered_count,
                len(messages),
                len(filtered_messages),
            )

        return filtered_messages

    def _build_headers(self, user_api_key_alias: Optional[str] = None) -> dict:
        headers = {"APP-ID": self.api_key, "Content-Type": "application/json"}
        if user_api_key_alias:
            headers["X-LiteLLM-Key-Alias"] = user_api_key_alias
        return headers

    @staticmethod
    def _redact_headers(headers: dict) -> dict:
        return {
            name: ("REDACTED" if name.lower() == "app-id" else value)
            for name, value in headers.items()
        }

    def _log_api_request(
        self,
        method: str,
        url: str,
        headers: dict,
        payload: Any,
    ) -> None:
        verbose_proxy_logger.debug(
            "Prompt Security request %s %s headers=%s payload=%s",
            method,
            url,
            self._redact_headers(headers),
            payload,
        )

    def _log_api_response(
        self,
        url: str,
        status_code: int,
        payload: Any,
    ) -> None:
        verbose_proxy_logger.debug(
            "Prompt Security response %s status=%s payload=%s",
            url,
            status_code,
            payload,
        )

    async def call_prompt_security_guardrail(
        self,
        data: dict,
        call_type: Optional[str] = None,
        user_api_key_alias: Optional[str] = None,
    ) -> dict:
        messages = data.get("messages", [])

        # Handle /responses endpoint by extracting messages from input
        if not messages and call_type:
            try:
                call_type_enum = CallTypes(call_type)
                if call_type_enum in {CallTypes.responses, CallTypes.aresponses}:
                    verbose_proxy_logger.debug(
                        "Prompt Security Guardrail: Extracting messages from /responses endpoint"
                    )
                    messages = self.get_guardrails_messages_for_call_type(
                        call_type=call_type_enum,
                        data=data,
                    )
            except (ValueError, AttributeError):
                pass

        verbose_proxy_logger.debug(
            "Prompt Security Guardrail: Processing %d messages", len(messages)
        )

        # First, sanitize any files in the messages
        messages = await self.process_message_files(
            messages, user_api_key_alias=user_api_key_alias
        )

        # Second, filter messages by role
        messages = self.filter_messages_by_role(messages)

        data["messages"] = messages

        # Then, run the regular prompt security check
        headers = self._build_headers(user_api_key_alias)
        self._log_api_request(
            method="POST",
            url=f"{self.api_base}/api/protect",
            headers=headers,
            payload={"messages": messages},
        )
        response = await self.async_handler.post(
            f"{self.api_base}/api/protect",
            headers=headers,
            json={
                "messages": messages,
                "user": user_api_key_alias or self.user,
                "system_prompt": self.system_prompt,
            },
        )
        response.raise_for_status()
        res = response.json()
        self._log_api_response(
            url=f"{self.api_base}/api/protect",
            status_code=response.status_code,
            payload={"result": res.get("result")},
        )
        result = res.get("result", {}).get("prompt", {})
        if result is None:  # prompt can exist but be with value None!
            return data
        action = result.get("action")
        violations = result.get("violations", [])
        if action == "block":
            raise HTTPException(
                status_code=400,
                detail="Blocked by Prompt Security, Violations: "
                + ", ".join(violations),
            )
        elif action == "modify":
            data["messages"] = result.get("modified_messages", [])
        return data

    async def call_prompt_security_guardrail_on_output(
        self, output: str, user_api_key_alias: Optional[str] = None
    ) -> dict:
        headers = self._build_headers(user_api_key_alias)
        self._log_api_request(
            method="POST",
            url=f"{self.api_base}/api/protect",
            headers=headers,
            payload={"response": output},
        )
        response = await self.async_handler.post(
            f"{self.api_base}/api/protect",
            headers=headers,
            json={
                "response": output,
                "user": user_api_key_alias or self.user,
                "system_prompt": self.system_prompt,
            },
        )
        response.raise_for_status()
        res = response.json()
        self._log_api_response(
            url=f"{self.api_base}/api/protect",
            status_code=response.status_code,
            payload={"result": res.get("result")},
        )
        result = res.get("result", {}).get("response", {})
        if result is None:  # prompt can exist but be with value None!
            return {}
        violations = result.get("violations", [])
        return {
            "action": result.get("action"),
            "modified_text": result.get("modified_text"),
            "violations": violations,
        }

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[Any, ModelResponse, EmbeddingResponse, ImageResponse],
    ) -> Any:
        verbose_proxy_logger.debug("Prompt Security Guardrail: Post-call hook")

        if (
            isinstance(response, ModelResponse)
            and response.choices
            and isinstance(response.choices[0], Choices)
        ):
            content = response.choices[0].message.content or ""
            verbose_proxy_logger.debug(
                "Prompt Security Guardrail: Checking response content (%d chars)",
                len(content),
            )
            alias = self._resolve_key_alias(user_api_key_dict, data)
            ret = await self.call_prompt_security_guardrail_on_output(
                content, user_api_key_alias=alias
            )
            violations = ret.get("violations", [])
            if ret.get("action") == "block":
                raise HTTPException(
                    status_code=400,
                    detail="Blocked by Prompt Security, Violations: "
                    + ", ".join(violations),
                )
            elif ret.get("action") == "modify":
                response.choices[0].message.content = ret.get("modified_text")
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        verbose_proxy_logger.debug(
            "Prompt Security Guardrail: Streaming response hook (window_size=%d)", 250
        )
        buffer: str = ""
        WINDOW_SIZE = 250  # Adjust window size as needed

        alias = self._resolve_key_alias(user_api_key_dict, request_data)

        async for item in response:
            if (
                not isinstance(item, ModelResponseStream)
                or not item.choices
                or len(item.choices) == 0
            ):
                yield item
                continue

            choice = item.choices[0]
            if choice.delta and choice.delta.content:
                buffer += choice.delta.content

            if choice.finish_reason or len(buffer) >= WINDOW_SIZE:
                if buffer:
                    if not choice.finish_reason and re.search(r"\s", buffer):
                        chunk, buffer = re.split(r"(?=\s\S*$)", buffer, 1)
                    else:
                        chunk, buffer = buffer, ""

                    ret = await self.call_prompt_security_guardrail_on_output(
                        chunk, user_api_key_alias=alias
                    )
                    violations = ret.get("violations", [])
                    if ret.get("action") == "block":
                        from litellm.proxy.proxy_server import StreamingCallbackError

                        raise StreamingCallbackError(
                            "Blocked by Prompt Security, Violations: "
                            + ", ".join(violations)
                        )
                    elif ret.get("action") == "modify":
                        chunk = ret.get("modified_text")

                    if choice.delta:
                        choice.delta.content = chunk
                    else:
                        choice.delta = Delta(content=chunk)
            yield item
        return

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.prompt_security import (
            PromptSecurityGuardrailConfigModel,
        )

        return PromptSecurityGuardrailConfigModel
