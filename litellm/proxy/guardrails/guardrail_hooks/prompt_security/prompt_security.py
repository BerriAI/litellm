import asyncio
import base64
import os
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Type

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
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

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Apply Prompt Security guardrail to the given inputs.

        This method is called by LiteLLM's guardrail framework for ALL endpoints:
        - /chat/completions
        - /responses
        - /messages (Anthropic)
        - /embeddings
        - /image/generations
        - /audio/transcriptions
        - /rerank
        - MCP server
        - and more...

        Args:
            inputs: Dictionary containing:
                - texts: List of texts to check
                - images: Optional list of image URLs
                - tool_calls: Optional list of tool calls
                - structured_messages: Optional full message structure
            request_data: The original request data
            input_type: "request" for input checking, "response" for output checking
            logging_obj: Optional logging object

        Returns:
            The inputs (potentially modified if action is "modify")

        Raises:
            HTTPException: If content is blocked by Prompt Security
        """
        texts = inputs.get("texts", [])
        images = inputs.get("images", [])
        structured_messages = inputs.get("structured_messages", [])

        # Resolve user API key alias from request metadata
        user_api_key_alias = self._resolve_key_alias_from_request_data(request_data)

        verbose_proxy_logger.debug(
            "Prompt Security Guardrail: apply_guardrail called with input_type=%s, "
            "texts=%d, images=%d, structured_messages=%d",
            input_type,
            len(texts),
            len(images),
            len(structured_messages),
        )

        if input_type == "request":
            return await self._apply_guardrail_on_request(
                inputs=inputs,
                texts=texts,
                images=images,
                structured_messages=structured_messages,
                request_data=request_data,
                user_api_key_alias=user_api_key_alias,
            )
        else:  # response
            return await self._apply_guardrail_on_response(
                inputs=inputs,
                texts=texts,
                user_api_key_alias=user_api_key_alias,
            )

    async def _apply_guardrail_on_request(
        self,
        inputs: GenericGuardrailAPIInputs,
        texts: List[str],
        images: List[str],
        structured_messages: list,
        request_data: dict,
        user_api_key_alias: Optional[str],
    ) -> GenericGuardrailAPIInputs:
        """Handle request-side guardrail checks."""
        # If we have structured messages, use them (they contain role information)
        # Otherwise, convert texts to simple user messages
        if structured_messages:
            messages = list(structured_messages)
        else:
            messages = [{"role": "user", "content": text} for text in texts]

        # Process any embedded files/images in messages
        messages = await self.process_message_files(
            messages, user_api_key_alias=user_api_key_alias
        )

        # Also process standalone images from inputs
        if images:
            await self._process_standalone_images(images, user_api_key_alias)

        # Filter messages by role for the API call
        filtered_messages = self.filter_messages_by_role(messages)

        if not filtered_messages:
            verbose_proxy_logger.debug(
                "Prompt Security Guardrail: No messages to check after filtering"
            )
            return inputs

        # Call Prompt Security API
        headers = self._build_headers(user_api_key_alias)
        payload = {
            "messages": filtered_messages,
            "user": user_api_key_alias or self.user,
            "system_prompt": self.system_prompt,
        }

        self._log_api_request(
            method="POST",
            url=f"{self.api_base}/api/protect",
            headers=headers,
            payload={"messages_count": len(filtered_messages)},
        )

        response = await self.async_handler.post(
            f"{self.api_base}/api/protect",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        res = response.json()

        self._log_api_response(
            url=f"{self.api_base}/api/protect",
            status_code=response.status_code,
            payload={"result": res.get("result")},
        )

        result = res.get("result", {}).get("prompt", {})
        if result is None:
            return inputs

        action = result.get("action")
        violations = result.get("violations", [])

        if action == "block":
            raise HTTPException(
                status_code=400,
                detail="Blocked by Prompt Security, Violations: "
                + ", ".join(violations),
            )
        elif action == "modify":
            # Extract modified texts from modified_messages
            modified_messages = result.get("modified_messages", [])
            modified_texts = self._extract_texts_from_messages(modified_messages)
            if modified_texts:
                inputs["texts"] = modified_texts

        return inputs

    async def _apply_guardrail_on_response(
        self,
        inputs: GenericGuardrailAPIInputs,
        texts: List[str],
        user_api_key_alias: Optional[str],
    ) -> GenericGuardrailAPIInputs:
        """Handle response-side guardrail checks."""
        if not texts:
            return inputs

        # Combine all texts for response checking
        combined_text = "\n".join(texts)

        headers = self._build_headers(user_api_key_alias)
        payload = {
            "response": combined_text,
            "user": user_api_key_alias or self.user,
            "system_prompt": self.system_prompt,
        }

        self._log_api_request(
            method="POST",
            url=f"{self.api_base}/api/protect",
            headers=headers,
            payload={"response_length": len(combined_text)},
        )

        response = await self.async_handler.post(
            f"{self.api_base}/api/protect",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        res = response.json()

        self._log_api_response(
            url=f"{self.api_base}/api/protect",
            status_code=response.status_code,
            payload={"result": res.get("result")},
        )

        result = res.get("result", {}).get("response", {})
        if result is None:
            return inputs

        action = result.get("action")
        violations = result.get("violations", [])

        if action == "block":
            raise HTTPException(
                status_code=400,
                detail="Blocked by Prompt Security, Violations: "
                + ", ".join(violations),
            )
        elif action == "modify":
            modified_text = result.get("modified_text")
            if modified_text is not None:
                # If we combined multiple texts, return the modified version as single text
                # The framework will handle distributing it back
                inputs["texts"] = [modified_text]

        return inputs

    def _extract_texts_from_messages(self, messages: list) -> List[str]:
        """Extract text content from messages."""
        texts = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text")
                        if text:
                            texts.append(text)
        return texts

    async def _process_standalone_images(
        self, images: List[str], user_api_key_alias: Optional[str]
    ) -> None:
        """Process standalone images from inputs (data URLs)."""
        for image_url in images:
            if image_url.startswith("data:"):
                try:
                    header, encoded = image_url.split(",", 1)
                    file_data = base64.b64decode(encoded)
                    mime_type = header.split(";")[0].split(":")[1]
                    extension = mime_type.split("/")[-1]
                    filename = f"image.{extension}"

                    result = await self.sanitize_file_content(
                        file_data, filename, user_api_key_alias=user_api_key_alias
                    )

                    if result.get("action") == "block":
                        violations = result.get("violations", [])
                        raise HTTPException(
                            status_code=400,
                            detail=f"Image blocked by Prompt Security. Violations: {', '.join(violations)}",
                        )
                except HTTPException:
                    raise
                except Exception as e:
                    verbose_proxy_logger.error(f"Error processing image: {str(e)}")

    @staticmethod
    def _resolve_key_alias_from_request_data(request_data: dict) -> Optional[str]:
        """Resolve user API key alias from request_data metadata."""
        # Check litellm_metadata first (set by guardrail framework)
        litellm_metadata = request_data.get("litellm_metadata", {})
        if litellm_metadata:
            alias = litellm_metadata.get("user_api_key_alias")
            if alias:
                return alias

        # Then check regular metadata
        metadata = request_data.get("metadata", {})
        if metadata:
            alias = metadata.get("user_api_key_alias")
            if alias:
                return alias

        return None

    async def sanitize_file_content(
        self,
        file_data: bytes,
        filename: str,
        user_api_key_alias: Optional[str] = None,
    ) -> dict:
        """
        Sanitize file content using Prompt Security API.
        Returns: dict with keys 'action', 'content', 'metadata'
        """
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

    def filter_messages_by_role(self, messages: list) -> list:
        """Filter messages to only include standard OpenAI/Anthropic roles.

        Behavior depends on check_tool_results flag:
        - False (default): Filters out tool/function roles completely
        - True: Transforms tool/function to "other" role and includes them

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

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.prompt_security import (
            PromptSecurityGuardrailConfigModel,
        )

        return PromptSecurityGuardrailConfigModel
