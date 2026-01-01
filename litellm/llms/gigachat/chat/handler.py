"""
GigaChat Chat Completion Handler

Transforms OpenAI-format requests to GigaChat format and back.
Handles OAuth automatically via gigachat library.

Based on gpt2giga transformation logic.
"""

import base64
import hashlib
import json
import os
import re
import time
import uuid
import logging
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import httpx

from litellm._logging import verbose_logger
from litellm.llms.custom_llm import CustomLLM
from litellm.types.llms.openai import (
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
)
from litellm.types.utils import (
    Choices,
    EmbeddingResponse,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    Usage,
)

logger = logging.getLogger(__name__)


class AttachmentProcessor:
    """Handles image/file uploads to GigaChat with caching."""

    def __init__(self, giga_client: Any):
        self.giga = giga_client
        self.cache: Dict[str, str] = {}

    def _prepare_file_data(
        self, image_url: str, filename: Optional[str] = None
    ) -> Tuple[str, Optional[Tuple[str, bytes, str]], Optional[str]]:
        """
        Prepare file data for upload.
        Returns (url_hash, file_data, download_url) where:
        - file_data is (filename, content_bytes, content_type) or None if cached/need download
        - download_url is set if we need to download from URL
        """
        url_hash = hashlib.sha256(image_url.encode()).hexdigest()

        # Check cache
        if url_hash in self.cache:
            verbose_logger.debug(f"Image found in cache: {url_hash[:16]}...")
            return url_hash, None, None

        # Check for base64 data URL
        base64_match = re.search(r"data:(.+);base64,(.+)", image_url)

        if base64_match:
            content_type = base64_match.group(1)
            image_data = base64_match.group(2)
            content_bytes = base64.b64decode(image_data)
            verbose_logger.info("Decoded base64 image")
        else:
            # Signal to download from URL
            return url_hash, None, image_url

        ext = content_type.split("/")[-1].split(";")[0] or "jpg"
        final_filename = filename or f"{uuid.uuid4()}.{ext}"

        return url_hash, (final_filename, content_bytes, content_type), None

    async def upload_file(
        self, image_url: str, filename: Optional[str] = None
    ) -> Optional[str]:
        """Upload file to GigaChat and return file_id (async)."""
        try:
            url_hash, file_data, download_url = self._prepare_file_data(image_url, filename)

            # Return cached
            if file_data is None and download_url is None:
                return self.cache.get(url_hash)

            # Download if needed
            if download_url is not None:
                verbose_logger.info(f"Downloading image from URL: {download_url[:80]}...")
                async with httpx.AsyncClient() as client:
                    response = await client.get(download_url, timeout=30)
                    response.raise_for_status()
                content_type = response.headers.get("content-type", "image/jpeg")
                content_bytes = response.content
                ext = content_type.split("/")[-1].split(";")[0] or "jpg"
                final_filename = filename or f"{uuid.uuid4()}.{ext}"
            elif file_data is not None:
                final_filename, content_bytes, _ = file_data
            else:
                return None

            verbose_logger.info(f"Uploading file to GigaChat: {final_filename}")
            file = await self.giga.aupload_file((final_filename, content_bytes))

            self.cache[url_hash] = file.id_
            verbose_logger.info(f"File uploaded successfully, file_id: {file.id_}")
            return file.id_

        except Exception as e:
            verbose_logger.error(f"Error processing file: {e}")
            return None

    def upload_file_sync(
        self, image_url: str, filename: Optional[str] = None
    ) -> Optional[str]:
        """Upload file to GigaChat and return file_id (sync)."""
        try:
            url_hash, file_data, download_url = self._prepare_file_data(image_url, filename)

            # Return cached
            if file_data is None and download_url is None:
                return self.cache.get(url_hash)

            # Download if needed
            if download_url is not None:
                response = httpx.get(download_url, timeout=30)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "image/jpeg")
                content_bytes = response.content
                ext = content_type.split("/")[-1].split(";")[0] or "jpg"
                final_filename = filename or f"{uuid.uuid4()}.{ext}"
            elif file_data is not None:
                final_filename, content_bytes, _ = file_data
            else:
                return None

            file = self.giga.upload_file((final_filename, content_bytes))

            self.cache[url_hash] = file.id_
            return file.id_

        except Exception as e:
            verbose_logger.error(f"Error processing file: {e}")
            return None


class GigaChatChatHandler(CustomLLM):
    """
    Custom LLM handler for GigaChat API.

    Transforms OpenAI-format requests to GigaChat format and back.
    Handles OAuth automatically via gigachat library.

    Supported features:
    - Chat completion (sync/async)
    - Streaming (sync/async)
    - Function calling / Tools
    - Structured output (via function call emulation)
    - Image input (base64 and URL)
    - Embeddings
    """

    # Configuration
    MAX_IMAGES_PER_MESSAGE = 2
    MAX_TOTAL_IMAGES = 10

    def __init__(self):
        super().__init__()
        self._clients: Dict[str, Any] = {}
        self._attachment_processors: Dict[int, AttachmentProcessor] = {}
        # Track structured output requests by response_id
        self._structured_output_requests: Dict[str, str] = {}  # response_id -> schema_name

    def _get_client(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Get or create a GigaChat client with caching."""
        try:
            from gigachat import GigaChat
        except ImportError:
            raise ImportError(
                "GigaChat SDK not installed. Please install it with: pip install gigachat"
            )

        credentials = None
        scope = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
        access_token = None
        user = None
        password = None

        if api_key:
            if api_key.startswith("giga-cred-"):
                parts = api_key.replace("giga-cred-", "", 1).split(":")
                credentials = parts[0]
                scope = parts[1] if len(parts) > 1 else scope
            elif api_key.startswith("giga-auth-"):
                access_token = api_key.replace("giga-auth-", "", 1)
            elif api_key.startswith("giga-user-"):
                parts = api_key.replace("giga-user-", "", 1).split(":")
                user = parts[0]
                password = parts[1] if len(parts) > 1 else None
            else:
                credentials = api_key

        cache_key = f"{credentials}:{scope}:{access_token}:{user}:{api_base}"

        if cache_key not in self._clients:
            verify_ssl = os.environ.get("GIGACHAT_VERIFY_SSL_CERTS", "False").lower() in ("true", "1", "yes")
            client_kwargs = {
                "verify_ssl_certs": verify_ssl,
                "timeout": kwargs.get("timeout", 600),
            }

            if credentials:
                client_kwargs["credentials"] = credentials
                client_kwargs["scope"] = scope
            elif access_token:
                client_kwargs["access_token"] = access_token
            elif user and password:
                client_kwargs["user"] = user
                client_kwargs["password"] = password

            if api_base:
                client_kwargs["base_url"] = api_base

            self._clients[cache_key] = GigaChat(**client_kwargs)

        return self._clients[cache_key]

    def _get_attachment_processor(self, giga_client: Any) -> AttachmentProcessor:
        """Get or create an AttachmentProcessor for the client."""
        client_id = id(giga_client)
        if client_id not in self._attachment_processors:
            self._attachment_processors[client_id] = AttachmentProcessor(giga_client)
        return self._attachment_processors[client_id]

    # ==================== REQUEST TRANSFORMATION ====================

    def _transform_single_message(self, message: Dict, index: int) -> Dict:
        """Transform a single message from OpenAI to GigaChat format (no I/O)."""
        msg = message.copy()

        # Remove unsupported fields
        msg.pop("name", None)

        # Transform roles
        role = msg.get("role", "user")
        if role == "developer":
            msg["role"] = "system"
        elif role == "system" and index > 0:
            msg["role"] = "user"
        elif role == "tool":
            msg["role"] = "function"
            content = msg.get("content", "")
            if not isinstance(content, str):
                msg["content"] = json.dumps(content, ensure_ascii=False)

        # Handle None content
        if msg.get("content") is None:
            msg["content"] = ""

        # Transform tool_calls to function_call
        if "tool_calls" in msg and msg["tool_calls"]:
            tool_call = msg["tool_calls"][0]
            func = tool_call.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            msg["function_call"] = {
                "name": func.get("name", ""),
                "arguments": args,
            }
            msg.pop("tool_calls", None)

        return msg

    def _transform_messages_sync(
        self,
        messages: List[Dict],
        attachment_processor: Optional[AttachmentProcessor] = None,
    ) -> List[Dict]:
        """Transform OpenAI messages to GigaChat format (sync)."""
        transformed = []
        total_attachments = 0

        for i, msg in enumerate(messages):
            message = self._transform_single_message(msg, i)

            # Handle list content (multimodal)
            if isinstance(message.get("content"), list):
                texts, attachments = self._process_content_parts(
                    message["content"], attachment_processor, is_async=False
                )
                message["content"] = "\n".join(texts)
                if attachments:
                    message["attachments"] = attachments
                    total_attachments += len(attachments)

            transformed.append(message)

        if total_attachments > self.MAX_TOTAL_IMAGES:
            self._limit_attachments(transformed)

        return self._collapse_user_messages(transformed)

    async def _transform_messages_async(
        self,
        messages: List[Dict],
        attachment_processor: Optional[AttachmentProcessor] = None,
    ) -> List[Dict]:
        """Transform OpenAI messages to GigaChat format (async)."""
        transformed = []
        total_attachments = 0

        for i, msg in enumerate(messages):
            message = self._transform_single_message(msg, i)

            # Handle list content (multimodal)
            if isinstance(message.get("content"), list):
                texts, attachments = await self._process_content_parts_async(
                    message["content"], attachment_processor
                )
                message["content"] = "\n".join(texts)
                if attachments:
                    message["attachments"] = attachments
                    total_attachments += len(attachments)

            transformed.append(message)

        if total_attachments > self.MAX_TOTAL_IMAGES:
            self._limit_attachments(transformed)

        return self._collapse_user_messages(transformed)

    def _process_content_parts(
        self,
        content_parts: List[Dict],
        attachment_processor: Optional[AttachmentProcessor],
        is_async: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """Process multimodal content parts (sync)."""
        texts: List[str] = []
        attachments: List[str] = []

        for part in content_parts:
            part_type = part.get("type")

            if part_type == "text":
                texts.append(part.get("text", ""))

            elif part_type == "image_url" and attachment_processor:
                if len(attachments) >= self.MAX_IMAGES_PER_MESSAGE:
                    verbose_logger.warning("Max images per message reached, skipping")
                    continue
                image_url = part.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if url:
                    file_id = attachment_processor.upload_file_sync(url)
                    if file_id:
                        attachments.append(file_id)

            elif part_type == "file" and attachment_processor:
                file_data = part.get("file", {})
                filename = file_data.get("filename")
                data = file_data.get("file_data")
                if data:
                    file_id = attachment_processor.upload_file_sync(data, filename)
                    if file_id:
                        attachments.append(file_id)

        return texts, attachments

    async def _process_content_parts_async(
        self,
        content_parts: List[Dict],
        attachment_processor: Optional[AttachmentProcessor],
    ) -> Tuple[List[str], List[str]]:
        """Process multimodal content parts (async)."""
        texts: List[str] = []
        attachments: List[str] = []

        for part in content_parts:
            part_type = part.get("type")

            if part_type == "text":
                texts.append(part.get("text", ""))

            elif part_type == "image_url" and attachment_processor:
                if len(attachments) >= self.MAX_IMAGES_PER_MESSAGE:
                    verbose_logger.warning("Max images per message reached, skipping")
                    continue
                image_url = part.get("image_url", {})
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if url:
                    file_id = await attachment_processor.upload_file(url)
                    if file_id:
                        attachments.append(file_id)

            elif part_type == "file" and attachment_processor:
                file_data = part.get("file", {})
                filename = file_data.get("filename")
                data = file_data.get("file_data")
                if data:
                    file_id = await attachment_processor.upload_file(data, filename)
                    if file_id:
                        attachments.append(file_id)

        return texts, attachments

    def _limit_attachments(self, messages: List[Dict]):
        """Limit total attachments to MAX_TOTAL_IMAGES (keep most recent)."""
        current_count = 0
        for message in reversed(messages):
            msg_attachments = message.get("attachments", [])
            if current_count + len(msg_attachments) > self.MAX_TOTAL_IMAGES:
                allowed = self.MAX_TOTAL_IMAGES - current_count
                message["attachments"] = msg_attachments[-allowed:] if allowed > 0 else []
                verbose_logger.warning(f"Limited attachments in message to {allowed}")
            current_count += len(message.get("attachments", []))

    def _collapse_user_messages(self, messages: List[Dict]) -> List[Dict]:
        """Collapse consecutive user messages into one."""
        collapsed: List[Dict] = []
        prev_user_msg: Optional[Dict] = None
        content_parts: List[str] = []

        for msg in messages:
            if msg.get("role") == "user" and prev_user_msg is not None:
                content_parts.append(msg.get("content", ""))
            else:
                if content_parts and prev_user_msg:
                    prev_user_msg["content"] = "\n".join(
                        [prev_user_msg.get("content", "")] + content_parts
                    )
                    content_parts = []
                collapsed.append(msg)
                prev_user_msg = msg if msg.get("role") == "user" else None

        if content_parts and prev_user_msg:
            prev_user_msg["content"] = "\n".join(
                [prev_user_msg.get("content", "")] + content_parts
            )

        return collapsed

    def _transform_tools_to_functions(self, tools: Optional[List[Dict]]) -> Optional[List[Any]]:
        """Convert OpenAI tools format to GigaChat functions format."""
        if not tools:
            return None

        try:
            from gigachat.models import Function, FunctionParameters
        except ImportError:
            raise ImportError(
                "GigaChat SDK not installed. Please install it with: pip install gigachat"
            )

        functions = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", tool)
            else:
                func = tool

            giga_func = Function(
                name=func.get("name", ""),
                description=func.get("description", ""),
                parameters=FunctionParameters(**func.get("parameters", {})),
            )
            functions.append(giga_func)

        return functions

    def _transform_params(self, optional_params: Dict, response_id: str) -> Tuple[Dict, bool]:
        """Transform optional parameters to GigaChat format.

        Returns:
            Tuple of (transformed_params, is_structured_output)
        """
        try:
            from gigachat.models import Function, FunctionParameters
        except ImportError:
            raise ImportError(
                "GigaChat SDK not installed. Please install it with: pip install gigachat"
            )

        params = optional_params.copy()
        is_structured_output = False

        # Handle temperature
        temperature = params.pop("temperature", 0)
        if temperature == 0:
            params["top_p"] = 0
        elif temperature > 0:
            params["temperature"] = temperature

        # Handle max_tokens variations
        max_tokens = params.pop("max_completion_tokens", None) or params.pop("max_output_tokens", None)
        if max_tokens:
            params["max_tokens"] = max_tokens

        # Handle response_format (structured output via function calling)
        response_format = params.pop("response_format", None)
        if response_format:
            if response_format.get("type") == "json_schema":
                # Convert json_schema to virtual function call
                json_schema = response_format.get("json_schema", {})
                schema_name = json_schema.get("name", "structured_output")
                schema = json_schema.get("schema", {})

                # Create virtual function from schema
                function_def = Function(
                    name=schema_name,
                    description=f"Output response in structured format: {schema_name}",
                    parameters=FunctionParameters(**schema) if schema else FunctionParameters(type="object", properties={}),
                )

                # Add to existing functions or create new list
                if "functions" not in params:
                    params["_structured_output_function"] = function_def
                else:
                    params["functions"].append(function_def)

                # Force GigaChat to call this function
                params["function_call"] = {"name": schema_name}

                # Track this request as structured output
                is_structured_output = True
                self._structured_output_requests[response_id] = schema_name
            else:
                # For json_object type, pass through
                params["response_format"] = {
                    "type": response_format.get("type"),
                    **response_format.get("json_schema", {}),
                }

        # Remove unsupported params
        unsupported = ["logprobs", "top_logprobs", "n", "presence_penalty",
                      "frequency_penalty", "logit_bias", "user", "seed",
                      "parallel_tool_calls", "service_tier", "stream"]
        for key in unsupported:
            params.pop(key, None)

        return params, is_structured_output

    # ==================== RESPONSE TRANSFORMATION ====================

    def _transform_response(
        self,
        giga_response: Any,
        model: str,
        response_id: str,
    ) -> ModelResponse:
        """Transform GigaChat response to OpenAI format."""
        giga_dict = giga_response.dict()

        # Check if this was a structured output request
        is_structured_output = response_id in self._structured_output_requests

        choices = []
        for choice in giga_dict.get("choices", []):
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "stop")

            # Transform function_call to tool_calls or content (for structured output)
            if finish_reason == "function_call" and message.get("function_call"):
                func_call = message["function_call"]
                args = func_call.get("arguments", {})

                if is_structured_output:
                    # For structured output: convert function arguments to content
                    if isinstance(args, dict):
                        content = json.dumps(args, ensure_ascii=False)
                    else:
                        content = str(args)

                    message["content"] = content
                    message.pop("function_call", None)
                    message.pop("functions_state_id", None)  # Remove GigaChat-specific field
                    finish_reason = "stop"
                else:
                    # Regular function call: convert to tool_calls format
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)

                    message["tool_calls"] = [{
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": func_call.get("name", ""),
                            "arguments": args,
                        }
                    }]
                    message.pop("function_call", None)
                    finish_reason = "tool_calls"

            message["refusal"] = None

            choices.append(
                Choices(
                    index=0,
                    message=Message(**message),
                    finish_reason=finish_reason,
                    logprobs=None,
                )
            )

        usage_data = giga_dict.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Cleanup tracking
        if is_structured_output:
            del self._structured_output_requests[response_id]

        return ModelResponse(
            id=f"chatcmpl-{response_id}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=usage,
            system_fingerprint=f"fp_{response_id}",
        )

    def _transform_stream_chunk(
        self,
        chunk: Any,
        model: str,
        response_id: str,
    ) -> GenericStreamingChunk:
        """Transform GigaChat streaming chunk to LiteLLM format."""
        chunk_dict = chunk.dict()

        # Check if this was a structured output request
        is_structured_output = response_id in self._structured_output_requests

        choice = chunk_dict.get("choices", [{}])[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        text = delta.get("content", "")
        tool_use: Optional[ChatCompletionToolCallChunk] = None

        # Handle function_call in stream
        if finish_reason == "function_call" and delta.get("function_call"):
            func_call = delta["function_call"]
            args = func_call.get("arguments", {})

            if is_structured_output:
                # For structured output: convert function arguments to text content
                if isinstance(args, dict):
                    text = json.dumps(args, ensure_ascii=False)
                else:
                    text = str(args)
                finish_reason = "stop"
                # Cleanup tracking on final chunk
                if response_id in self._structured_output_requests:
                    del self._structured_output_requests[response_id]
            else:
                # Regular function call: convert to tool_calls format
                finish_reason = "tool_calls"
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                tool_use = ChatCompletionToolCallChunk(
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    type="function",
                    function=ChatCompletionToolCallFunctionChunk(
                        name=func_call.get("name", ""),
                        arguments=args,
                    ),
                    index=0,
                )

        return GenericStreamingChunk(
            text=text,
            tool_use=tool_use,
            is_finished=finish_reason is not None,
            finish_reason=finish_reason,
            usage=None,
            index=0,
        )

    def _transform_embedding(self, giga_response: Any) -> EmbeddingResponse:
        """Transform GigaChat embeddings to LiteLLM format."""
        result = giga_response.dict(by_alias=True)
        used_tokens = sum(emb.usage.prompt_tokens for emb in giga_response.data)
        result["usage"] = {
            "prompt_tokens": used_tokens,
            "total_tokens": used_tokens
        }
        return EmbeddingResponse.model_validate(result)

    # ==================== COMPLETION METHODS ====================

    def _build_chat_request(
        self,
        model: str,
        giga_messages: List[Dict],
        giga_params: Dict,
        optional_params: Dict,
    ) -> Any:
        """Build GigaChat Chat request object."""
        try:
            from gigachat.models import Chat, Messages
        except ImportError:
            raise ImportError(
                "GigaChat SDK not installed. Please install it with: pip install gigachat"
            )

        tools = optional_params.get("tools") or optional_params.get("functions")
        functions = self._transform_tools_to_functions(tools)

        # Handle structured output function
        structured_output_function = giga_params.pop("_structured_output_function", None)
        if structured_output_function:
            if functions is None:
                functions = []
            functions.append(structured_output_function)

        return Chat(
            messages=[Messages(**m) for m in giga_messages],
            model=model,
            functions=functions,
            **giga_params,
        )

    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: Optional[str],
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> ModelResponse:
        """Synchronous completion."""
        response_id = uuid.uuid4().hex[:12]

        giga_client = self._get_client(api_key=api_key, api_base=api_base, timeout=timeout or 600)
        attachment_processor = self._get_attachment_processor(giga_client)

        giga_messages = self._transform_messages_sync(messages, attachment_processor)
        giga_params, _ = self._transform_params(optional_params, response_id)
        chat = self._build_chat_request(model, giga_messages, giga_params, optional_params)

        print_verbose(f"GigaChat request: {chat}")
        response = giga_client.chat(chat)
        print_verbose(f"GigaChat response: {response}")

        return self._transform_response(response, model, response_id)

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: Optional[str],
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> ModelResponse:
        """Async completion."""
        response_id = uuid.uuid4().hex[:12]

        giga_client = self._get_client(api_key=api_key, api_base=api_base, timeout=timeout or 600)
        attachment_processor = self._get_attachment_processor(giga_client)

        giga_messages = await self._transform_messages_async(messages, attachment_processor)
        giga_params, _ = self._transform_params(optional_params, response_id)
        chat = self._build_chat_request(model, giga_messages, giga_params, optional_params)

        print_verbose(f"GigaChat async request: {chat}")
        response = await giga_client.achat(chat)
        print_verbose(f"GigaChat async response: {response}")

        return self._transform_response(response, model, response_id)

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: Optional[str],
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> Iterator[GenericStreamingChunk]:
        """Synchronous streaming."""
        response_id = uuid.uuid4().hex[:12]

        giga_client = self._get_client(api_key=api_key, api_base=api_base, timeout=timeout or 600)
        attachment_processor = self._get_attachment_processor(giga_client)

        giga_messages = self._transform_messages_sync(messages, attachment_processor)
        giga_params, _ = self._transform_params(optional_params, response_id)
        chat = self._build_chat_request(model, giga_messages, giga_params, optional_params)

        print_verbose(f"GigaChat streaming request: {chat}")

        for chunk in giga_client.stream(chat):
            yield self._transform_stream_chunk(chunk, model, response_id)

    async def astreaming(  # type: ignore[override]
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key: Optional[str],
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client=None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Async streaming (yields chunks, so actual return is AsyncGenerator)."""
        response_id = uuid.uuid4().hex[:12]

        giga_client = self._get_client(api_key=api_key, api_base=api_base, timeout=timeout or 600)
        attachment_processor = self._get_attachment_processor(giga_client)

        giga_messages = await self._transform_messages_async(messages, attachment_processor)
        giga_params, _ = self._transform_params(optional_params, response_id)
        chat = self._build_chat_request(model, giga_messages, giga_params, optional_params)

        print_verbose(f"GigaChat async streaming request: {chat}")

        async for chunk in giga_client.astream(chat):
            yield self._transform_stream_chunk(chunk, model, response_id)

    # ==================== EMBEDDING METHODS ====================

    def embedding(
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        print_verbose: Callable,
        logging_obj: Any,
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        litellm_params=None,
    ) -> EmbeddingResponse:
        """Synchronous embedding."""
        giga_client = self._get_client(api_key=api_key, api_base=api_base, timeout=timeout or 600)

        print_verbose(f"GigaChat embedding request: {input}")

        result = giga_client.embeddings(input, model)
        print_verbose(f'Generate embedding: {result}')

        return self._transform_embedding(result)

    async def aembedding(
        self,
        model: str,
        input: list,
        model_response: EmbeddingResponse,
        print_verbose: Callable,
        logging_obj: Any,
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        litellm_params=None,
    ) -> EmbeddingResponse:
        """Async embedding."""
        giga_client = self._get_client(api_key=api_key, api_base=api_base, timeout=timeout or 600)

        print_verbose(f"GigaChat embedding request: {input}")

        result = await giga_client.aembeddings(input, model)
        print_verbose(f'Generate embedding: {result}')

        return self._transform_embedding(result)

    # ==================== MODEL LISTING ====================

    def get_models(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> List[str]:
        """Get list of available GigaChat models from API."""
        giga_client = self._get_client(api_key=api_key, api_base=api_base)
        models = giga_client.get_models()
        return [f"gigachat/{m.id_}" for m in models.data]

    async def aget_models(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> List[str]:
        """Async version of get_models."""
        giga_client = self._get_client(api_key=api_key, api_base=api_base)
        models = await giga_client.aget_models()
        return [f"gigachat/{m.id_}" for m in models.data]


# Instance for LiteLLM to use
gigachat_chat_handler = GigaChatChatHandler()
