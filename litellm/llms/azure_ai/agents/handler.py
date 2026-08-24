"""
Handler for Azure Foundry Agent Service API.

This handler executes the multi-step agent flow:
1. Create thread (or use existing)
2. Add messages to thread
3. Create and poll a run
4. Retrieve the assistant's response messages

Model format: azure_ai/agents/<agent_id>
API Base format: https://<AIFoundryResourceName>.services.ai.azure.com/api/projects/<ProjectName>

Authentication: Uses Azure AD Bearer tokens (not API keys)
  Get token via: az account get-access-token --resource 'https://ai.azure.com'

Supports both polling-based and native streaming (SSE) modes.

See: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/quickstart
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Mapping
from typing import TYPE_CHECKING, Any, Final, Protocol, TypeAlias, TypedDict

import httpx
from typing_extensions import ReadOnly

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.azure_ai.agents.transformation import (
    AzureAIAgentsConfig,
    AzureAIAgentsError,
)
from litellm.types.llms.openai import (
    ChatCompletionAnnotation,
    ChatCompletionAnnotationURLCitation,
)
from litellm.types.utils import ModelResponse, ModelResponseStream

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any
    HTTPHandler = Any
    AsyncHTTPHandler = Any


class _AzureRawAnnotation(TypedDict, total=False):
    type: ReadOnly[str]
    text: ReadOnly[str]
    start_index: ReadOnly[int]
    end_index: ReadOnly[int]
    url_citation: ReadOnly[ChatCompletionAnnotationURLCitation]


_TransformedAnnotation: TypeAlias = ChatCompletionAnnotation | _AzureRawAnnotation


class _AzureText(TypedDict, total=False):
    value: ReadOnly[str]
    annotations: ReadOnly[list[_AzureRawAnnotation]]


class _AzureContentItem(TypedDict, total=False):
    type: ReadOnly[str]
    text: ReadOnly[_AzureText]


class _AzureMessage(TypedDict, total=False):
    role: ReadOnly[str]
    content: ReadOnly[list[_AzureContentItem]]


class _AzureMessagesData(TypedDict, total=False):
    data: ReadOnly[list[_AzureMessage]]


class _CreatedObject(TypedDict):
    id: ReadOnly[str]


class _RunError(TypedDict, total=False):
    message: ReadOnly[str]


class _RunStatus(TypedDict, total=False):
    status: ReadOnly[str]
    last_error: ReadOnly[_RunError]


class _SSEDelta(TypedDict, total=False):
    content: ReadOnly[list[_AzureContentItem]]


class _SSEEventData(TypedDict, total=False):
    id: ReadOnly[str]
    content: ReadOnly[list[_AzureContentItem]]
    delta: ReadOnly[_SSEDelta]


class _SyncAgentRequest(Protocol):
    def __call__(self, method: str, url: str, json_data: Mapping[str, object] | None = None) -> httpx.Response: ...


class _AsyncAgentRequest(Protocol):
    def __call__(
        self, method: str, url: str, json_data: Mapping[str, object] | None = None
    ) -> Awaitable[httpx.Response]: ...


class AzureAIAgentsHandler:
    """
    Handler for Azure AI Agent Service.

    Executes the complete agent flow which requires multiple API calls.
    """

    def __init__(self):
        self.config = AzureAIAgentsConfig()

    # -------------------------------------------------------------------------
    # URL Builders
    # -------------------------------------------------------------------------
    # Azure Foundry Agents API uses /assistants, /threads, etc. directly
    # See: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/quickstart
    # -------------------------------------------------------------------------
    def _build_thread_url(self, api_base: str, api_version: str) -> str:
        return f"{api_base}/threads?api-version={api_version}"

    def _build_messages_url(self, api_base: str, thread_id: str, api_version: str) -> str:
        encoded_thread_id: Final = encode_url_path_segment(thread_id, field_name="thread_id")
        return f"{api_base}/threads/{encoded_thread_id}/messages?api-version={api_version}"

    def _build_runs_url(self, api_base: str, thread_id: str, api_version: str) -> str:
        encoded_thread_id: Final = encode_url_path_segment(thread_id, field_name="thread_id")
        return f"{api_base}/threads/{encoded_thread_id}/runs?api-version={api_version}"

    def _build_run_status_url(self, api_base: str, thread_id: str, run_id: str, api_version: str) -> str:
        encoded_thread_id: Final = encode_url_path_segment(thread_id, field_name="thread_id")
        encoded_run_id: Final = encode_url_path_segment(run_id, field_name="run_id")
        return f"{api_base}/threads/{encoded_thread_id}/runs/{encoded_run_id}?api-version={api_version}"

    def _build_list_messages_url(self, api_base: str, thread_id: str, api_version: str) -> str:
        encoded_thread_id: Final = encode_url_path_segment(thread_id, field_name="thread_id")
        return f"{api_base}/threads/{encoded_thread_id}/messages?api-version={api_version}"

    def _build_create_thread_and_run_url(self, api_base: str, api_version: str) -> str:
        """URL for the create-thread-and-run endpoint (supports streaming)."""
        return f"{api_base}/threads/runs?api-version={api_version}"

    # -------------------------------------------------------------------------
    # Response Helpers
    # -------------------------------------------------------------------------
    def _extract_content_from_messages(
        self, messages_data: _AzureMessagesData
    ) -> tuple[str, list[_TransformedAnnotation] | None]:
        """Extract assistant content and annotations from the messages response.

        Returns (content, annotations) where annotations is a list of
        OpenAI-compatible ChatCompletionAnnotation dicts, or None.
        """
        for msg in messages_data.get("data", []):
            if msg.get("role") == "assistant":
                for content_item in msg.get("content", []):
                    if content_item.get("type") == "text":
                        text_obj = content_item.get("text", {})
                        content = text_obj.get("value", "")
                        raw_annotations = text_obj.get("annotations")
                        annotations = self._transform_annotations(raw_annotations)
                        return content, annotations
        return "", None

    def _transform_annotations(
        self,
        raw_annotations: list[_AzureRawAnnotation] | None,
    ) -> list[_TransformedAnnotation] | None:
        """Transform Azure AI Foundry annotations to OpenAI-compatible format.

        Azure AI returns annotations like:
            {"type": "url_citation", "text": "[1]", "start_index": 10,
             "end_index": 13, "url_citation": {"url": "...", "title": "..."}}

        OpenAI expects:
            {"type": "url_citation", "url_citation": {"url": "...", "title": "...",
             "start_index": 10, "end_index": 13}}
        """
        if not raw_annotations:
            return None

        result: Final[list[_TransformedAnnotation]] = []
        for ann in raw_annotations:
            ann_type = ann.get("type")
            if ann_type == "url_citation":
                url_citation: ChatCompletionAnnotationURLCitation = {**ann.get("url_citation", {})}
                # Azure puts start/end_index at annotation level; OpenAI
                # expects them inside url_citation
                if "start_index" in ann and "start_index" not in url_citation:
                    url_citation["start_index"] = ann["start_index"]
                if "end_index" in ann and "end_index" not in url_citation:
                    url_citation["end_index"] = ann["end_index"]
                result.append({"type": "url_citation", "url_citation": url_citation})
            else:
                # Pass through unknown annotation types as-is
                result.append(ann)

        return result if result else None

    def _build_model_response(
        self,
        model: str,
        content: str,
        model_response: ModelResponse,
        thread_id: str,
        messages: list[dict[str, object]],
        annotations: list[_TransformedAnnotation] | None = None,
    ) -> ModelResponse:
        """Build the ModelResponse from agent output."""
        from litellm.types.utils import Choices, Message, Usage

        message_kwargs: Final[dict[str, Any]] = {
            "content": content,
            "role": "assistant",
        }
        if annotations:
            message_kwargs["annotations"] = annotations

        model_response.choices = [
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(**message_kwargs),
            )
        ]
        model_response.model = model

        # Store thread_id for conversation continuity
        if not hasattr(model_response, "_hidden_params") or model_response._hidden_params is None:
            model_response._hidden_params = {}
        model_response._hidden_params["thread_id"] = thread_id

        # Estimate token usage
        try:
            from litellm.utils import token_counter

            prompt_tokens: Final = token_counter(model="gpt-3.5-turbo", messages=messages)
            completion_tokens: Final = token_counter(model="gpt-3.5-turbo", text=content, count_response_tokens=True)
            setattr(
                model_response,
                "usage",
                Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        except Exception as e:
            verbose_logger.warning("Failed to calculate token usage: %s", e)

        return model_response

    def _prepare_completion_params(
        self,
        model: str,
        api_base: str,
        api_key: str,
        optional_params: dict,
        headers: dict | None,
    ) -> tuple[dict[str, str], str, str, str | None, str]:
        """Prepare common parameters for completion.

        Azure Foundry Agents API uses Bearer token authentication:
        - Authorization: Bearer <token> (Azure AD token from 'az account get-access-token --resource https://ai.azure.com')

        See: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/quickstart
        """
        if headers is None:
            headers = {}
        headers["Content-Type"] = "application/json"

        # Azure Foundry Agents uses Bearer token authentication
        # The api_key here is expected to be an Azure AD token
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        api_version: Final = optional_params.get("api_version", self.config.DEFAULT_API_VERSION)
        agent_id: Final = self.config._get_agent_id(model, optional_params)
        thread_id: Final = optional_params.get("thread_id")
        api_base = api_base.rstrip("/")

        verbose_logger.debug("Azure AI Agents completion - api_base: %s, agent_id: %s", api_base, agent_id)

        return headers, api_version, agent_id, thread_id, api_base

    def _check_response(self, response: httpx.Response, expected_codes: list[int], error_msg: str):
        """Check response status and raise error if not expected."""
        if response.status_code not in expected_codes:
            raise AzureAIAgentsError(
                status_code=response.status_code,
                message=f"{error_msg}: {response.text}",
            )

    # -------------------------------------------------------------------------
    # Sync Completion
    # -------------------------------------------------------------------------
    def completion(
        self,
        model: str,
        messages: list[dict[str, object]],
        api_base: str,
        api_key: str,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        timeout: float,
        client: HTTPHandler | None = None,
        headers: dict | None = None,
    ) -> ModelResponse:
        """Execute synchronous completion using Azure Agent Service."""
        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        if client is None:
            client = _get_httpx_client(params={"ssl_verify": litellm_params.get("ssl_verify", None)})

        (
            headers,
            api_version,
            agent_id,
            thread_id,
            api_base,
        ) = self._prepare_completion_params(model, api_base, api_key, optional_params, headers)

        def make_request(method: str, url: str, json_data: Mapping[str, object] | None = None) -> httpx.Response:
            if method == "GET":
                return client.get(url=url, headers=headers)
            return client.post(
                url=url,
                headers=headers,
                data=json.dumps(json_data) if json_data else None,
            )

        # Execute the agent flow
        thread_id, content, annotations = self._execute_agent_flow_sync(
            make_request=make_request,
            api_base=api_base,
            api_version=api_version,
            agent_id=agent_id,
            thread_id=thread_id,
            messages=messages,
            optional_params=optional_params,
        )

        return self._build_model_response(model, content, model_response, thread_id, messages, annotations)

    def _execute_agent_flow_sync(
        self,
        make_request: _SyncAgentRequest,
        api_base: str,
        api_version: str,
        agent_id: str,
        thread_id: str | None,
        messages: list[dict[str, object]],
        optional_params: dict,
    ) -> tuple[str, str, list[_TransformedAnnotation] | None]:
        """Execute the agent flow synchronously. Returns (thread_id, content, annotations)."""

        # Step 1: Create thread if not provided
        if not thread_id:
            verbose_logger.debug("Creating thread at: %s", self._build_thread_url(api_base, api_version))
            response = make_request("POST", self._build_thread_url(api_base, api_version), {})
            self._check_response(response, [200, 201], "Failed to create thread")
            thread_data: Final[_CreatedObject] = response.json()
            thread_id = thread_data["id"]
            verbose_logger.debug("Created thread: %s", thread_id)

        # At this point thread_id is guaranteed to be a string
        assert thread_id is not None

        # Step 2: Add messages to thread
        for msg in messages:
            if msg.get("role") in ["user", "system"]:
                url = self._build_messages_url(api_base, thread_id, api_version)
                response = make_request("POST", url, {"role": "user", "content": msg.get("content", "")})
                self._check_response(response, [200, 201], "Failed to add message")

        # Step 3: Create run
        run_payload: Final = {"assistant_id": agent_id}
        if "instructions" in optional_params:
            run_payload["instructions"] = optional_params["instructions"]

        response = make_request("POST", self._build_runs_url(api_base, thread_id, api_version), run_payload)
        self._check_response(response, [200, 201], "Failed to create run")
        run_data: Final[_CreatedObject] = response.json()
        run_id: Final = run_data["id"]
        verbose_logger.debug("Created run: %s", run_id)

        # Step 4: Poll for completion
        status_url: Final = self._build_run_status_url(api_base, thread_id, run_id, api_version)
        for _ in range(self.config.MAX_POLL_ATTEMPTS):
            response = make_request("GET", status_url)
            self._check_response(response, [200], "Failed to get run status")

            status_data: _RunStatus = response.json()
            status = status_data.get("status")
            verbose_logger.debug("Run status: %s", status)

            if status == "completed":
                break
            elif status in ["failed", "cancelled", "expired"]:
                error_data: _RunStatus = response.json()
                error_msg = error_data.get("last_error", {}).get("message", "Unknown error")
                raise AzureAIAgentsError(status_code=500, message=f"Run {status}: {error_msg}")

            time.sleep(self.config.POLL_INTERVAL_SECONDS)
        else:
            raise AzureAIAgentsError(status_code=408, message="Run timed out waiting for completion")

        # Step 5: Get messages
        response = make_request("GET", self._build_list_messages_url(api_base, thread_id, api_version))
        self._check_response(response, [200], "Failed to get messages")

        messages_data: Final[_AzureMessagesData] = response.json()
        content, annotations = self._extract_content_from_messages(messages_data)
        return thread_id, content, annotations

    # -------------------------------------------------------------------------
    # Async Completion
    # -------------------------------------------------------------------------
    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, object]],
        api_base: str,
        api_key: str,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        timeout: float,
        client: AsyncHTTPHandler | None = None,
        headers: dict | None = None,
    ) -> ModelResponse:
        """Execute asynchronous completion using Azure Agent Service."""
        import litellm
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        if client is None:
            client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.AZURE_AI,
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )

        (
            headers,
            api_version,
            agent_id,
            thread_id,
            api_base,
        ) = self._prepare_completion_params(model, api_base, api_key, optional_params, headers)

        async def make_request(method: str, url: str, json_data: Mapping[str, object] | None = None) -> httpx.Response:
            if method == "GET":
                return await client.get(url=url, headers=headers)
            return await client.post(
                url=url,
                headers=headers,
                data=json.dumps(json_data) if json_data else None,
            )

        # Execute the agent flow
        thread_id, content, annotations = await self._execute_agent_flow_async(
            make_request=make_request,
            api_base=api_base,
            api_version=api_version,
            agent_id=agent_id,
            thread_id=thread_id,
            messages=messages,
            optional_params=optional_params,
        )

        return self._build_model_response(model, content, model_response, thread_id, messages, annotations)

    async def _execute_agent_flow_async(
        self,
        make_request: _AsyncAgentRequest,
        api_base: str,
        api_version: str,
        agent_id: str,
        thread_id: str | None,
        messages: list[dict[str, object]],
        optional_params: dict,
    ) -> tuple[str, str, list[_TransformedAnnotation] | None]:
        """Execute the agent flow asynchronously. Returns (thread_id, content, annotations)."""

        # Step 1: Create thread if not provided
        if not thread_id:
            verbose_logger.debug("Creating thread at: %s", self._build_thread_url(api_base, api_version))
            response = await make_request("POST", self._build_thread_url(api_base, api_version), {})
            self._check_response(response, [200, 201], "Failed to create thread")
            thread_data: Final[_CreatedObject] = response.json()
            thread_id = thread_data["id"]
            verbose_logger.debug("Created thread: %s", thread_id)

        # At this point thread_id is guaranteed to be a string
        assert thread_id is not None

        # Step 2: Add messages to thread
        for msg in messages:
            if msg.get("role") in ["user", "system"]:
                url = self._build_messages_url(api_base, thread_id, api_version)
                response = await make_request("POST", url, {"role": "user", "content": msg.get("content", "")})
                self._check_response(response, [200, 201], "Failed to add message")

        # Step 3: Create run
        run_payload: Final = {"assistant_id": agent_id}
        if "instructions" in optional_params:
            run_payload["instructions"] = optional_params["instructions"]

        response = await make_request("POST", self._build_runs_url(api_base, thread_id, api_version), run_payload)
        self._check_response(response, [200, 201], "Failed to create run")
        run_data: Final[_CreatedObject] = response.json()
        run_id: Final = run_data["id"]
        verbose_logger.debug("Created run: %s", run_id)

        # Step 4: Poll for completion
        status_url: Final = self._build_run_status_url(api_base, thread_id, run_id, api_version)
        for _ in range(self.config.MAX_POLL_ATTEMPTS):
            response = await make_request("GET", status_url)
            self._check_response(response, [200], "Failed to get run status")

            status_data: _RunStatus = response.json()
            status = status_data.get("status")
            verbose_logger.debug("Run status: %s", status)

            if status == "completed":
                break
            elif status in ["failed", "cancelled", "expired"]:
                error_data: _RunStatus = response.json()
                error_msg = error_data.get("last_error", {}).get("message", "Unknown error")
                raise AzureAIAgentsError(status_code=500, message=f"Run {status}: {error_msg}")

            await asyncio.sleep(self.config.POLL_INTERVAL_SECONDS)
        else:
            raise AzureAIAgentsError(status_code=408, message="Run timed out waiting for completion")

        # Step 5: Get messages
        response = await make_request("GET", self._build_list_messages_url(api_base, thread_id, api_version))
        self._check_response(response, [200], "Failed to get messages")

        messages_data: Final[_AzureMessagesData] = response.json()
        content, annotations = self._extract_content_from_messages(messages_data)
        return thread_id, content, annotations

    # -------------------------------------------------------------------------
    # Streaming Completion (Native SSE)
    # -------------------------------------------------------------------------
    async def acompletion_stream(
        self,
        model: str,
        messages: list[dict[str, object]],
        api_base: str,
        api_key: str,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        timeout: float,
        headers: dict | None = None,
    ) -> AsyncIterator[ModelResponseStream]:
        """Execute async streaming completion using Azure Agent Service with native SSE."""
        import litellm
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        (
            headers,
            api_version,
            agent_id,
            thread_id,
            api_base,
        ) = self._prepare_completion_params(model, api_base, api_key, optional_params, headers)

        # Build payload for create-thread-and-run with streaming
        thread_messages: Final[list[dict[str, object]]] = []
        for msg in messages:
            if msg.get("role") in ["user", "system"]:
                thread_messages.append({"role": "user", "content": msg.get("content", "")})

        payload: Final[dict[str, object]] = {
            "assistant_id": agent_id,
            "stream": True,
        }

        # Add thread with messages if we don't have an existing thread
        if not thread_id:
            payload["thread"] = {"messages": thread_messages}

        if "instructions" in optional_params:
            payload["instructions"] = optional_params["instructions"]

        url: Final = self._build_create_thread_and_run_url(api_base, api_version)
        verbose_logger.debug("Azure AI Agents streaming - URL: %s", url)

        # Use LiteLLM's async HTTP client for streaming
        client: Final = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.AZURE_AI,
            params={"ssl_verify": litellm_params.get("ssl_verify", None)},
        )

        response: Final = await client.post(
            url=url,
            headers=headers,
            data=json.dumps(payload),
            stream=True,
        )

        if response.status_code not in [200, 201]:
            error_text: Final = await response.aread()
            raise AzureAIAgentsError(
                status_code=response.status_code,
                message=f"Streaming request failed: {error_text.decode()}",
            )

        async for chunk in self._process_sse_stream(response, model):
            yield chunk

    async def _process_sse_stream(
        self,
        response: httpx.Response,
        model: str,
    ) -> AsyncIterator[ModelResponseStream]:
        """Process SSE stream and yield OpenAI-compatible streaming chunks."""
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        response_id: Final = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created: Final = int(time.time())
        thread_id = None
        collected_annotations: list[_TransformedAnnotation] | None = None

        current_event = None

        async for line in response.aiter_lines():
            line = line.strip()

            if line.startswith("event:"):
                current_event = line[6:].strip()
                continue

            if line.startswith("data:"):
                data_str = line[5:].strip()

                if data_str == "[DONE]":
                    # Send final chunk with finish_reason
                    final_delta_kwargs: dict[str, Any] = {"content": None}
                    if collected_annotations:
                        final_delta_kwargs["annotations"] = collected_annotations
                    final_chunk = ModelResponseStream(
                        id=response_id,
                        created=created,
                        model=model,
                        object="chat.completion.chunk",
                        choices=[
                            StreamingChoices(
                                finish_reason="stop",
                                index=0,
                                delta=Delta(**final_delta_kwargs),
                            )
                        ],
                    )
                    if thread_id:
                        final_chunk._hidden_params = {"thread_id": thread_id}
                    yield final_chunk
                    return

                try:
                    data: _SSEEventData = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                # Extract thread_id from thread.created event
                if current_event == "thread.created" and "id" in data:
                    thread_id = data["id"]
                    verbose_logger.debug("Stream created thread: %s", thread_id)

                # Extract annotations from completed message
                if current_event == "thread.message.completed":
                    for content_item in data.get("content", []):
                        if content_item.get("type") == "text":
                            raw_annotations = content_item.get("text", {}).get("annotations")
                            transformed = self._transform_annotations(raw_annotations)
                            if transformed:
                                if collected_annotations is None:
                                    collected_annotations = []
                                collected_annotations.extend(transformed)

                # Process message deltas - this is where the actual content comes
                if current_event == "thread.message.delta":
                    delta_content = data.get("delta", {}).get("content", [])
                    for content_item in delta_content:
                        if content_item.get("type") == "text":
                            text_value = content_item.get("text", {}).get("value", "")
                            if text_value:
                                chunk = ModelResponseStream(
                                    id=response_id,
                                    created=created,
                                    model=model,
                                    object="chat.completion.chunk",
                                    choices=[
                                        StreamingChoices(
                                            finish_reason=None,
                                            index=0,
                                            delta=Delta(content=text_value, role="assistant"),
                                        )
                                    ],
                                )
                                if thread_id:
                                    chunk._hidden_params = {"thread_id": thread_id}
                                yield chunk


# Singleton instance
azure_ai_agents_handler: Final = AzureAIAgentsHandler()
