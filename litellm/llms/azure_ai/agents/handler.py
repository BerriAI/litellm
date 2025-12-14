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
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import httpx

from litellm._logging import verbose_logger
from litellm.llms.azure_ai.agents.transformation import (
    AzureAIAgentsConfig,
    AzureAIAgentsError,
)
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any
    HTTPHandler = Any
    AsyncHTTPHandler = Any


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
        return f"{api_base}/threads/{thread_id}/messages?api-version={api_version}"

    def _build_runs_url(self, api_base: str, thread_id: str, api_version: str) -> str:
        return f"{api_base}/threads/{thread_id}/runs?api-version={api_version}"

    def _build_run_status_url(self, api_base: str, thread_id: str, run_id: str, api_version: str) -> str:
        return f"{api_base}/threads/{thread_id}/runs/{run_id}?api-version={api_version}"

    def _build_list_messages_url(self, api_base: str, thread_id: str, api_version: str) -> str:
        return f"{api_base}/threads/{thread_id}/messages?api-version={api_version}"

    def _build_create_thread_and_run_url(self, api_base: str, api_version: str) -> str:
        """URL for the create-thread-and-run endpoint (supports streaming)."""
        return f"{api_base}/threads/runs?api-version={api_version}"

    # -------------------------------------------------------------------------
    # Response Helpers
    # -------------------------------------------------------------------------
    def _extract_content_from_messages(self, messages_data: dict) -> str:
        """Extract assistant content from the messages response."""
        for msg in messages_data.get("data", []):
            if msg.get("role") == "assistant":
                for content_item in msg.get("content", []):
                    if content_item.get("type") == "text":
                        return content_item.get("text", {}).get("value", "")
        return ""

    def _build_model_response(
        self,
        model: str,
        content: str,
        model_response: ModelResponse,
        thread_id: str,
        messages: List[Dict[str, Any]],
    ) -> ModelResponse:
        """Build the ModelResponse from agent output."""
        from litellm.types.utils import Choices, Message, Usage

        model_response.choices = [
            Choices(finish_reason="stop", index=0, message=Message(content=content, role="assistant"))
        ]
        model_response.model = model

        # Store thread_id for conversation continuity
        if not hasattr(model_response, "_hidden_params") or model_response._hidden_params is None:
            model_response._hidden_params = {}
        model_response._hidden_params["thread_id"] = thread_id

        # Estimate token usage
        try:
            from litellm.utils import token_counter

            prompt_tokens = token_counter(model="gpt-3.5-turbo", messages=messages)
            completion_tokens = token_counter(model="gpt-3.5-turbo", text=content, count_response_tokens=True)
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
            verbose_logger.warning(f"Failed to calculate token usage: {str(e)}")

        return model_response

    def _prepare_completion_params(
        self,
        model: str,
        api_base: str,
        api_key: str,
        optional_params: dict,
        headers: Optional[dict],
    ) -> tuple:
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

        api_version = optional_params.get("api_version", self.config.DEFAULT_API_VERSION)
        agent_id = self.config._get_agent_id(model, optional_params)
        thread_id = optional_params.get("thread_id")
        api_base = api_base.rstrip("/")

        verbose_logger.debug(f"Azure AI Agents completion - api_base: {api_base}, agent_id: {agent_id}")

        return headers, api_version, agent_id, thread_id, api_base

    def _check_response(self, response: httpx.Response, expected_codes: List[int], error_msg: str):
        """Check response status and raise error if not expected."""
        if response.status_code not in expected_codes:
            raise AzureAIAgentsError(status_code=response.status_code, message=f"{error_msg}: {response.text}")

    # -------------------------------------------------------------------------
    # Sync Completion
    # -------------------------------------------------------------------------
    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: str,
        api_key: str,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        timeout: float,
        client: Optional[HTTPHandler] = None,
        headers: Optional[dict] = None,
    ) -> ModelResponse:
        """Execute synchronous completion using Azure Agent Service."""
        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        if client is None:
            client = _get_httpx_client(params={"ssl_verify": litellm_params.get("ssl_verify", None)})

        headers, api_version, agent_id, thread_id, api_base = self._prepare_completion_params(
            model, api_base, api_key, optional_params, headers
        )

        def make_request(method: str, url: str, json_data: Optional[dict] = None) -> httpx.Response:
            if method == "GET":
                return client.get(url=url, headers=headers)
            return client.post(url=url, headers=headers, data=json.dumps(json_data) if json_data else None)

        # Execute the agent flow
        thread_id, content = self._execute_agent_flow_sync(
            make_request=make_request,
            api_base=api_base,
            api_version=api_version,
            agent_id=agent_id,
            thread_id=thread_id,
            messages=messages,
            optional_params=optional_params,
        )

        return self._build_model_response(model, content, model_response, thread_id, messages)

    def _execute_agent_flow_sync(
        self,
        make_request: Callable,
        api_base: str,
        api_version: str,
        agent_id: str,
        thread_id: Optional[str],
        messages: List[Dict[str, Any]],
        optional_params: dict,
    ) -> Tuple[str, str]:
        """Execute the agent flow synchronously. Returns (thread_id, content)."""
        
        # Step 1: Create thread if not provided
        if not thread_id:
            verbose_logger.debug(f"Creating thread at: {self._build_thread_url(api_base, api_version)}")
            response = make_request("POST", self._build_thread_url(api_base, api_version), {})
            self._check_response(response, [200, 201], "Failed to create thread")
            thread_id = response.json()["id"]
            verbose_logger.debug(f"Created thread: {thread_id}")

        # At this point thread_id is guaranteed to be a string
        assert thread_id is not None

        # Step 2: Add messages to thread
        for msg in messages:
            if msg.get("role") in ["user", "system"]:
                url = self._build_messages_url(api_base, thread_id, api_version)
                response = make_request("POST", url, {"role": "user", "content": msg.get("content", "")})
                self._check_response(response, [200, 201], "Failed to add message")

        # Step 3: Create run
        run_payload = {"assistant_id": agent_id}
        if "instructions" in optional_params:
            run_payload["instructions"] = optional_params["instructions"]
        
        response = make_request("POST", self._build_runs_url(api_base, thread_id, api_version), run_payload)
        self._check_response(response, [200, 201], "Failed to create run")
        run_id = response.json()["id"]
        verbose_logger.debug(f"Created run: {run_id}")

        # Step 4: Poll for completion
        status_url = self._build_run_status_url(api_base, thread_id, run_id, api_version)
        for _ in range(self.config.MAX_POLL_ATTEMPTS):
            response = make_request("GET", status_url)
            self._check_response(response, [200], "Failed to get run status")
            
            status = response.json().get("status")
            verbose_logger.debug(f"Run status: {status}")
            
            if status == "completed":
                break
            elif status in ["failed", "cancelled", "expired"]:
                error_msg = response.json().get("last_error", {}).get("message", "Unknown error")
                raise AzureAIAgentsError(status_code=500, message=f"Run {status}: {error_msg}")
            
            time.sleep(self.config.POLL_INTERVAL_SECONDS)
        else:
            raise AzureAIAgentsError(status_code=408, message="Run timed out waiting for completion")

        # Step 5: Get messages
        response = make_request("GET", self._build_list_messages_url(api_base, thread_id, api_version))
        self._check_response(response, [200], "Failed to get messages")
        
        content = self._extract_content_from_messages(response.json())
        return thread_id, content

    # -------------------------------------------------------------------------
    # Async Completion
    # -------------------------------------------------------------------------
    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: str,
        api_key: str,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        timeout: float,
        client: Optional[AsyncHTTPHandler] = None,
        headers: Optional[dict] = None,
    ) -> ModelResponse:
        """Execute asynchronous completion using Azure Agent Service."""
        import litellm
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        if client is None:
            client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.AZURE_AI,
                params={"ssl_verify": litellm_params.get("ssl_verify", None)},
            )

        headers, api_version, agent_id, thread_id, api_base = self._prepare_completion_params(
            model, api_base, api_key, optional_params, headers
        )

        async def make_request(method: str, url: str, json_data: Optional[dict] = None) -> httpx.Response:
            if method == "GET":
                return await client.get(url=url, headers=headers)
            return await client.post(url=url, headers=headers, data=json.dumps(json_data) if json_data else None)

        # Execute the agent flow
        thread_id, content = await self._execute_agent_flow_async(
            make_request=make_request,
            api_base=api_base,
            api_version=api_version,
            agent_id=agent_id,
            thread_id=thread_id,
            messages=messages,
            optional_params=optional_params,
        )

        return self._build_model_response(model, content, model_response, thread_id, messages)

    async def _execute_agent_flow_async(
        self,
        make_request: Callable,
        api_base: str,
        api_version: str,
        agent_id: str,
        thread_id: Optional[str],
        messages: List[Dict[str, Any]],
        optional_params: dict,
    ) -> Tuple[str, str]:
        """Execute the agent flow asynchronously. Returns (thread_id, content)."""
        
        # Step 1: Create thread if not provided
        if not thread_id:
            verbose_logger.debug(f"Creating thread at: {self._build_thread_url(api_base, api_version)}")
            response = await make_request("POST", self._build_thread_url(api_base, api_version), {})
            self._check_response(response, [200, 201], "Failed to create thread")
            thread_id = response.json()["id"]
            verbose_logger.debug(f"Created thread: {thread_id}")

        # At this point thread_id is guaranteed to be a string
        assert thread_id is not None

        # Step 2: Add messages to thread
        for msg in messages:
            if msg.get("role") in ["user", "system"]:
                url = self._build_messages_url(api_base, thread_id, api_version)
                response = await make_request("POST", url, {"role": "user", "content": msg.get("content", "")})
                self._check_response(response, [200, 201], "Failed to add message")

        # Step 3: Create run
        run_payload = {"assistant_id": agent_id}
        if "instructions" in optional_params:
            run_payload["instructions"] = optional_params["instructions"]
        
        response = await make_request("POST", self._build_runs_url(api_base, thread_id, api_version), run_payload)
        self._check_response(response, [200, 201], "Failed to create run")
        run_id = response.json()["id"]
        verbose_logger.debug(f"Created run: {run_id}")

        # Step 4: Poll for completion
        status_url = self._build_run_status_url(api_base, thread_id, run_id, api_version)
        for _ in range(self.config.MAX_POLL_ATTEMPTS):
            response = await make_request("GET", status_url)
            self._check_response(response, [200], "Failed to get run status")
            
            status = response.json().get("status")
            verbose_logger.debug(f"Run status: {status}")
            
            if status == "completed":
                break
            elif status in ["failed", "cancelled", "expired"]:
                error_msg = response.json().get("last_error", {}).get("message", "Unknown error")
                raise AzureAIAgentsError(status_code=500, message=f"Run {status}: {error_msg}")
            
            await asyncio.sleep(self.config.POLL_INTERVAL_SECONDS)
        else:
            raise AzureAIAgentsError(status_code=408, message="Run timed out waiting for completion")

        # Step 5: Get messages
        response = await make_request("GET", self._build_list_messages_url(api_base, thread_id, api_version))
        self._check_response(response, [200], "Failed to get messages")
        
        content = self._extract_content_from_messages(response.json())
        return thread_id, content

    # -------------------------------------------------------------------------
    # Streaming Completion (Native SSE)
    # -------------------------------------------------------------------------
    async def acompletion_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        api_base: str,
        api_key: str,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        timeout: float,
        headers: Optional[dict] = None,
    ) -> AsyncIterator:
        """Execute async streaming completion using Azure Agent Service with native SSE."""
        import litellm
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        headers, api_version, agent_id, thread_id, api_base = self._prepare_completion_params(
            model, api_base, api_key, optional_params, headers
        )

        # Build payload for create-thread-and-run with streaming
        thread_messages = []
        for msg in messages:
            if msg.get("role") in ["user", "system"]:
                thread_messages.append({
                    "role": "user",
                    "content": msg.get("content", "")
                })

        payload: Dict[str, Any] = {
            "assistant_id": agent_id,
            "stream": True,
        }
        
        # Add thread with messages if we don't have an existing thread
        if not thread_id:
            payload["thread"] = {"messages": thread_messages}
        
        if "instructions" in optional_params:
            payload["instructions"] = optional_params["instructions"]

        url = self._build_create_thread_and_run_url(api_base, api_version)
        verbose_logger.debug(f"Azure AI Agents streaming - URL: {url}")

        # Use LiteLLM's async HTTP client for streaming
        client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.AZURE_AI,
            params={"ssl_verify": litellm_params.get("ssl_verify", None)},
        )

        response = await client.post(
            url=url,
            headers=headers,
            data=json.dumps(payload),
            stream=True,
        )

        if response.status_code not in [200, 201]:
            error_text = await response.aread()
            raise AzureAIAgentsError(
                status_code=response.status_code,
                message=f"Streaming request failed: {error_text.decode()}"
            )

        async for chunk in self._process_sse_stream(response, model):
            yield chunk

    async def _process_sse_stream(
        self,
        response: httpx.Response,
        model: str,
    ) -> AsyncIterator:
        """Process SSE stream and yield OpenAI-compatible streaming chunks."""
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
        
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        thread_id = None
        
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
                    final_chunk = ModelResponseStream(
                        id=response_id,
                        created=created,
                        model=model,
                        object="chat.completion.chunk",
                        choices=[
                            StreamingChoices(
                                finish_reason="stop",
                                index=0,
                                delta=Delta(content=None),
                            )
                        ],
                    )
                    if thread_id:
                        final_chunk._hidden_params = {"thread_id": thread_id}
                    yield final_chunk
                    return
                
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                
                # Extract thread_id from thread.created event
                if current_event == "thread.created" and "id" in data:
                    thread_id = data["id"]
                    verbose_logger.debug(f"Stream created thread: {thread_id}")
                
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
azure_ai_agents_handler = AzureAIAgentsHandler()
