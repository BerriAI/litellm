"""
Transformation for Bedrock AgentCore

https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agentcore_InvokeAgentRuntime.html
"""

import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast
from urllib.parse import quote

import httpx

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.llms.bedrock_agentcore import (
    AgentCoreMessage,
    AgentCoreParsedResponse,
    AgentCoreUsage,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Delta, Message, ModelResponse, StreamingChoices, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any
    HTTPHandler = Any
    AsyncHTTPHandler = Any


class AmazonAgentCoreConfig(BaseConfig, BaseAWSLLM):
    def __init__(self, **kwargs):
        BaseConfig.__init__(self, **kwargs)
        BaseAWSLLM.__init__(self, **kwargs)

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        Bedrock AgentCore has 0 OpenAI compatible params
        """
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI params to AgentCore params
        """
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete url for the request
        """
        ### SET RUNTIME ENDPOINT ###
        aws_bedrock_runtime_endpoint = optional_params.get(
            "aws_bedrock_runtime_endpoint", None
        )

        # Extract ARN from model string
        agent_runtime_arn = self._get_agent_runtime_arn(model)

        # Parse ARN to get region
        region = self._extract_region_from_arn(agent_runtime_arn)

        # Build the base endpoint URL for AgentCore
        # Note: We don't use get_runtime_endpoint as AgentCore has its own endpoint structure
        if aws_bedrock_runtime_endpoint:
            base_url = aws_bedrock_runtime_endpoint
        else:
            base_url = f"https://bedrock-agentcore.{region}.amazonaws.com"

        # Based on boto3 client.invoke_agent_runtime, the path is:
        # /runtimes/{URL-ENCODED-ARN}/invocations?qualifier=<value>
        encoded_arn = quote(agent_runtime_arn, safe="")
        endpoint_url = f"{base_url}/runtimes/{encoded_arn}/invocations"

        # Add qualifier as query parameter if provided
        if "qualifier" in optional_params:
            endpoint_url = f"{endpoint_url}?qualifier={optional_params['qualifier']}"

        return endpoint_url

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        # Check if api_key (bearer token) is provided for Cognito authentication
        # Priority: api_key parameter first, then optional_params
        jwt_token = api_key or optional_params.get("api_key")
        if jwt_token:
            verbose_logger.debug(
                f"AgentCore: Using Bearer token authentication (Cognito/JWT) - token: {jwt_token[:50]}..."
            )
            headers["Content-Type"] = "application/json"
            headers["Authorization"] = f"Bearer {jwt_token}"
            # Return headers with bearer token and JSON-encoded body (not SigV4 signed)
            return headers, json.dumps(request_data).encode()

        # Otherwise, use AWS SigV4 authentication
        verbose_logger.debug("AgentCore: Using AWS SigV4 authentication (IAM)")
        return self._sign_request(
            service_name="bedrock-agentcore",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
            api_key=api_key,
        )

    def _get_agent_runtime_arn(self, model: str) -> str:
        """
        Extract ARN from model string
        model = "agentcore/arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC"
        returns: "arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC"
        """
        parts = model.split("/", 1)
        if len(parts) != 2 or parts[0] != "agentcore":
            raise ValueError(
                "Invalid model format. Expected format: 'model=bedrock/agentcore/arn:aws:bedrock-agentcore:region:account:runtime/runtime_id'"
            )
        return parts[1]

    def _extract_region_from_arn(self, arn: str) -> str:
        """
        Extract region from ARN
        arn:aws:bedrock-agentcore:us-west-2:888602223428:runtime/hosted_agent_r9jvp-3ySZuRHjLC
        returns: us-west-2
        """
        parts = arn.split(":")
        if len(parts) >= 4:
            return parts[3]
        raise ValueError(f"Invalid ARN format: {arn}")

    def _get_runtime_session_id(self, optional_params: dict) -> str:
        """
        Get or generate runtime session ID (must be 33+ chars)
        """
        session_id = optional_params.get("runtimeSessionId", None)
        if session_id:
            verbose_logger.debug(f"Using provided runtimeSessionId: {session_id}")
            return session_id

        # Generate a session ID with 33+ characters
        generated_id = f"litellm-session-{str(uuid.uuid4())}"
        verbose_logger.debug(f"Generated new session ID: {generated_id}")
        return generated_id

    def _get_runtime_user_id(self, optional_params: dict) -> Optional[str]:
        """
        Get runtime user ID if provided
        """
        user_id = optional_params.get("runtimeUserId", None)
        if user_id:
            verbose_logger.debug(f"Using provided runtimeUserId: {user_id}")
        return user_id

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request to AgentCore format.

        Based on boto3's implementation:
        - Session ID goes in header: X-Amzn-Bedrock-AgentCore-Runtime-Session-Id
        - User ID goes in header: X-Amzn-Bedrock-AgentCore-Runtime-User-Id
        - Qualifier goes as query parameter
        - Only the payload goes in the request body

        Returns:
            dict: Payload dict containing the prompt
        """
        verbose_logger.debug(
            f"AgentCore transform_request - optional_params keys: {list(optional_params.keys())}"
        )

        # Use the last message content as the prompt
        prompt = convert_content_list_to_str(messages[-1])

        # Create the payload - this is what goes in the body (raw JSON)
        payload: dict = {"prompt": prompt}

        # Get or generate session ID - this goes in the header
        runtime_session_id = self._get_runtime_session_id(optional_params)
        headers["X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"] = runtime_session_id

        # Get user ID if provided - this goes in the header
        runtime_user_id = self._get_runtime_user_id(optional_params)
        if runtime_user_id:
            headers["X-Amzn-Bedrock-AgentCore-Runtime-User-Id"] = runtime_user_id

        # The request data is the payload dict (will be JSON encoded by the HTTP handler)
        # Qualifier will be handled as a query parameter in get_complete_url

        verbose_logger.debug(f"PAYLOAD: {payload}")
        return payload

    def _extract_sse_json(self, line: str) -> Optional[Dict]:
        """Extract and parse JSON from an SSE data line."""
        if not line.startswith("data:"):
            return None

        json_str = line[5:].strip()
        if not json_str:
            return None

        try:
            data = json.loads(json_str)
            # Skip non-dict data (some lines contain JSON strings)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            verbose_logger.debug(f"Skipping non-JSON line: {line[:100]}")
            return None

    def _extract_usage_from_event(self, event_data: Dict) -> Optional[AgentCoreUsage]:
        """Extract usage information from event metadata."""
        event_payload = event_data.get("event")
        if not event_payload:
            return None

        metadata = event_payload.get("metadata")
        if metadata and "usage" in metadata:
            return metadata["usage"]  # type: ignore

        return None

    def _extract_content_delta(self, event_data: Dict) -> Optional[str]:
        """Extract text content from contentBlockDelta event."""
        event_payload = event_data.get("event")
        if not event_payload:
            return None

        content_block_delta = event_payload.get("contentBlockDelta")
        if not content_block_delta:
            return None

        delta = content_block_delta.get("delta", {})
        return delta.get("text")

    def _extract_content_from_message(self, message: AgentCoreMessage) -> str:
        """
        Extract text content from message content blocks.
        This works for both SSE messages and JSON responses.
        """
        content_list = message.get("content", [])
        if not isinstance(content_list, list):
            return ""

        return "".join(
            block["text"]
            for block in content_list
            if isinstance(block, dict) and "text" in block
        )

    def _calculate_usage(
        self, model: str, messages: List[AllMessageValues], content: str
    ) -> Optional[Usage]:
        """
        Calculate token usage using LiteLLM's token counter.

        Args:
            model: The model name
            messages: Input messages
            content: Response content

        Returns:
            Usage object with calculated tokens, or None if calculation fails
        """
        try:
            from litellm.utils import token_counter

            prompt_tokens = token_counter(model=model, messages=messages)
            completion_tokens = token_counter(
                model=model, text=content, count_response_tokens=True
            )
            total_tokens = prompt_tokens + completion_tokens

            verbose_logger.debug(
                f"Calculated usage - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}"
            )

            return Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        except Exception as e:
            verbose_logger.warning(f"Failed to calculate token usage: {str(e)}")
            return None

    def _parse_json_response(self, response_json: dict) -> AgentCoreParsedResponse:
        """
        Parse direct JSON response (non-streaming).

        JSON response structure:
        {
            "result": {
                "role": "assistant",
                "content": [{"text": "..."}]
            }
        }
        """
        result = response_json.get("result", {})

        # Extract content using the same helper as SSE parsing
        content = self._extract_content_from_message(result)  # type: ignore

        # JSON responses don't include usage data
        return AgentCoreParsedResponse(
            content=content,
            usage=None,
            final_message=result,  # type: ignore
        )

    def _get_parsed_response(
        self, raw_response: httpx.Response
    ) -> AgentCoreParsedResponse:
        """
        Parse AgentCore response based on content type.

        Args:
            raw_response: Raw HTTP response from AgentCore

        Returns:
            AgentCoreParsedResponse: Parsed response data
        """
        content_type = raw_response.headers.get("content-type", "").lower()
        verbose_logger.debug(f"AgentCore response Content-Type: {content_type}")

        # Parse response based on content type
        if "application/json" in content_type:
            # Direct JSON response
            verbose_logger.debug("Parsing JSON response")
            response_json = raw_response.json()
            verbose_logger.debug(f"Response JSON: {response_json}")
            return self._parse_json_response(response_json)
        else:
            # SSE stream response (text/event-stream or default)
            verbose_logger.debug("Parsing SSE stream response")
            response_text = raw_response.text
            verbose_logger.debug(
                f"AgentCore response (first 500 chars): {response_text[:500]}"
            )
            return self._parse_sse_stream(response_text)

    def _parse_sse_stream(self, response_text: str) -> AgentCoreParsedResponse:
        """
        Parse Server-Sent Events (SSE) stream format.
        Each line starts with 'data:' followed by JSON.

        Returns:
            AgentCoreParsedResponse: Parsed response with content, usage, and message
        """
        final_message: Optional[AgentCoreMessage] = None
        usage_data: Optional[AgentCoreUsage] = None
        content_blocks: List[str] = []

        for line in response_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            data = self._extract_sse_json(line)
            if not data:
                continue

            verbose_logger.debug(f"SSE event keys: {list(data.keys())}")

            # Check for final complete message
            if "message" in data and isinstance(data["message"], dict):
                final_message = data["message"]  # type: ignore
                verbose_logger.debug("Found final message")

            # Process event data
            if "event" in data and isinstance(data["event"], dict):
                event_payload = data["event"]
                verbose_logger.debug(
                    f"Event payload keys: {list(event_payload.keys())}"
                )

                # Extract usage metadata
                if usage := self._extract_usage_from_event(data):
                    usage_data = usage
                    verbose_logger.debug(f"Found usage data: {usage_data}")

                # Collect content deltas
                if text := self._extract_content_delta(data):
                    content_blocks.append(text)

        # Build final content
        content = (
            self._extract_content_from_message(final_message)
            if final_message
            else "".join(content_blocks)
        )

        verbose_logger.debug(f"Final usage_data: {usage_data}")

        return AgentCoreParsedResponse(
            content=content, usage=usage_data, final_message=final_message
        )

    def _stream_agentcore_response_sync(
        self,
        response: httpx.Response,
        model: str,
    ):
        """
        Internal sync generator that parses SSE and yields ModelResponse chunks.
        """
        buffer = ""
        for text_chunk in response.iter_text():
            buffer += text_chunk

            # Process complete lines
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()

                if not line or not line.startswith('data:'):
                    continue

                json_str = line[5:].strip()
                if not json_str:
                    continue

                try:
                    data_obj = json.loads(json_str)
                    if not isinstance(data_obj, dict):
                        continue

                    # Process contentBlockDelta events
                    if "event" in data_obj and isinstance(data_obj["event"], dict):
                        event_payload = data_obj["event"]
                        content_block_delta = event_payload.get("contentBlockDelta")

                        if content_block_delta:
                            delta = content_block_delta.get("delta", {})
                            text = delta.get("text", "")

                            if text:
                                chunk = ModelResponse(
                                    id=f"chatcmpl-{uuid.uuid4()}",
                                    created=0,
                                    model=model,
                                    object="chat.completion.chunk",
                                )
                                chunk.choices = [
                                    StreamingChoices(
                                        finish_reason=None,
                                        index=0,
                                        delta=Delta(content=text, role="assistant"),
                                    )
                                ]
                                yield chunk

                        # Process metadata/usage
                        metadata = event_payload.get("metadata")
                        if metadata and "usage" in metadata:
                            chunk = ModelResponse(
                                id=f"chatcmpl-{uuid.uuid4()}",
                                created=0,
                                model=model,
                                object="chat.completion.chunk",
                            )
                            chunk.choices = [
                                StreamingChoices(
                                    finish_reason="stop",
                                    index=0,
                                    delta=Delta(),
                                )
                            ]
                            usage_data: AgentCoreUsage = metadata["usage"]  # type: ignore
                            setattr(chunk, "usage", Usage(
                                prompt_tokens=usage_data.get("inputTokens", 0),
                                completion_tokens=usage_data.get("outputTokens", 0),
                                total_tokens=usage_data.get("totalTokens", 0),
                            ))
                            yield chunk

                    # Process final message
                    if "message" in data_obj and isinstance(data_obj["message"], dict):
                        chunk = ModelResponse(
                            id=f"chatcmpl-{uuid.uuid4()}",
                            created=0,
                            model=model,
                            object="chat.completion.chunk",
                        )
                        chunk.choices = [
                            StreamingChoices(
                                finish_reason="stop",
                                index=0,
                                delta=Delta(),
                            )
                        ]
                        yield chunk

                except json.JSONDecodeError:
                    verbose_logger.debug(f"Skipping non-JSON SSE line: {line[:100]}")
                    continue

    def get_sync_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: Optional[Union[HTTPHandler, "AsyncHTTPHandler"]] = None,
        json_mode: Optional[bool] = None,
        signed_json_body: Optional[bytes] = None,
    ) -> "CustomStreamWrapper":
        """
        Simplified sync streaming - returns a generator that yields ModelResponse chunks.
        """
        from litellm.llms.custom_httpx.http_handler import (
            HTTPHandler,
            _get_httpx_client,
        )

        if client is None or not isinstance(client, HTTPHandler):
            client = _get_httpx_client(params={})

        verbose_logger.debug(f"Making sync streaming request to: {api_base}")

        # Make streaming request
        response = client.post(
            api_base,
            headers=headers,
            data=signed_json_body if signed_json_body else json.dumps(data),
            stream=True,
            logging_obj=logging_obj,
        )

        if response.status_code != 200:
            raise BedrockError(
                status_code=response.status_code, message=str(response.read())
            )

        # LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        # Wrap the generator in CustomStreamWrapper
        return CustomStreamWrapper(
            completion_stream=self._stream_agentcore_response_sync(response, model),
            model=model,
            custom_llm_provider="bedrock",
            logging_obj=logging_obj,
        )

    async def _stream_agentcore_response(
        self,
        response: httpx.Response,
        model: str,
    ) -> AsyncGenerator[ModelResponse, None]:
        """
        Internal async generator that parses SSE and yields ModelResponse chunks.
        """
        buffer = ""
        async for text_chunk in response.aiter_text():
            buffer += text_chunk

            # Process complete lines
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()

                if not line or not line.startswith('data:'):
                    continue

                json_str = line[5:].strip()
                if not json_str:
                    continue

                try:
                    data_obj = json.loads(json_str)
                    if not isinstance(data_obj, dict):
                        continue

                    # Process contentBlockDelta events
                    if "event" in data_obj and isinstance(data_obj["event"], dict):
                        event_payload = data_obj["event"]
                        content_block_delta = event_payload.get("contentBlockDelta")

                        if content_block_delta:
                            delta = content_block_delta.get("delta", {})
                            text = delta.get("text", "")

                            if text:
                                chunk = ModelResponse(
                                    id=f"chatcmpl-{uuid.uuid4()}",
                                    created=0,
                                    model=model,
                                    object="chat.completion.chunk",
                                )
                                chunk.choices = [
                                    StreamingChoices(
                                        finish_reason=None,
                                        index=0,
                                        delta=Delta(content=text, role="assistant"),
                                    )
                                ]
                                yield chunk

                        # Process metadata/usage
                        metadata = event_payload.get("metadata")
                        if metadata and "usage" in metadata:
                            chunk = ModelResponse(
                                id=f"chatcmpl-{uuid.uuid4()}",
                                created=0,
                                model=model,
                                object="chat.completion.chunk",
                            )
                            chunk.choices = [
                                StreamingChoices(
                                    finish_reason="stop",
                                    index=0,
                                    delta=Delta(),
                                )
                            ]
                            usage_data: AgentCoreUsage = metadata["usage"]  # type: ignore
                            setattr(chunk, "usage", Usage(
                                prompt_tokens=usage_data.get("inputTokens", 0),
                                completion_tokens=usage_data.get("outputTokens", 0),
                                total_tokens=usage_data.get("totalTokens", 0),
                            ))
                            yield chunk

                    # Process final message
                    if "message" in data_obj and isinstance(data_obj["message"], dict):
                        chunk = ModelResponse(
                            id=f"chatcmpl-{uuid.uuid4()}",
                            created=0,
                            model=model,
                            object="chat.completion.chunk",
                        )
                        chunk.choices = [
                            StreamingChoices(
                                finish_reason="stop",
                                index=0,
                                delta=Delta(),
                            )
                        ]
                        yield chunk

                except json.JSONDecodeError:
                    verbose_logger.debug(f"Skipping non-JSON SSE line: {line[:100]}")
                    continue

    async def get_async_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: Optional["AsyncHTTPHandler"] = None,
        json_mode: Optional[bool] = None,
        signed_json_body: Optional[bytes] = None,
    ) -> "CustomStreamWrapper":
        """
        Simplified async streaming - returns an async generator that yields ModelResponse chunks.
        """
        from litellm.llms.custom_httpx.http_handler import (
            AsyncHTTPHandler,
            get_async_httpx_client,
        )

        if client is None or not isinstance(client, AsyncHTTPHandler):
            client = get_async_httpx_client(
                llm_provider=cast(Any, "bedrock"), params={}
            )

        verbose_logger.debug(f"Making async streaming request to: {api_base}")

        # Make async streaming request
        response = await client.post(
            api_base,
            headers=headers,
            data=signed_json_body if signed_json_body else json.dumps(data),
            stream=True,
            logging_obj=logging_obj,
        )

        if response.status_code != 200:
            raise BedrockError(
                status_code=response.status_code, message=str(await response.aread())
            )

        # LOGGING
        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response="first stream response received",
            additional_args={"complete_input_dict": data},
        )

        # Wrap the async generator in CustomStreamWrapper
        return CustomStreamWrapper(
            completion_stream=self._stream_agentcore_response(response, model),
            model=model,
            custom_llm_provider="bedrock",
            logging_obj=logging_obj,
        )

    @property
    def has_custom_stream_wrapper(self) -> bool:
        """Indicates that this config has custom streaming support."""
        return True

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        """
        AgentCore does not allow passing `stream` in the request body.
        Streaming is automatic based on the response format.
        """
        return False

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform the AgentCore response to LiteLLM ModelResponse format.
        AgentCore can return either JSON or SSE (Server-Sent Events) stream responses.

        Note: For streaming responses, use get_streaming_response() instead.
        """
        try:
            # Parse the response based on content type (JSON or SSE)
            parsed_data = self._get_parsed_response(raw_response)

            content = parsed_data["content"]
            usage_data = parsed_data["usage"]

            verbose_logger.debug(f"Parsed content length: {len(content)}")
            verbose_logger.debug(f"Usage data: {usage_data}")

            # Create the message
            message = Message(content=content, role="assistant")

            # Create choices
            choice = Choices(finish_reason="stop", index=0, message=message)

            # Update model response
            model_response.choices = [choice]
            model_response.model = model

            # Add usage information if available
            # Note: AgentCore JSON responses don't include usage data
            # SSE responses may include usage in metadata events
            if usage_data:
                usage = Usage(
                    prompt_tokens=usage_data.get("inputTokens", 0),
                    completion_tokens=usage_data.get("outputTokens", 0),
                    total_tokens=usage_data.get("totalTokens", 0),
                )
                setattr(model_response, "usage", usage)
            else:
                # Calculate token usage using LiteLLM's token counter
                verbose_logger.debug(
                    "No usage data from AgentCore - calculating tokens"
                )
                calculated_usage = self._calculate_usage(model, messages, content)
                if calculated_usage:
                    setattr(model_response, "usage", calculated_usage)

            return model_response

        except Exception as e:
            verbose_logger.error(
                f"Error processing Bedrock AgentCore response: {str(e)}"
            )
            raise BedrockError(
                message=f"Error processing response: {str(e)}",
                status_code=raw_response.status_code,
            )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message)

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        # AgentCore supports true streaming - don't buffer
        return False
