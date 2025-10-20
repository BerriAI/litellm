"""
AWS Bedrock AgentCore Runtime Provider for LiteLLM

This module implements support for AWS Bedrock AgentCore Runtime API,
enabling AI agents to be invoked through LiteLLM's unified interface.

AgentCore provides serverless deployment, auto-scaling, and managed runtime
for AI agents built with frameworks like Strands, LangGraph, and CrewAI.

Model Formats:
    1. Simple agent name:
       model="bedrock/agentcore/my-agent"
       Requires: aws_region_name

    2. Full ARN:
       model="bedrock/agentcore/arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent"

    3. With qualifier (version/endpoint):
       model="bedrock/agentcore/my-agent"
       qualifier="1.0" or qualifier="production"

    4. With session continuity:
       model="bedrock/agentcore/my-agent"
       runtime_session_id="my-session-123..."

Multi-Modal Support:
    AgentCore Runtime accepts flexible JSON payloads up to 100MB with any structure.
    Actual content type support depends on your agent's foundation model:

    - Images (JPEG, PNG, GIF, WebP): ✅ Confirmed for Claude models
    - Video/Audio/Documents: ⚠️  Model-dependent (check your model's capabilities)

    AgentCore doesn't enforce a strict payload schema. This implementation supports
    all content types using LiteLLM's utilities, but your agent's model must be
    able to process the content you send.

Examples:
    # Basic text-only usage
    response = litellm.completion(
        model="bedrock/agentcore/my-agent",
        messages=[{"role": "user", "content": "Hello"}],
        aws_region_name="us-west-2"
    )

    # Multi-modal: Single image with text (✅ Confirmed for Claude models)
    import base64
    with open("image.jpg", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    response = litellm.completion(
        model="bedrock/agentcore/vision-agent",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }
            ]
        }],
        aws_region_name="us-west-2"
    )

    # Multi-modal: Multiple images
    response = litellm.completion(
        model="bedrock/agentcore/vision-agent",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Compare these images:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }],
        aws_region_name="us-west-2"
    )

    # Multi-modal: Video content (⚠️ Model-dependent - verify your model supports video)
    with open("video.mp4", "rb") as f:
        video_data = base64.b64encode(f.read()).decode('utf-8')

    response = litellm.completion(
        model="bedrock/agentcore/video-agent",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this video:"},
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_data}"}
                }
            ]
        }],
        aws_region_name="us-west-2"
    )

    # Multi-modal: Audio content (⚠️ Model-dependent - verify your model supports audio)
    with open("audio.mp3", "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')

    response = litellm.completion(
        model="bedrock/agentcore/audio-agent",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio:"},
                {
                    "type": "audio",
                    "input_audio": {"data": audio_data, "format": "mp3"}
                }
            ]
        }],
        aws_region_name="us-west-2"
    )

    # Multi-modal: Document content (⚠️ Model-dependent - verify your model supports documents)
    # Note: For PDFs with Claude models, consider converting to images first
    with open("document.pdf", "rb") as f:
        doc_data = base64.b64encode(f.read()).decode('utf-8')

    response = litellm.completion(
        model="bedrock/agentcore/doc-agent",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this document:"},
                {
                    "type": "document",
                    "source": {"type": "text", "media_type": "application/pdf", "data": doc_data}
                }
            ]
        }],
        aws_region_name="us-west-2"
    )

    # With qualifier (version/endpoint)
    response = litellm.completion(
        model="bedrock/agentcore/my-agent",
        messages=[{"role": "user", "content": "Hello"}],
        aws_region_name="us-west-2",
        qualifier="production"
    )

    # With session continuity
    response = litellm.completion(
        model="bedrock/agentcore/my-agent",
        messages=[{"role": "user", "content": "Hello"}],
        aws_region_name="us-west-2",
        runtime_session_id="my-session-123..."
    )

    # Streaming with SSE
    response = litellm.completion(
        model="bedrock/agentcore/my-agent",
        messages=[{"role": "user", "content": "Hello"}],
        aws_region_name="us-west-2",
        stream=True
    )
    for chunk in response:
        print(chunk.choices[0].delta.content)

    # Streaming with multi-modal input
    response = litellm.completion(
        model="bedrock/agentcore/vision-agent",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this:"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }],
        aws_region_name="us-west-2",
        stream=True
    )
    for chunk in response:
        print(chunk.choices[0].delta.content)
"""

import json
import os
import time
import uuid
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    NoReturn,
)

import boto3
import litellm
from botocore.exceptions import ClientError, NoCredentialsError
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.types.llms.bedrock_agentcore import (
    AgentCoreResponseUnion,
    AgentCoreRequestPayload,
    AgentCoreInvokeParams,
)
from litellm.types.utils import ModelResponse, StreamingChoices, Usage
from litellm.utils import CustomStreamWrapper, token_counter


# Note: Using BedrockError for consistency with LiteLLM's Bedrock ecosystem
# AgentCore is part of AWS Bedrock services, so we use the same error class


class AgentCoreConfig(BaseAWSLLM):
    """
    Configuration and implementation for AWS Bedrock AgentCore Runtime.

    Uses standard boto3 client for authentication and API calls.
    Handles transformation between LiteLLM's message format and AgentCore's
    prompt/context structure.

    Attributes:
        service_name: The AWS service name for AgentCore
    """

    def __init__(self):
        super().__init__()
        self.service_name = "bedrock-agentcore"
        # STS account ID cache to avoid repeated calls (50-200ms latency reduction)
        self._account_id_cache: Dict[str, str] = {}
        self._cache_ttl = 3600  # 1 hour TTL
        self._cache_timestamps: Dict[str, float] = {}

    def _parse_model(self, model: str) -> Dict[str, Any]:
        """
        Parse AgentCore model string.

        Note: LiteLLM's get_llm_provider already strips the "agentcore/" prefix,
        so this method receives either:
        - "agent-name" (simple name, requires aws_region_name)
        - "agent-name/qualifier" (simple name with version/endpoint, requires aws_region_name)
        - "arn:aws:bedrock-agentcore:region:account:runtime/agent" (full ARN)
        - "arn:aws:bedrock-agentcore:region:account:runtime/agent/qualifier" (full ARN with qualifier)

        Args:
            model: Model string to parse (without "agentcore/" prefix)

        Returns:
            Dict with 'arn', 'agent_name', 'region', and 'qualifier' keys

        Raises:
            ValueError: If model format is invalid
        """
        if model.startswith("arn:aws:"):
            # Full ARN provided - validate it's bedrock-agentcore
            if not model.startswith("arn:aws:bedrock-agentcore:"):
                raise ValueError(f"Invalid AgentCore ARN format: '{model}'")

            parts = model.split(":")
            if len(parts) < 6:
                raise ValueError(f"Invalid AgentCore ARN format: '{model}'")

            # Check if there's a qualifier after the agent name
            # Format: arn:aws:bedrock-agentcore:region:account:runtime/agent-name OR
            #         arn:aws:bedrock-agentcore:region:account:runtime/agent-name/qualifier
            runtime_part = parts[
                5
            ]  # "runtime/agent-name" or "runtime/agent-name/qualifier"
            runtime_segments = runtime_part.split("/")

            if len(runtime_segments) == 2:
                # No qualifier: runtime/agent-name
                agent_name = runtime_segments[1]
                qualifier = None
            elif len(runtime_segments) == 3:
                # With qualifier: runtime/agent-name/qualifier
                agent_name = runtime_segments[1]
                qualifier = runtime_segments[2]
            else:
                raise ValueError(f"Invalid AgentCore ARN format: '{model}'")

            # Build ARN without qualifier
            arn_without_qualifier = (
                f"arn:aws:bedrock-agentcore:{parts[3]}:{parts[4]}:runtime/{agent_name}"
            )

            return {
                "arn": arn_without_qualifier,
                "agent_name": agent_name,
                "region": parts[3],
                "qualifier": qualifier,
            }
        else:
            # Simple agent name, possibly with qualifier
            # Format: "agent-name" or "agent-name/qualifier"
            parts = model.split("/")

            if len(parts) == 1:
                # No qualifier
                return {
                    "arn": None,
                    "agent_name": parts[0],
                    "region": None,
                    "qualifier": None,
                }
            elif len(parts) == 2:
                # With qualifier
                return {
                    "arn": None,
                    "agent_name": parts[0],
                    "region": None,
                    "qualifier": parts[1],
                }
            else:
                raise ValueError(f"Invalid AgentCore model format: '{model}'")

    def _get_account_id(self, region: str) -> str:
        """
        Get AWS account ID with caching to avoid repeated STS calls.

        This reduces latency by 50-200ms per request after the first call.
        Cache has 1 hour TTL to handle credential rotation scenarios.

        Args:
            region: AWS region

        Returns:
            AWS account ID

        Raises:
            NoCredentialsError: If AWS credentials not configured
            ClientError: If STS call fails
        """
        cache_key = f"account_id_{region}"
        current_time = time.time()

        # Check cache
        if cache_key in self._account_id_cache:
            cached_time = self._cache_timestamps.get(cache_key, 0)
            if current_time - cached_time < self._cache_ttl:
                litellm.verbose_logger.debug(
                    f"Using cached account ID for region {region}"
                )
                return self._account_id_cache[cache_key]

        # Fetch from STS
        try:
            sts = boto3.client("sts", region_name=region)
            account_id = sts.get_caller_identity()["Account"]

            # Cache result
            self._account_id_cache[cache_key] = account_id
            self._cache_timestamps[cache_key] = current_time

            return account_id

        except NoCredentialsError as e:
            raise BedrockError(
                status_code=401,
                message=(
                    f"AWS credentials not configured for AgentCore. Configure using:\n"
                    f"1) Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)\n"
                    f"2) AWS profile (set aws_profile_name parameter)\n"
                    f"3) IAM role (for EC2/ECS/Lambda execution)\n"
                    f"Error: {e}"
                ),
            ) from e
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            http_status = e.response.get("ResponseMetadata", {}).get(
                "HTTPStatusCode", 500
            )
            raise BedrockError(
                status_code=http_status,
                message=f"AgentCore STS call failed ({error_code}): {error_message}. Check AWS credentials and permissions.",
            ) from e

    def _build_agent_arn(
        self, agent_name: str, region: str, client: Optional[boto3.client] = None
    ) -> str:
        """
        Build the agent runtime ARN from agent name and region.

        Uses cached account ID to avoid repeated STS calls.

        Args:
            agent_name: The agent identifier
            region: AWS region
            client: Optional boto3 client (not used, kept for compatibility)

        Returns:
            Agent runtime ARN
        """
        # AgentCore ARN format: arn:aws:bedrock-agentcore:region:account:runtime/agent-name
        try:
            account_id = self._get_account_id(region)
        except Exception:
            # Fall back to wildcard if STS call fails
            account_id = "*"
        return f"arn:aws:bedrock-agentcore:{region}:{account_id}:runtime/{agent_name}"

    def _create_agentcore_client(self, region: str, **optional_params) -> boto3.client:
        """
        Create AgentCore boto3 client with proper credentials.

        Uses BaseAWSLLM.get_credentials() for comprehensive credential management:
        - Environment variables
        - AWS profiles
        - IAM roles
        - Web identity tokens
        - STS assume role
        - Secret managers

        Args:
            region: AWS region
            **optional_params: AWS credential parameters

        Returns:
            boto3 AgentCore client
        """
        try:
            # Use BaseAWSLLM's comprehensive credential management
            credentials = self.get_credentials(
                aws_access_key_id=optional_params.get("aws_access_key_id"),
                aws_secret_access_key=optional_params.get("aws_secret_access_key"),
                aws_session_token=optional_params.get("aws_session_token"),
                aws_region_name=region,
                aws_session_name=optional_params.get("aws_session_name"),
                aws_profile_name=optional_params.get("aws_profile_name"),
                aws_role_name=optional_params.get("aws_role_name"),
                aws_web_identity_token=optional_params.get("aws_web_identity_token"),
                aws_sts_endpoint=optional_params.get("aws_sts_endpoint"),
            )

            # Create boto3 client with resolved credentials
            client = boto3.client(
                "bedrock-agentcore",
                region_name=region,
                aws_access_key_id=credentials.access_key,
                aws_secret_access_key=credentials.secret_key,
                aws_session_token=credentials.token,
            )

            return client

        except Exception as e:
            litellm.verbose_logger.error(
                f"Failed to create AgentCore client with credentials: {e}"
            )
            # Fallback to default credential chain if BaseAWSLLM credentials fail
            try:
                client = boto3.client("bedrock-agentcore", region_name=region)
                litellm.verbose_logger.info(
                    "Using default AWS credential chain for AgentCore"
                )
                return client
            except Exception as fallback_error:
                raise BedrockError(
                    status_code=401,
                    message=f"AgentCore: Failed to create client with both explicit credentials and default chain: {e} | {fallback_error}",
                )

    def _process_image_element(
        self, element: Dict[str, Any], media_items: List[Dict[str, Any]]
    ) -> None:
        """Process image_url element and append to media_items."""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            convert_to_anthropic_image_obj,
        )

        image_url_data = element.get("image_url", {})
        url = (
            image_url_data.get("url", "")
            if isinstance(image_url_data, dict)
            else image_url_data
        )
        format_override = (
            image_url_data.get("format")
            if isinstance(image_url_data, dict)
            else None
        )

        if not url:
            return

        try:
            parsed = convert_to_anthropic_image_obj(url, format=format_override)
            media_format = (
                parsed["media_type"].split("/")[-1]
                if "/" in parsed["media_type"]
                else "jpeg"
            )
            media_items.append(
                {"type": "image", "format": media_format, "data": parsed["data"]}
            )
        except ValueError as e:
            litellm.verbose_logger.error(
                f"Invalid image format at index {len(media_items)}: {e}. "
                f"URL: {url[:100]}{'...' if len(url) > 100 else ''}"
            )
        except Exception as e:
            litellm.verbose_logger.error(
                f"Unexpected error parsing image at index {len(media_items)}: "
                f"{type(e).__name__}: {e}"
            )

    def _process_video_element(
        self, element: Dict[str, Any], media_items: List[Dict[str, Any]]
    ) -> None:
        """Process video_url element and append to media_items."""
        from litellm.litellm_core_utils.prompt_templates.factory import (
            convert_to_anthropic_image_obj,
        )

        video_url_data = element.get("video_url", {})
        url = (
            video_url_data.get("url", "")
            if isinstance(video_url_data, dict)
            else video_url_data
        )
        format_override = (
            video_url_data.get("format")
            if isinstance(video_url_data, dict)
            else None
        )

        if not url:
            return

        try:
            parsed = convert_to_anthropic_image_obj(url, format=format_override)
            media_format = (
                parsed["media_type"].split("/")[-1]
                if "/" in parsed["media_type"]
                else "mp4"
            )
            media_items.append(
                {"type": "video", "format": media_format, "data": parsed["data"]}
            )
        except Exception as e:
            litellm.verbose_logger.error(
                f"Invalid video format: {e}. "
                f"URL: {url[:100]}{'...' if len(url) > 100 else ''}"
            )

    def _process_audio_element(
        self, element: Dict[str, Any], media_items: List[Dict[str, Any]]
    ) -> None:
        """Process audio element and append to media_items."""
        input_audio = element.get("input_audio", {})

        if not isinstance(input_audio, dict):
            litellm.verbose_logger.error(
                f"Unexpected audio format: {element}. Skipping audio."
            )
            return

        audio_data = input_audio.get("data", "")
        audio_format = input_audio.get("format", "mp3")

        if audio_data:
            media_items.append(
                {"type": "audio", "format": audio_format, "data": audio_data}
            )

    def _process_document_element(
        self, element: Dict[str, Any], media_items: List[Dict[str, Any]]
    ) -> None:
        """Process document element and append to media_items."""
        source = element.get("source", {})

        if not isinstance(source, dict):
            litellm.verbose_logger.error(
                f"Unexpected document format: {element}. Skipping document."
            )
            return

        doc_data = source.get("data", "")
        doc_media_type = source.get("media_type", "application/pdf")
        doc_format = (
            doc_media_type.split("/")[-1] if "/" in doc_media_type else "pdf"
        )

        if doc_data:
            media_items.append(
                {"type": "document", "format": doc_format, "data": doc_data}
            )

    def _extract_text_and_media_from_content(
        self, content: Union[str, List[Dict[str, Any]]]
    ) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
        """
        Extract text prompt and media from LiteLLM message content.

        Supports multi-modal content including images, videos, audio, and documents.
        Uses LiteLLM's content processing utilities to properly parse media.

        AgentCore Runtime accepts flexible JSON payloads (up to 100MB) with any structure.
        Actual content type support depends on your agent's foundation model:
        - Images (JPEG, PNG, GIF, WebP): ✅ Confirmed for Claude models
        - Video/Audio/Documents: ⚠️  Model-dependent (verify your model's capabilities)

        Args:
            content: Either a string or list of content parts (text + media)

        Returns:
            Tuple of (text_prompt, media_list) where media_list is None if no media

        Supported Content Types (implementation):
            - text: Plain text content
            - image_url: Images (png, jpeg, gif, webp) - ✅ Works with Claude models
            - video_url: Videos (mp4, mov, mkv, webm, etc.) - ⚠️  Model-dependent
            - audio: Audio files - ⚠️  Model-dependent
            - document: Documents (pdf, doc, txt, etc.) - ⚠️  Model-dependent

        Note:
            For PDFs with Claude models, consider converting to images first.
            The implementation supports all types, but your agent's model must support them.
        """
        # Simple text-only content
        if isinstance(content, str):
            return content, None

        # Multi-modal content with array of parts
        if isinstance(content, list):
            text_parts = []
            media_items = []

            for element in content:
                if not isinstance(element, dict):
                    continue

                element_type = element.get("type", "")

                if element_type == "text":
                    text_parts.append(element.get("text", ""))
                elif element_type == "image_url":
                    self._process_image_element(element, media_items)
                elif element_type == "video_url":
                    self._process_video_element(element, media_items)
                elif element_type == "audio":
                    self._process_audio_element(element, media_items)
                elif element_type == "document":
                    self._process_document_element(element, media_items)

            # Combine text parts
            text_prompt = " ".join(text_parts) if text_parts else ""

            # Return media only if we found any
            return text_prompt, media_items if media_items else None

        # Fallback for unexpected content type
        return str(content), None

    def _transform_messages_to_agentcore(
        self, messages: List[Dict[str, Any]], session_id: Optional[str] = None
    ) -> AgentCoreRequestPayload:
        """
        Transform LiteLLM messages to AgentCore request format.

        AgentCore expects:
        - prompt: The latest user message (text)
        - media: Multi-modal content (optional, for images)
        - context: Conversation history (optional)
        - runtimeSessionId: Session ID (required, min 33 chars)

        Supports both text-only and multi-modal (text + images) requests.

        Args:
            messages: List of message dicts with 'role' and 'content'
            session_id: Runtime session ID (auto-generated if not provided)

        Returns:
            Dict with 'prompt', optionally 'media', 'context', and 'runtimeSessionId'
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")

        # Last message should be from user
        last_message = messages[-1]
        if last_message.get("role") != "user":
            raise ValueError("Last message must be from user")

        # Extract text and media from last message content
        content = last_message.get("content", "")
        prompt, media_items = self._extract_text_and_media_from_content(content)

        # Generate session ID if not provided
        # AgentCore requires session IDs >= 33 characters for uniqueness guarantees
        # UUID4 format: 8-4-4-4-12 = 36 chars (with hyphens), exceeds requirement
        if not session_id:
            session_id = str(uuid.uuid4())

        # Build request data
        request_data = {"prompt": prompt, "runtimeSessionId": session_id}

        # Add media if present (multi-modal request)
        if media_items:
            # AgentCore supports single media item or list
            if len(media_items) == 1:
                request_data["media"] = media_items[0]
            else:
                # Multiple images - use array format
                request_data["media"] = media_items

        # Build context from conversation history (all messages except last)
        if len(messages) > 1:
            # Convert message history to context string
            context_messages = []
            for msg in messages[:-1]:
                role = msg.get("role", "")
                content = msg.get("content", "")

                # For context, extract only text (no media in context)
                if isinstance(content, list):
                    text, _ = self._extract_text_and_media_from_content(content)
                    content = text

                context_messages.append(f"{role}: {content}")

            request_data["context"] = "\n".join(context_messages)

        return request_data

    def _transform_agentcore_to_litellm(
        self,
        agentcore_response: AgentCoreResponseUnion,
        model: str,
        created_at: int,
        session_id: Optional[str] = None,
        custom_llm_provider: str = "bedrock",
        prompt_text: Optional[str] = None,
    ) -> ModelResponse:
        """
        Transform AgentCore response to LiteLLM ModelResponse.

        Args:
            agentcore_response: Response from AgentCore API
            model: Original model string
            created_at: Unix timestamp of request
            session_id: Runtime session ID for continuity
            custom_llm_provider: Provider name
            prompt_text: Original prompt text for accurate token counting

        Returns:
            LiteLLM ModelResponse object
        """
        # Handle both string and dictionary responses from AgentCore
        # - String response: Agent using BedrockAgentCoreApp returns plain string
        # - Dictionary response: Legacy format with {"response": "...", "metadata": {...}}
        if isinstance(agentcore_response, str):
            response_text = agentcore_response
            metadata = {}
        else:
            response_text = agentcore_response.get("response", "")
            metadata = agentcore_response.get("metadata", {})

        # Calculate token usage
        # Note: AgentCore may provide actual token counts in metadata
        prompt_tokens = metadata.get("prompt_tokens", 0)
        completion_tokens = metadata.get("completion_tokens", 0)

        # Fallback to estimation if not provided
        if prompt_tokens == 0 or completion_tokens == 0:
            try:
                # Use LiteLLM's token counter as fallback
                # Use actual prompt text if available, otherwise estimate
                if prompt_text and prompt_tokens == 0:
                    prompt_tokens = token_counter(
                        model=model, messages=[{"role": "user", "content": prompt_text}]
                    )
                else:
                    prompt_tokens = prompt_tokens or 10

                if completion_tokens == 0:
                    completion_tokens = token_counter(model=model, text=response_text)
            except Exception as e:
                # If token counting fails, use rough estimates based on word count
                litellm.verbose_logger.warning(
                    f"Token counting failed: {e}. Using rough estimates."
                )
                prompt_tokens = prompt_tokens or (
                    len(prompt_text.split()) if prompt_text else 10
                )
                completion_tokens = completion_tokens or len(response_text.split()) * 2

        model_response = ModelResponse(
            id=f"agentcore-{int(time.time())}",
            choices=[
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                }
            ],
            created=created_at,
            model=model,
            object="chat.completion",
            system_fingerprint=None,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

        # Add AgentCore metadata to response, including session ID
        model_response._hidden_params = {
            "custom_llm_provider": custom_llm_provider,
            "runtime_session_id": session_id,
            "agentcore_metadata": metadata,
        }

        return model_response

    def _parse_streaming_chunk(
        self, chunk: str, model: str, created_at: int
    ) -> Optional[ModelResponse]:
        """
        Parse Server-Sent Events (SSE) chunk from AgentCore streaming.

        Args:
            chunk: SSE formatted string (e.g., "data: {...}")
            model: Model identifier
            created_at: Unix timestamp

        Returns:
            ModelResponse object or None if chunk is not parseable
        """
        # SSE format: "data: {...}"
        if not chunk.strip():
            return None

        if chunk.startswith("data: "):
            json_str = chunk[6:].strip()

            # Handle SSE keep-alive or end markers
            if json_str in ["", "[DONE]"]:
                return None

            try:
                data = json.loads(json_str)

                # Extract token or response text
                token = data.get("token", "")
                if not token:
                    # Some implementations might use 'response' or 'text'
                    token = data.get("response", data.get("text", ""))

                if not token:
                    return None

                # Create streaming response chunk
                return ModelResponse(
                    id=f"agentcore-{created_at}",
                    choices=[
                        StreamingChoices(
                            finish_reason=data.get("finish_reason"),
                            index=0,
                            delta={"role": "assistant", "content": token},
                        )
                    ],
                    created=created_at,
                    model=model,
                    object="chat.completion.chunk",
                    system_fingerprint=None,
                )
            except json.JSONDecodeError:
                # Log but don't fail on malformed chunks
                litellm.print_verbose(f"Failed to parse SSE chunk: {chunk}")
                return None

        return None

    def _resolve_aws_region(
        self, model_region: Optional[str], **kwargs
    ) -> str:
        """
        Resolve AWS region from model ARN or kwargs/environment.

        Args:
            model_region: Region extracted from ARN (if provided)
            **kwargs: Keyword arguments that may contain aws_region or aws_region_name

        Returns:
            AWS region string

        Raises:
            BedrockError: If region cannot be determined
        """
        if model_region:
            return model_region

        aws_region = (
            kwargs.get("aws_region")
            or kwargs.get("aws_region_name")
            or os.getenv("AWS_REGION")
        )

        if not aws_region:
            raise BedrockError(
                status_code=400,
                message="AgentCore: aws_region_name is required when not using full ARN. Provide via aws_region_name parameter or AWS_REGION environment variable.",
            )

        return aws_region

    def _resolve_agent_arn(
        self,
        provided_arn: Optional[str],
        api_base: str,
        agent_name: str,
        aws_region: str,
        client: boto3.client,
    ) -> str:
        """
        Resolve agent ARN from provided sources or construct from agent name.

        Args:
            provided_arn: ARN from model string (if provided)
            api_base: API base parameter (may contain ARN)
            agent_name: Agent identifier
            aws_region: AWS region
            client: Boto3 client

        Returns:
            Agent runtime ARN
        """
        if provided_arn:
            return provided_arn

        if api_base and api_base.startswith("arn:aws:bedrock-agentcore:"):
            return api_base

        return self._build_agent_arn(agent_name, aws_region, client)

    def completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        api_base: str,
        model_response: ModelResponse,
        print_verbose: callable,
        encoding: Any,
        logging_obj: Any,
        optional_params: Dict[str, Any],
        timeout: Optional[Union[float, int]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
        acompletion: bool = False,
        stream: bool = False,
        **kwargs,
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """
        Synchronous completion for AgentCore.

        Args:
            model: Format "agentcore/agent-name" or "agentcore/arn:aws:bedrock-agentcore:..."
            messages: List of conversation messages
            api_base: AgentCore Runtime API endpoint (can be agent ARN)
            model_response: ModelResponse object to populate
            print_verbose: Logging function
            encoding: Tokenizer encoding
            logging_obj: Logging object
            optional_params: Additional parameters (qualifier, runtime_session_id, etc.)
            timeout: Request timeout
            litellm_params: LiteLLM specific parameters
            acompletion: Whether this is async (should be False)
            stream: Whether to stream response

        Returns:
            ModelResponse or CustomStreamWrapper for streaming
        """
        # Parse model string and extract parameters
        model_info = self._parse_model(model)
        agent_name = model_info["agent_name"]
        provided_arn = model_info["arn"]
        model_region = model_info["region"]

        qualifier = optional_params.pop("qualifier", None) or model_info.get(
            "qualifier"
        )
        runtime_session_id = optional_params.pop("runtime_session_id", None)

        # Resolve AWS region and create client
        aws_region = self._resolve_aws_region(model_region, **kwargs)

        try:
            client = self._create_agentcore_client(region=aws_region, **kwargs)
        except BedrockError:
            raise
        except Exception as e:
            litellm.verbose_logger.error(f"Failed to create AgentCore client: {e}")
            raise BedrockError(
                status_code=500, message=f"AgentCore: AWS client creation failed: {e}"
            ) from e

        # Resolve agent ARN and build request
        agent_arn = self._resolve_agent_arn(
            provided_arn, api_base, agent_name, aws_region, client
        )

        request_data = self._transform_messages_to_agentcore(
            messages, session_id=runtime_session_id
        )
        response_session_id = request_data.get("runtimeSessionId")
        request_data.update(optional_params)

        # Execute request
        created_at = int(time.time())

        if stream:
            return self._handle_streaming(
                client=client,
                agent_arn=agent_arn,
                qualifier=qualifier,
                data=request_data,
                model=model,
                created_at=created_at,
                session_id=response_session_id,
                timeout=timeout,
            )
        else:
            return self._handle_completion(
                client=client,
                agent_arn=agent_arn,
                qualifier=qualifier,
                data=request_data,
                model=model,
                created_at=created_at,
                session_id=response_session_id,
                timeout=timeout,
            )

    def _build_invoke_params(
        self, agent_arn: str, qualifier: Optional[str], data: Dict[str, Any]
    ) -> Tuple[AgentCoreInvokeParams, Optional[str]]:
        """
        Build invoke parameters for AgentCore Runtime API.

        Extracts runtimeSessionId from data and constructs boto3 invoke parameters.
        This avoids code duplication between streaming and non-streaming invocations.

        Args:
            agent_arn: Agent runtime ARN
            qualifier: Version/endpoint qualifier
            data: Request payload data

        Returns:
            Tuple of (invoke_params dict, runtime_session_id)
        """
        # CRITICAL FIX: runtimeSessionId must be a boto3 parameter, NOT in the JSON payload
        # Extract runtimeSessionId from data before encoding payload
        runtime_session_id = data.pop("runtimeSessionId", None)

        # Build invoke params
        # IMPORTANT: Match official AWS samples - payload as JSON string, not bytes
        # Official samples don't use contentType or accept headers
        invoke_params = {
            "agentRuntimeArn": agent_arn,
            "payload": json.dumps(
                data
            ),  # JSON string, not bytes (matches official samples)
        }

        # Add runtimeSessionId as separate boto3 parameter (not in payload)
        if runtime_session_id:
            invoke_params["runtimeSessionId"] = runtime_session_id

        # Add qualifier only if provided (no default)
        if qualifier:
            invoke_params["qualifier"] = qualifier

        return invoke_params, runtime_session_id

    def _handle_completion(
        self,
        client: boto3.client,
        agent_arn: str,
        qualifier: Optional[str],
        data: Dict[str, Any],
        model: str,
        created_at: int,
        session_id: Optional[str],
        timeout: Optional[Union[float, int]],
    ) -> ModelResponse:
        """Handle non-streaming completion request using boto3 with retry logic for cold starts."""
        # Build invoke parameters using shared method
        invoke_params, runtime_session_id = self._build_invoke_params(
            agent_arn, qualifier, data
        )

        # Retry logic for RuntimeClientError (cold start after 15min inactivity)
        # AgentCore containers scale to zero after 15 minutes of inactivity
        # Cold starts can take 30-60 seconds for ARM64 containers
        max_retries = 6
        retry_delays = [
            10,
            15,
            20,
            25,
            30,
            40,
        ]  # Exponential backoff: 10-15-20-25-30-40s (total: 140s)

        for attempt in range(max_retries):
            try:
                response = client.invoke_agent_runtime(**invoke_params)

                # Validate response structure
                if not response:
                    raise BedrockError(
                        status_code=500, message="AgentCore returned empty response"
                    )

                if "ResponseMetadata" not in response:
                    raise BedrockError(
                        status_code=500,
                        message="AgentCore response missing ResponseMetadata",
                    )

                http_status = response["ResponseMetadata"].get("HTTPStatusCode")
                if http_status != 200:
                    raise BedrockError(
                        status_code=http_status,
                        message=f"AgentCore returned HTTP {http_status}",
                    )

                # Get session ID from response if available
                response_session_id = response.get("runtimeSessionId", session_id)

                # Read response payload
                if "response" in response:
                    # AgentCore returns 'response' key with StreamingBody
                    payload_data = response["response"]
                    # Handle streaming response body
                    if hasattr(payload_data, "read"):
                        response_text = payload_data.read().decode("utf-8")
                    else:
                        response_text = str(payload_data)

                    try:
                        agentcore_response = json.loads(response_text)
                    except json.JSONDecodeError:
                        # If response is not JSON, treat as plain text
                        agentcore_response = {"response": response_text}
                else:
                    agentcore_response = {"response": ""}

                return self._transform_agentcore_to_litellm(
                    agentcore_response=agentcore_response,
                    model=model,
                    created_at=created_at,
                    session_id=response_session_id,
                    prompt_text=data.get("prompt", ""),
                )

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                error_message = e.response.get("Error", {}).get("Message", str(e))

                # Retry only RuntimeClientError (cold start)
                if error_code == "RuntimeClientError" and attempt < max_retries - 1:
                    retry_delay = retry_delays[attempt]
                    litellm.print_verbose(
                        f"RuntimeClientError on attempt {attempt + 1}/{max_retries}. "
                        f"Runtime container cold starting (ARM64 takes 20-30s). Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    # No more retries or different error - raise it
                    self._handle_boto3_error(error_code, error_message)
            except Exception as e:
                raise BedrockError(
                    status_code=500, message=f"AgentCore: API request failed: {str(e)}"
                ) from e

        # Should not reach here, but just in case
        raise BedrockError(
            status_code=500,
            message="AgentCore: API request failed after all retries (cold start timeout)",
        )

    def _handle_streaming(
        self,
        client: boto3.client,
        agent_arn: str,
        qualifier: Optional[str],
        data: Dict[str, Any],
        model: str,
        created_at: int,
        session_id: Optional[str],
        timeout: Optional[Union[float, int]],
    ) -> CustomStreamWrapper:
        """Handle streaming completion request with proper SSE parsing."""
        # Variable to store the actual session ID from response
        actual_session_id = session_id

        def stream_generator() -> Iterator[ModelResponse]:
            nonlocal actual_session_id  # Allow updating from generator

            try:
                # Build invoke parameters using shared method
                invoke_params, runtime_session_id = self._build_invoke_params(
                    agent_arn, qualifier, data
                )

                response = client.invoke_agent_runtime(**invoke_params)

                # Get session ID from response if available and update nonlocal
                actual_session_id = response.get("runtimeSessionId", session_id)

                # AgentCore returns StreamingBody in 'response' key for SSE streaming
                stream_body = response.get("response")
                if not stream_body:
                    return

                # Parse SSE stream line by line
                for line in stream_body.iter_lines():
                    if line:
                        decoded = line.decode("utf-8").strip()

                        # Parse SSE format: "data: {...}"
                        if decoded.startswith("data: "):
                            json_str = decoded[6:]  # Remove "data: " prefix

                            # Handle SSE end marker
                            if json_str == "[DONE]":
                                break

                            try:
                                data_chunk = json.loads(json_str)
                                token = data_chunk.get("token", "")
                                finish_reason = data_chunk.get("finish_reason")

                                # Yield chunk only if it has token content or finish_reason
                                # Skip empty chunks without finish_reason
                                if token or finish_reason:
                                    chunk = ModelResponse(
                                        id=f"agentcore-{created_at}",
                                        choices=[
                                            StreamingChoices(
                                                finish_reason=finish_reason,
                                                index=0,
                                                delta={
                                                    "role": "assistant",
                                                    "content": token,
                                                },
                                            )
                                        ],
                                        created=created_at,
                                        model=model,
                                        object="chat.completion.chunk",
                                        system_fingerprint=None,
                                    )

                                    # Initialize _hidden_params if it doesn't exist
                                    if not hasattr(chunk, "_hidden_params"):
                                        chunk._hidden_params = {}

                                    # Add session ID to hidden params for session continuity
                                    chunk._hidden_params[
                                        "custom_llm_provider"
                                    ] = "bedrock"
                                    chunk._hidden_params[
                                        "runtime_session_id"
                                    ] = actual_session_id

                                    yield chunk

                            except json.JSONDecodeError:
                                litellm.verbose_logger.warning(
                                    f"Failed to parse SSE chunk: {decoded}"
                                )
                                continue

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                error_message = e.response.get("Error", {}).get("Message", str(e))
                self._handle_boto3_error(error_code, error_message)
            except Exception as e:
                raise BedrockError(
                    status_code=500, message=f"AgentCore: Streaming failed: {str(e)}"
                ) from e

        # Create a minimal logging object for CustomStreamWrapper
        from litellm.litellm_core_utils.litellm_logging import Logging

        logging_obj = Logging(
            model=model,
            messages=[],
            stream=True,
            call_type="completion",
            litellm_call_id="",
            start_time=time.time(),
            function_id="",
        )
        logging_obj.model_call_details = {"litellm_params": {}}

        # Create wrapper - session_id will be set in each chunk by the generator
        # Don't set in wrapper._hidden_params because actual_session_id isn't known until first API call
        return CustomStreamWrapper(
            completion_stream=stream_generator(),
            model=model,
            custom_llm_provider="bedrock",
            logging_obj=logging_obj,
        )

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        api_base: str,
        model_response: ModelResponse,
        print_verbose: callable,
        encoding: Any,
        logging_obj: Any,
        optional_params: Dict[str, Any],
        timeout: Optional[Union[float, int]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs,
    ) -> Union[ModelResponse, AsyncIterator[ModelResponse]]:
        """
        Asynchronous completion for AgentCore.

        Note: AgentCore boto3 client is synchronous, so this wraps the sync call
        """
        # For now, AgentCore boto3 client doesn't support async operations
        # We'll wrap the synchronous call in an async function
        import asyncio

        def sync_call():
            return self.completion(
                model=model,
                messages=messages,
                api_base=api_base,
                model_response=model_response,
                print_verbose=print_verbose,
                encoding=encoding,
                logging_obj=logging_obj,
                optional_params=optional_params,
                timeout=timeout,
                litellm_params=litellm_params,
                acompletion=False,  # Mark as sync internally
                stream=stream,
                **kwargs,
            )

        # Run synchronous call in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, sync_call)

        if stream:
            # Convert synchronous stream to async iterator
            async def async_stream_wrapper():
                for chunk in result:
                    yield chunk

            return async_stream_wrapper()
        else:
            return result

    def _handle_boto3_error(self, error_code: str, error_message: str) -> NoReturn:
        """
        Handle boto3 ClientError exceptions from AgentCore API.

        Args:
            error_code: AWS error code from ClientError
            error_message: Error message from ClientError

        Raises:
            BedrockError with appropriate status code
        """
        # Map AWS error codes to HTTP status codes
        status_code_map = {
            "ValidationException": 400,
            "UnauthorizedException": 401,
            "AccessDeniedException": 403,
            "ResourceNotFoundException": 404,
            "ThrottlingException": 429,
            "InternalServerException": 500,
            "ServiceUnavailableException": 503,
            "RuntimeClientError": 424,  # Failed Dependency - container not ready
        }

        error_message_map = {
            "ValidationException": f"AgentCore: Bad Request - {error_message}",
            "UnauthorizedException": f"AgentCore: Authentication Failed - {error_message}",
            "AccessDeniedException": f"AgentCore: Permission Denied - {error_message}",
            "ResourceNotFoundException": f"AgentCore: Agent Not Found - {error_message}",
            "ThrottlingException": f"AgentCore: Rate Limit Exceeded - {error_message}",
            "InternalServerException": f"AgentCore: Internal Error - {error_message}",
            "ServiceUnavailableException": f"AgentCore: Service Unavailable - {error_message}",
            "RuntimeClientError": f"AgentCore: Runtime container unavailable (cold start) - {error_message}",
        }

        status_code = status_code_map.get(error_code, 500)
        formatted_message = error_message_map.get(
            error_code, f"AgentCore: API Error ({error_code}) - {error_message}"
        )

        raise BedrockError(status_code=status_code, message=formatted_message)


def completion(
    model: str,
    messages: List[Dict[str, str]],
    api_base: str,
    model_response: ModelResponse,
    print_verbose: callable,
    encoding: Any,
    logging_obj: Any,
    optional_params: Dict[str, Any],
    timeout: Optional[Union[float, int]] = None,
    litellm_params: Optional[Dict[str, Any]] = None,
    acompletion: bool = False,
    stream: bool = False,
    **kwargs,
) -> Union[ModelResponse, CustomStreamWrapper]:
    """
    Main entry point for AgentCore completions (sync).

    Called by LiteLLM when model starts with "agentcore/".
    """
    config = AgentCoreConfig()
    return config.completion(
        model=model,
        messages=messages,
        api_base=api_base,
        model_response=model_response,
        print_verbose=print_verbose,
        encoding=encoding,
        logging_obj=logging_obj,
        optional_params=optional_params,
        timeout=timeout,
        litellm_params=litellm_params,
        acompletion=acompletion,
        stream=stream,
        **kwargs,
    )


async def acompletion(
    model: str,
    messages: List[Dict[str, str]],
    api_base: str,
    model_response: ModelResponse,
    print_verbose: callable,
    encoding: Any,
    logging_obj: Any,
    optional_params: Dict[str, Any],
    timeout: Optional[Union[float, int]] = None,
    litellm_params: Optional[Dict[str, Any]] = None,
    stream: bool = False,
    **kwargs,
) -> Union[ModelResponse, AsyncIterator[ModelResponse]]:
    """
    Main entry point for AgentCore completions (async).

    Called by LiteLLM when model starts with "agentcore/" and async mode is used.
    """
    config = AgentCoreConfig()
    return await config.acompletion(
        model=model,
        messages=messages,
        api_base=api_base,
        model_response=model_response,
        print_verbose=print_verbose,
        encoding=encoding,
        logging_obj=logging_obj,
        optional_params=optional_params,
        timeout=timeout,
        litellm_params=litellm_params,
        stream=stream,
        **kwargs,
    )
