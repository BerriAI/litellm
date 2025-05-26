"""
Translating between OpenAI's `/chat/completion` format and Amazon's `/converse` format
"""

import copy
import hashlib
import time
import types
from datetime import datetime, timedelta
from typing import List, Literal, Optional, Tuple, Union, cast, overload, Iterator, AsyncIterator, Dict

import httpx

import litellm
from litellm import verbose_logger, DualCache
from litellm.litellm_core_utils.core_helpers import map_finish_reason
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _parse_content_for_reasoning,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    BedrockConverseMessagesProcessor,
    _bedrock_converse_messages_pt,
    _bedrock_tools_pt,
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret
from litellm.types.llms.bedrock import *
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionResponseMessage,
    ChatCompletionSystemMessage,
    ChatCompletionThinkingBlock,
    ChatCompletionToolCallChunk,
    ChatCompletionToolCallFunctionChunk,
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
    ChatCompletionUserMessage,
    OpenAIChatCompletionToolParam,
    OpenAIMessageContentListBlock,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Function,
    Message,
    ModelResponse,
    PromptTokensDetailsWrapper,
    Usage, ModelResponseStream, StreamingChoices, Delta,
)
from litellm.utils import add_dummy_tool, has_tool_call_blocks, supports_reasoning
from ..common_utils import SAPOAuthToken
from ...base_llm.base_model_iterator import BaseModelResponseIterator
from ...bedrock.chat.converse_transformation import AmazonConverseConfig
from ...bedrock.chat.invoke_handler import AWSEventStreamDecoder

from ...bedrock.common_utils import BedrockError, BedrockModelInfo, get_bedrock_tool_name
from ...openai_like.common_utils import OpenAILikeError


class SAPConverseConfig(AmazonConverseConfig):
    """
    Reference - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
    #2 - https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html#conversation-inference-supported-models-features
    """

    maxTokens: Optional[int]
    stopSequences: Optional[List[str]]
    temperature: Optional[int]
    topP: Optional[int]
    topK: Optional[int]

    def __init__(self, maxTokens: Optional[int] = None, stopSequences: Optional[List[str]] = None,
                 temperature: Optional[int] = None, topP: Optional[int] = None, topK: Optional[int] = None) -> None:
        super().__init__(maxTokens, stopSequences, temperature, topP, topK)
        self.token_cache = DualCache()

    def _transform_request_helper(
        self,
        model: str,
        system_content_blocks: List[SystemContentBlock],
        optional_params: dict,
        messages: Optional[List[AllMessageValues]] = None,
    ) -> CommonRequestObject:
        ## VALIDATE REQUEST
        """
        Bedrock doesn't support tool calling without `tools=` param specified.
        """
        if (
            "tools" not in optional_params
            and messages is not None
            and has_tool_call_blocks(messages)
        ):
            if litellm.modify_params:
                optional_params["tools"] = add_dummy_tool(
                    custom_llm_provider="bedrock_converse"
                )
            else:
                raise litellm.UnsupportedParamsError(
                    message="Bedrock doesn't support tool calling without `tools=` param specified. Pass `tools=` param OR set `litellm.modify_params = True` // `litellm_settings::modify_params: True` to add dummy tool to the request.",
                    model="",
                    llm_provider="bedrock",
                )

        inference_params = copy.deepcopy(optional_params)
        supported_converse_params = list(
            SAPConverseConfig.__annotations__.keys()
        ) + ["top_k"]
        supported_tool_call_params = ["tools", "tool_choice"]
        supported_config_params = list(self.get_config_blocks().keys())
        total_supported_params = (
            supported_converse_params
            + supported_tool_call_params
            + supported_config_params
        )
        inference_params.pop("json_mode", None)  # used for handling json_schema

        # keep supported params in 'inference_params', and set all model-specific params in 'additional_request_params'

        additional_request_params = {
            "reasoning_config": {"type": "enabled", "budget_tokens": 1024}
        } if "thinking" in model else {}

        inference_params = {
            k: v for k, v in inference_params.items() if k in total_supported_params
        }

        if "thinking" in model:
            inference_params.pop('topP', None)

        # Only set the topK value in for models that support it
        additional_request_params.update(
            self._handle_top_k_value(model, inference_params)
        )

        bedrock_tools: List[ToolBlock] = _bedrock_tools_pt(
            inference_params.pop("tools", [])
        )
        bedrock_tool_config: Optional[ToolConfigBlock] = None
        if len(bedrock_tools) > 0:
            tool_choice_values: ToolChoiceValuesBlock = inference_params.pop(
                "tool_choice", None
            )
            bedrock_tool_config = ToolConfigBlock(
                tools=bedrock_tools,
            )
            if tool_choice_values is not None:
                bedrock_tool_config["toolChoice"] = tool_choice_values

        data: CommonRequestObject = {
            "additionalModelRequestFields": additional_request_params,
            "system": system_content_blocks,
            "inferenceConfig": self._transform_inference_params(
                inference_params=inference_params
            ),
        }

        # Handle all config blocks
        for config_name, config_class in self.get_config_blocks().items():
            config_value = inference_params.pop(config_name, None)
            if config_value is not None:
                data[config_name] = config_class(**config_value)  # type: ignore

        # Tool Config
        if bedrock_tool_config is not None:
            data["toolConfig"] = bedrock_tool_config

        return data

    async def _async_transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
    ) -> RequestObject:
        messages, system_content_blocks = self._transform_system_message(messages)
        ## TRANSFORMATION ##

        _data: CommonRequestObject = self._transform_request_helper(
            model=model,
            system_content_blocks=system_content_blocks,
            optional_params=optional_params,
            messages=messages,
        )

        bedrock_messages = (
            await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
                messages=messages,
                model=model,
                llm_provider="bedrock_converse",
                user_continue_message=litellm_params.pop("user_continue_message", None),
            )
        )

        data: RequestObject = {"messages": bedrock_messages, **_data}

        return data

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return cast(
            dict,
            self._transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
            ),
        )

    def _transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
    ) -> RequestObject:
        messages, system_content_blocks = self._transform_system_message(messages)

        _data: CommonRequestObject = self._transform_request_helper(
            model=model,
            system_content_blocks=system_content_blocks,
            optional_params=optional_params,
            messages=messages,
        )

        ## TRANSFORMATION ##
        bedrock_messages: List[MessageBlock] = _bedrock_converse_messages_pt(
            messages=messages,
            model=model,
            llm_provider="bedrock_converse",
            user_continue_message=litellm_params.pop("user_continue_message", None),
        )

        data: RequestObject = {"messages": bedrock_messages, **_data}

        return data

    def validate_environment(
            self,
            headers: dict,
            model: str,
            messages: List[AllMessageValues],
            optional_params: dict,
            litellm_params: dict,
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
    ) -> Tuple[str, dict]:
        """
        Override to add SAP OAuth token support.
        If SAP credentials are provided, use OAuth token instead of API key.
        """
        # Try to get SAP token first
        sap_token = None
        if optional_params:
            sap_token = self._get_sap_token_from_params(optional_params, headers)

            # Set up headers with OAuth token
            if headers is None:
                headers = {}

            headers.update({
                "Content-Type": "application/json",
                "Authorization": f"{sap_token.token_type} {sap_token.access_token}",
            })

            # TODO: Check embedding here
            # elif endpoint_type == "embeddings":
            #     api_base = f"{api_base}/embeddings?api-version=2025-03-01-preview"

            return headers

    def _get_sap_token_from_params(
            self,
            optional_params: dict,
            headers: Optional[dict] = None,
    ) -> Optional[SAPOAuthToken]:
        """
        Extract SAP credentials from optional_params and get OAuth token.
        Returns None if SAP credentials are not provided.
        """
        # Check if SAP credentials are provided
        sap_client_id = optional_params.pop("sap_client_id", None) or get_secret("UAA_CLIENT_ID")
        sap_client_secret = optional_params.pop("sap_client_secret", None) or get_secret("UAA_CLIENT_SECRET")
        sap_xsuaa_url = optional_params.pop("sap_xsuaa_url", None) or get_secret("UAA_URL")

        # If no SAP credentials, return None (fallback to regular auth)
        if not all([sap_client_id, sap_client_secret, sap_xsuaa_url]):
            return None

        # Check cache
        cache_key = self._get_cache_key(sap_client_id, sap_client_secret, sap_xsuaa_url)
        cached_token = self.token_cache.get_cache(cache_key)

        if cached_token and isinstance(cached_token, SAPOAuthToken):
            if cached_token.expires_at > datetime.now():
                verbose_logger.debug("Using cached SAP OAuth token")
                return cached_token

        # Get new token
        token = self._get_sap_oauth_token(
            client_id=sap_client_id,
            client_secret=sap_client_secret,
            xsuaa_url=sap_xsuaa_url,
        )

        # Cache the token
        ttl = (token.expires_at - datetime.now()).total_seconds()
        self.token_cache.set_cache(cache_key, token, ttl=int(ttl))

        return token

    def _get_cache_key(self, client_id: str, client_secret: str, xsuaa_url: str) -> str:
        """Generate a unique cache key based on credentials"""
        credential_str = json.dumps({
            "client_id": client_id,
            "client_secret": client_secret,
            "xsuaa_url": xsuaa_url
        }, sort_keys=True)
        return f"sap_oauth_{hashlib.sha256(credential_str.encode()).hexdigest()}"

    def _get_sap_oauth_token(
            self,
            client_id: str,
            client_secret: str,
            xsuaa_url: str,
    ) -> SAPOAuthToken:
        """
        Exchange client credentials for an OAuth token via SAP xsuaa.
        """
        verbose_logger.debug(
            f"Exchanging SAP credentials for OAuth token at {xsuaa_url}"
        )

        token_endpoint = f"{xsuaa_url}/oauth/token"

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }

        data = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }

        try:
            response = httpx.post(
                token_endpoint,
                headers=headers,
                data=data,
                timeout=30.0,
            )
            response.raise_for_status()

            token_data = response.json()

            # Calculate token expiration time (subtract 60 seconds for safety margin)
            expires_in = token_data.get("expires_in", 3600)
            expires_at = datetime.now() + timedelta(seconds=expires_in - 60)

            return SAPOAuthToken(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=expires_in,
                expires_at=expires_at,
            )

        except httpx.HTTPStatusError as e:
            raise OpenAILikeError(
                status_code=e.response.status_code,
                message=f"Failed to get SAP OAuth token: {e.response.text}",
            )
        except Exception as e:
            raise OpenAILikeError(
                status_code=500,
                message=f"Error getting SAP OAuth token: {str(e)}",
            )

    def get_complete_url(
    self,
    api_base: Optional[str],
    api_key: Optional[str],
    model: str,
    optional_params: dict,
    litellm_params: dict,
    stream: Optional[bool] = None):
        # Extract deployment ID if provided
        deployment_id = optional_params.pop("sap_deployment_id")
        if stream:
            # For SAP AI Core, the deployment ID is part of the URL path
            api_base = f"{api_base}/v2/inference/deployments/{deployment_id}/converse-stream"
        else:
            api_base = f"{api_base}/v2/inference/deployments/{deployment_id}/converse"
        return api_base
    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        return SAPConverseStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

converse_config = SAPConverseConfig()

class SAPConverseStreamingHandler(BaseModelResponseIterator):
    def chunk_parser(self, chunk: dict) -> ModelResponseStream:
        try:
            import json
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            finish_reason = ""
            usage: Optional[Usage] = None
            provider_specific_fields: Dict[str, Any] = {}
            reasoning_content: Optional[str] = None
            thinking_blocks: Optional[
                List[
                    Union[
                        ChatCompletionThinkingBlock, ChatCompletionRedactedThinkingBlock
                    ]
                ]
            ] = None
            index = 0  # Default index since this format doesn't seem to include it

            # Handle messageStart
            if 'messageStart' in chunk:
                """
                chunk = {'messageStart': {'role': 'assistant'}}
                """
                # MessageStart typically doesn't contain text content
                pass

            # Handle contentBlockDelta
            elif 'contentBlockDelta' in chunk:
                """
                chunk = {'contentBlockDelta': {'delta': {'text': "I'm doing well"}, 'contentBlockIndex': 0}}
                """
                content_block_delta = chunk['contentBlockDelta']
                index = content_block_delta.get('contentBlockIndex', 0)

                delta = content_block_delta.get('delta', {})

                # Handle text delta
                if 'text' in delta:
                    text = delta['text']

                # Handle tool use delta - following Bedrock structure
                elif 'toolUse' in delta:
                    tool_delta = delta['toolUse']

                    # Handle toolUseId and name (start of tool call)
                    if 'toolUseId' in tool_delta and 'name' in tool_delta:
                        tool_use = {
                            "id": tool_delta['toolUseId'],
                            "type": "function",
                            "function": {
                                "name": tool_delta['name'],
                                "arguments": "",
                            },
                        }

                    # Handle input arguments (continuing tool call)
                    elif 'input' in tool_delta:
                        # Convert input dict to JSON string if it's a dict
                        input_data = tool_delta['input']
                        if isinstance(input_data, dict):
                            input_str = json.dumps(input_data)
                        else:
                            input_str = str(input_data)

                        tool_use = {
                            "id": None,
                            "type": "function",
                            "function": {
                                "name": None,
                                "arguments": input_str,
                            },
                            # "index": self.tool_index,
                        }

                # Handle reasoning content blocks (Bedrock style)
                elif 'reasoningContent' in delta:
                    reasoning_block = delta['reasoningContent']
                    if reasoning_block:
                        # Transform reasoning content using similar logic from _transform_response
                        reasoning_content = reasoning_block.get('text')
                        thinking_block = self._transform_thinking_block(reasoning_block)
                        thinking_blocks = [thinking_block]
            # Handle contentBlockStop
            elif 'contentBlockStop' in chunk:
                """
                chunk = {'contentBlockStop': {'contentBlockIndex': 0}}
                """
                content_block_stop = chunk['contentBlockStop']
                index = content_block_stop.get('contentBlockIndex', 0)

                # Check if we have an incomplete tool call that needs empty args

                # TODO
                # is_empty = self.check_empty_tool_call_args()
                # if is_empty:
                #     tool_use = {
                #         "id": None,
                #         "type": "function",
                #         "function": {
                #             "name": None,
                #             "arguments": "{}",
                #         },
                #         "index": self.tool_index,
                #     }

            # Handle messageStop
            elif 'messageStop' in chunk:
                """
                chunk = {'messageStop': {'stopReason': 'end_turn'}}
                """
                message_stop = chunk['messageStop']
                stop_reason = message_stop.get('stopReason', 'stop')

                # Use the same mapping logic as in _transform_response
                finish_reason = map_finish_reason(stop_reason)

            # Handle metadata (usage and metrics)
            elif 'metadata' in chunk:
                """
                chunk = {'metadata': {'usage': {'inputTokens': 11, 'outputTokens': 18, 'totalTokens': 29}, 'metrics': {'latencyMs': 515}}}
                """
                metadata = chunk['metadata']

                if 'usage' in metadata:
                    usage_data = metadata['usage']
                    usage = Usage(
                        prompt_tokens=usage_data.get('inputTokens', 0),
                        completion_tokens=usage_data.get('outputTokens', 0),
                        total_tokens=usage_data.get('totalTokens', 0)
                    )

                # Store metrics in provider specific fields
                if 'metrics' in metadata:
                    provider_specific_fields['metrics'] = metadata['metrics']

            # Handle unknown chunk types
            else:
                # Log or handle unknown chunk types
                provider_specific_fields['unknown_chunk'] = chunk

            # Handle JSON mode processing similar to _transform_response
            if hasattr(self, 'json_mode') and self.json_mode is True and tool_use is not None:
                # Support 'json_schema' logic on bedrock models
                json_mode_content_str = tool_use.get("function", {}).get("arguments")
                if json_mode_content_str is not None:
                    text = json_mode_content_str
                    tool_use = None  # Clear tool_use when using json_mode

            # Apply additional processing if needed
            # text, tool_use = self._handle_json_mode_chunk(text=text, tool_use=tool_use)

            # Construct and return the response chunk
            returned_chunk = ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=index,
                        delta=Delta(
                            content=text,
                            tool_calls=[tool_use] if tool_use is not None else None,
                            provider_specific_fields=(
                                provider_specific_fields
                                if provider_specific_fields
                                else None
                            ),
                            thinking_blocks=(
                                thinking_blocks if thinking_blocks else None
                            ),
                            reasoning_content=reasoning_content,
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                usage=usage,
            )
            return returned_chunk

        except Exception as e:
            raise BedrockError(
                message=f"Failed to parse chunk: {chunk}. Error: {str(e)}",
                status_code=422,
            )
    def _transform_thinking_block(self, think_block):
        _thinking_block = ChatCompletionThinkingBlock(type="thinking")
        _text = think_block.get("text")
        _signature = think_block.get("signature")
        if _text is not None:
            _thinking_block["thinking"] = _text
        if _signature is not None:
            _thinking_block["signature"] = _signature
        return _thinking_block