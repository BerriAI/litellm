import json
from typing import Any, List, Literal, Optional, Tuple, Union, cast

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.llm_response_utils.get_headers import (
    get_response_headers,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionImageObject,
    ChatCompletionToolParam,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    ProviderSpecificModelInfo,
)
from litellm.utils import (
    supports_function_calling,
    supports_tool_choice,
)

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import FireworksAIException


class FireworksAIConfig(OpenAIGPTConfig):
    """
    Reference: https://docs.fireworks.ai/api-reference/post-chatcompletions

    Fireworks AI Chat Completions API configuration.

    Fireworks is largely OpenAI-compatible. The tweaks below document where
    this config diverges from the base ``OpenAIGPTConfig``:

    Request transforms
    ------------------
    - **Model name prefixing**: bare model names are expanded to
      ``accounts/fireworks/models/<model>`` before sending.
    - **Document inlining**: ``#transform=inline`` is appended to non-data
      image URLs so non-vision models can process documents/images. Skipped
      for ``data:`` URLs and vision models. Controllable via
      ``disable_add_transform_inline_image_block``.
    - **File → image migration**: ``file`` content parts (PDFs) are
      converted to ``image_url`` parts for inlining.
    - **Field stripping**: ``cache_control`` and ``provider_specific_fields``
      are removed from messages (Fireworks rejects them).

    Response transforms
    -------------------
    - **Model prefixing**: the returned model name is prefixed with
      ``fireworks_ai/``.
    - **Tool-calls-in-content workaround**: some older Fireworks models
      (e.g. Llama-v3p3-70b) return tool calls as a JSON string in the
      ``content`` field with ``tool_calls: null``. The response handler
      detects this and moves it to the proper ``tool_calls`` field.
      (Newer models like Kimi, MiniMax, GLM, DeepSeek return tool calls
      correctly.)

    Parameter handling
    ------------------
    - All standard OpenAI params are passed through (``tool_choice``,
      ``response_format``, ``max_completion_tokens``, ``strict`` in tools).
    - Additional Fireworks-supported params: ``top_k``, ``top_logprobs``,
      ``seed``, ``logit_bias``, ``parallel_tool_calls``, ``thinking``,
      ``prompt_truncate_length``, ``context_length_exceeded_behavior``.
    - ``tools`` and ``tool_choice`` are conditionally added based on
      capability flags from ``model_prices_and_context_window.json``.
      For unlisted models (custom/fine-tuned), ``get_provider_info``
      defaults both to allowed.
    - ``reasoning_effort`` is always passed through — the Fireworks API
      accepts it on all models and handles unsupported cases itself.
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "fireworks_ai"

    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    top_k: Optional[int] = None
    frequency_penalty: Optional[int] = None
    presence_penalty: Optional[int] = None
    n: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    response_format: Optional[dict] = None
    user: Optional[str] = None
    logprobs: Optional[int] = None
    reasoning_effort: Optional[str] = None

    # Non OpenAI parameters - Fireworks AI only params
    prompt_truncate_length: Optional[int] = None
    context_length_exceeded_behavior: Optional[Literal["error", "truncate"]] = None

    def __init__(
        self,
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        frequency_penalty: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        n: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        response_format: Optional[dict] = None,
        user: Optional[str] = None,
        logprobs: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        prompt_truncate_length: Optional[int] = None,
        context_length_exceeded_behavior: Optional[Literal["error", "truncate"]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str):
        supported_params = [
            "stream",
            "max_completion_tokens",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "n",
            "stop",
            "response_format",
            "user",
            "logprobs",
            "top_logprobs",
            "seed",
            "logit_bias",
            "parallel_tool_calls",
            "thinking",
            "reasoning_effort",
            "prompt_truncate_length",
            "context_length_exceeded_behavior",
        ]

        if supports_function_calling(model=model, custom_llm_provider="fireworks_ai"):
            supported_params.append("tools")

        if supports_tool_choice(model=model, custom_llm_provider="fireworks_ai"):
            supported_params.append("tool_choice")

        return supported_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)

        for param, value in non_default_params.items():
            if param == "tool_choice":
                optional_params["tool_choice"] = value
            elif param == "response_format":
                optional_params["response_format"] = value
            elif param in supported_openai_params:
                if value is not None:
                    optional_params[param] = value

        return optional_params

    def _add_transform_inline_image_block(
        self,
        content: ChatCompletionImageObject,
        model: str,
        disable_add_transform_inline_image_block: Optional[bool],
    ) -> ChatCompletionImageObject:
        """
        Add transform_inline to the image_url (allows non-vision models to parse documents/images/etc.)
        - ignore if model is a vision model
        - ignore if user has disabled this feature
        """
        if (
            "vision" in model or disable_add_transform_inline_image_block
        ):  # allow user to toggle this feature.
            return content
        if isinstance(content["image_url"], str):
            # Skip base64 data URLs — appending #transform=inline corrupts the
            # base64 payload and causes an "Incorrect padding" decode error on
            # the Fireworks side.  Data URLs are already inlined by definition.
            # Lower-case before checking: URI schemes are case-insensitive (RFC 3986).
            if not content["image_url"].lower().startswith("data:"):
                content["image_url"] = f"{content['image_url']}#transform=inline"
        elif isinstance(content["image_url"], dict):
            url = content["image_url"]["url"]
            if not url.lower().startswith("data:"):
                content["image_url"]["url"] = f"{url}#transform=inline"
        return content

    def _transform_messages_helper(
        self, messages: List[AllMessageValues], model: str, litellm_params: dict
    ) -> List[AllMessageValues]:
        """
        Add 'transform=inline' to the url of the image_url
        """
        from litellm.litellm_core_utils.prompt_templates.common_utils import (
            filter_value_from_dict,
            migrate_file_to_image_url,
        )

        disable_add_transform_inline_image_block = cast(
            Optional[bool],
            litellm_params.get("disable_add_transform_inline_image_block")
            or litellm.disable_add_transform_inline_image_block,
        )
        ## For any 'file' message type with pdf content, move to 'image_url' message type
        for message in messages:
            if message["role"] == "user":
                _message_content = message.get("content")
                if _message_content is not None and isinstance(_message_content, list):
                    for idx, content in enumerate(_message_content):
                        if content["type"] == "file":
                            _message_content[idx] = migrate_file_to_image_url(content)
        for message in messages:
            if message["role"] == "user":
                _message_content = message.get("content")
                if _message_content is not None and isinstance(_message_content, list):
                    for content in _message_content:
                        if content["type"] == "image_url":
                            content = self._add_transform_inline_image_block(
                                content=content,
                                model=model,
                                disable_add_transform_inline_image_block=disable_add_transform_inline_image_block,
                            )
            filter_value_from_dict(cast(dict, message), "cache_control")
            # Remove fields not permitted by FireworksAI that may cause:
            # "Not permitted, field: 'messages[n].provider_specific_fields'"
            if isinstance(message, dict) and "provider_specific_fields" in message:
                cast(dict, message).pop("provider_specific_fields", None)

        return messages

    def get_provider_info(self, model: str) -> ProviderSpecificModelInfo:
        return {
            "supports_function_calling": True,
            "supports_tool_choice": True,
            "supports_pdf_input": True,
        }

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        if not model.startswith("accounts/") and "#" not in model:
            model = f"accounts/fireworks/models/{model}"
        messages = self._transform_messages_helper(
            messages=messages, model=model, litellm_params=litellm_params
        )
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def _handle_message_content_with_tool_calls(
        self,
        message: Message,
        tool_calls: Optional[List[ChatCompletionToolParam]],
    ) -> Message:
        """
        Fireworks AI sends tool calls in the content field instead of tool_calls

        Relevant Issue: https://github.com/BerriAI/litellm/issues/7209#issuecomment-2813208780
        """
        if (
            tool_calls is not None
            and message.content is not None
            and message.tool_calls is None
        ):
            try:
                function = Function(**json.loads(message.content))
                if function.name != RESPONSE_FORMAT_TOOL_NAME and function.name in [
                    tool["function"]["name"] for tool in tool_calls
                ]:
                    tool_call = ChatCompletionMessageToolCall(
                        function=function, id=str(uuid.uuid4()), type="function"
                    )
                    message.tool_calls = [tool_call]

                    message.content = None
            except Exception:
                pass

        return message

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
        ## LOGGING
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        try:
            completion_response = raw_response.json()
        except Exception as e:
            response_headers = getattr(raw_response, "headers", None)
            raise FireworksAIException(
                message="Unable to get json response - {}, Original Response: {}".format(
                    str(e), raw_response.text
                ),
                status_code=raw_response.status_code,
                headers=response_headers,
            )

        raw_response_headers = dict(raw_response.headers)

        additional_headers = get_response_headers(raw_response_headers)

        response = ModelResponse(**completion_response)

        if response.model is not None:
            response.model = "fireworks_ai/" + response.model

        ## FIREWORKS AI sends tool calls in the content field instead of tool_calls
        for choice in response.choices:
            cast(Choices, choice).message = (
                self._handle_message_content_with_tool_calls(
                    message=cast(Choices, choice).message,
                    tool_calls=optional_params.get("tools", None),
                )
            )

        response._hidden_params = {"additional_headers": additional_headers}

        return response

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("FIREWORKS_API_BASE")
            or "https://api.fireworks.ai/inference/v1"
        )  # type: ignore
        dynamic_api_key = api_key or (
            get_secret_str("FIREWORKS_API_KEY")
            or get_secret_str("FIREWORKS_AI_API_KEY")
            or get_secret_str("FIREWORKSAI_API_KEY")
            or get_secret_str("FIREWORKS_AI_TOKEN")
        )
        return api_base, dynamic_api_key

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        """
        Fetches available models from Fireworks AI.

        Uses the Management API ``/v1/accounts/{account_id}/models`` endpoint
        (documented at https://docs.fireworks.ai/api-reference/list-models).

        - Always queries ``accounts/fireworks/models`` with
          ``filter=supports_serverless=true`` to list publicly available
          serverless models.
        - If ``FIREWORKS_ACCOUNT_ID`` is set, also queries the user's account
          for dedicated deployments and merges both lists.
        """
        api_key = self.get_api_key(api_key)
        if api_key is None:
            raise ValueError(
                "Fireworks AI API key is not set. Please set FIREWORKS_API_KEY "
                "(or FIREWORKS_AI_API_KEY / FIREWORKSAI_API_KEY / FIREWORKS_AI_TOKEN)."
            )

        base_url = (
            api_base
            or get_secret_str("FIREWORKS_API_BASE")
            or "https://api.fireworks.ai"
        )
        if base_url.endswith("/inference/v1"):
            base_url = base_url[: -len("/inference/v1")]
        base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"}
        seen: set = set()
        result: List[str] = []

        for account, query_filter in self._get_model_list_targets():
            self._fetch_models_from_account(
                base_url, account, query_filter, headers, seen, result
            )

        return result

    @staticmethod
    def _get_model_list_targets() -> List[tuple]:
        """Return (account_id, filter) pairs to query."""
        targets: List[tuple] = [
            ("fireworks", "supports_serverless=true"),
        ]
        user_account = get_secret_str("FIREWORKS_ACCOUNT_ID")
        if user_account and user_account != "fireworks":
            targets.append((user_account, None))
        return targets

    @staticmethod
    def _fetch_models_from_account(
        base_url: str,
        account_id: str,
        query_filter: Optional[str],
        headers: dict,
        seen: set,
        result: List[str],
    ) -> None:
        """Paginate through /v1/accounts/{account_id}/models and append unique models."""
        page_token: Optional[str] = None
        while True:
            params: dict = {"pageSize": "200"}
            if query_filter:
                params["filter"] = query_filter
            if page_token:
                params["pageToken"] = page_token

            response = litellm.module_level_client.get(
                url=f"{base_url}/v1/accounts/{account_id}/models",
                headers=headers,
                params=params,
            )
            if response.status_code != 200:
                verbose_logger.warning(
                    "Failed to fetch models from Fireworks AI account '%s'. "
                    "Status %d: %s",
                    account_id,
                    response.status_code,
                    response.text,
                )
                break

            data = response.json()
            for model in data.get("models", []):
                name = model.get("name", "")
                if name and name not in seen:
                    seen.add(name)
                    result.append("fireworks_ai/" + name)

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or (
            get_secret_str("FIREWORKS_API_KEY")
            or get_secret_str("FIREWORKS_AI_API_KEY")
            or get_secret_str("FIREWORKSAI_API_KEY")
            or get_secret_str("FIREWORKS_AI_TOKEN")
        )
