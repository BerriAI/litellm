"""
Utility functions for base LLM classes.
"""

import copy
import json
from abc import ABC, abstractmethod
from typing import Any, Final

from openai.lib import _parsing, _pydantic
from pydantic import BaseModel

from litellm._logging import verbose_logger
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolCallChunk
from litellm.types.utils import Message, ProviderSpecificModelInfo, TokenCountResponse


class BaseTokenCounter(ABC):
    @abstractmethod
    async def count_tokens(
        self,
        model_to_use: str,
        messages: list[dict[str, Any]] | None,
        contents: list[dict[str, Any]] | None,
        deployment: dict[str, Any] | None = None,
        request_model: str = "",
        tools: list[dict[str, Any]] | None = None,
        system: Any | None = None,
    ) -> TokenCountResponse | None:
        pass

    @abstractmethod
    def should_use_token_counting_api(
        self,
        custom_llm_provider: str | None = None,
    ) -> bool:
        """
        Returns True if we should the this API for token counting for the selected `custom_llm_provider`
        """
        return False


class BaseLLMModelInfo(ABC):
    def get_provider_info(
        self,
        model: str,
    ) -> ProviderSpecificModelInfo | None:
        """
        Default values all models of this provider support.
        """
        return None

    @abstractmethod
    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        """
        Returns a list of models supported by this provider.
        """
        return []

    @staticmethod
    @abstractmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        pass

    @staticmethod
    @abstractmethod
    def get_api_base(
        api_base: str | None = None,
    ) -> str | None:
        pass

    @abstractmethod
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def get_base_model(model: str) -> str | None:
        """
        Returns the base model name from the given model name.

        Some providers like bedrock - can receive model=`invoke/anthropic.claude-3-opus-20240229-v1:0` or `converse/anthropic.claude-3-opus-20240229-v1:0`
            This function will return `anthropic.claude-3-opus-20240229-v1:0`
        """

    def get_token_counter(self) -> BaseTokenCounter | None:
        """
        Factory method to create a token counter for this provider.

        Returns:
            Optional TokenCounterInterface implementation for this provider,
            or None if token counting is not supported.
        """
        return None


def _convert_tool_response_to_message(
    tool_calls: list[ChatCompletionToolCallChunk],
) -> Message | None:
    """
    In JSON mode, Anthropic API returns JSON schema as a tool call, we need to convert it to a message to follow the OpenAI format

    """
    ## HANDLE JSON MODE - anthropic returns single function call
    json_mode_content_str: Final[str | None] = tool_calls[0]["function"].get("arguments")
    try:
        if json_mode_content_str is not None:
            args: Final = json.loads(json_mode_content_str)
            if isinstance(args, dict) and (values := args.get("values")) is not None:
                _message = Message(content=json.dumps(values))
                return _message
            else:
                # a lot of the times the `values` key is not present in the tool response
                # relevant issue: https://github.com/BerriAI/litellm/issues/6741
                _message = Message(content=json.dumps(args))
                return _message
    except json.JSONDecodeError:
        # json decode error does occur, return the original tool response str
        return Message(content=json_mode_content_str)
    return None


def _dict_to_response_format_helper(response_format: dict, ref_template: str | None = None) -> dict:
    if ref_template is not None and response_format.get("type") == "json_schema":
        # Deep copy to avoid modifying original
        modified_format: Final = copy.deepcopy(response_format)
        schema: Final = modified_format["json_schema"]["schema"]

        # Update all $ref values in the schema
        def update_refs(schema):
            stack: Final = [(schema, [])]
            visited: Final = set()

            while stack:
                obj, path = stack.pop()
                obj_id = id(obj)

                if obj_id in visited:
                    continue
                visited.add(obj_id)

                if isinstance(obj, dict):
                    if "$ref" in obj:
                        ref_path = obj["$ref"]
                        model_name = ref_path.split("/")[-1]
                        obj["$ref"] = ref_template.format(model=model_name)

                    for k, v in obj.items():
                        if isinstance(v, (dict, list)):
                            stack.append((v, path + [k]))

                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        if isinstance(item, (dict, list)):
                            stack.append((item, path + [i]))

        update_refs(schema)
        return modified_format
    return response_format


def type_to_response_format_param(
    response_format: type[BaseModel] | dict | None,
    ref_template: str | None = None,
) -> dict | None:
    """
    Re-implementation of openai's 'type_to_response_format_param' function

    Used for converting pydantic object to api schema.
    """
    if response_format is None:
        return None

    if isinstance(response_format, dict):
        return _dict_to_response_format_helper(response_format, ref_template)

    # type checkers don't narrow the negation of a `TypeGuard` as it isn't
    # a safe default behaviour but we know that at this point the `response_format`
    # can only be a `type`
    if not _parsing._completions.is_basemodel_type(response_format):
        raise TypeError(f"Unsupported response_format type - {response_format}")

    if ref_template is not None:
        schema = response_format.model_json_schema(ref_template=ref_template)
    else:
        schema = _pydantic.to_strict_json_schema(response_format)

    return {
        "type": "json_schema",
        "json_schema": {
            "schema": schema,
            "name": response_format.__name__,
            "strict": True,
        },
    }


def map_developer_role_to_system_role(
    messages: list[AllMessageValues],
) -> list[AllMessageValues]:
    """
    Translate `developer` role to `system` role for non-OpenAI providers.
    """
    new_messages: Final[list[AllMessageValues]] = []
    for m in messages:
        if m["role"] == "developer":
            verbose_logger.debug(
                "Translating developer role to system role for non-OpenAI providers."
            )  # ensure user knows what's happening with their input.
            new_messages.append({"role": "system", "content": m["content"]})
        else:
            new_messages.append(m)
    return new_messages
