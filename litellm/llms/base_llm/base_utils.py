"""
Utility functions for base LLM classes.
"""

import copy
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from openai.lib import _parsing, _pydantic  # pyright: ignore[reportPrivateUsage]  # SDK parser internals
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

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

    def get_model_cost_key(self, model: str) -> str | None:
        """
        Maps the model name a user sends to the key `litellm.model_cost` stores it under, when the two differ.
        `get_model_info` tries this key once the exact `model` and `provider/model` keys miss. The default None means
        the provider's user-facing names already match the cost map, so there is nothing extra to try.
        """
        return None

    @abstractmethod
    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        """
        Returns a list of models supported by this provider.
        """
        return []

    def discover_models(
        self, litellm_params: Mapping[str, object] | None = None
    ) -> list[str]:  # mutable-ok: matches get_models' list[str] contract shared by every provider override
        """
        Live model discovery for a configured deployment. Defaults to the api_key/api_base
        facade every provider already implements via ``get_models``; a provider whose
        discovery needs more of ``litellm_params`` (e.g. Anthropic's workload identity
        federation) overrides this instead of widening ``get_models`` for every provider.
        """
        api_key: Final = litellm_params.get("api_key") if litellm_params is not None else None
        api_base: Final = litellm_params.get("api_base") if litellm_params is not None else None
        return self.get_models(
            api_key=api_key if isinstance(api_key, str) else None,
            api_base=api_base if isinstance(api_base, str) else None,
        )

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


def convert_tool_response_to_message(
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


_convert_tool_response_to_message = convert_tool_response_to_message


@dataclass(frozen=True, slots=True)
class ResponseFormatSchema:
    document: Mapping[str, JsonValue]
    property_names: frozenset[str]
    required: tuple[str, ...]

    def accepts(self, value: Mapping[str, JsonValue]) -> bool:
        from jsonschema.validators import validator_for
        from referencing import Registry
        from referencing.exceptions import Unresolvable

        validator: Final = validator_for(self.document)(dict(self.document), registry=Registry())
        try:
            return validator.is_valid(dict(value))
        except Unresolvable:
            return False


_JSON_OBJECT: Final = TypeAdapter(Mapping[str, JsonValue])
_REQUIRED_NAMES: Final = TypeAdapter(tuple[str, ...])


def as_json_object(value: object) -> Mapping[str, JsonValue] | None:
    try:
        return _JSON_OBJECT.validate_python(value)
    except ValidationError:
        return None


def response_format_schema(value: object) -> ResponseFormatSchema | None:
    document: Final = as_json_object(value)
    if document is None:
        return None
    try:
        required: Final = _REQUIRED_NAMES.validate_python(document.get("required", ()))
    except ValidationError:
        return None
    return ResponseFormatSchema(
        document=document,
        property_names=frozenset(as_json_object(document.get("properties")) or ()),
        required=required,
    )


def _is_schema_object(value: Mapping[str, JsonValue], schema: ResponseFormatSchema) -> bool:
    return not schema.property_names.isdisjoint(value) and all(key in value for key in schema.required)


RESPONSE_FORMAT_ENVELOPE_NAME_KEYS: Final = frozenset({"name", "tool_name"})


def _tool_call_envelope_payload(
    payload_object: Mapping[str, JsonValue], tool_name: str
) -> Mapping[str, JsonValue] | None:
    name_keys: Final = frozenset(
        key for key in RESPONSE_FORMAT_ENVELOPE_NAME_KEYS if payload_object.get(key) == tool_name
    )
    other_values: Final = tuple(as_json_object(value) for key, value in payload_object.items() if key not in name_keys)
    if not name_keys or len(other_values) != 1:
        return None
    return other_values[0]


def unwrap_response_format_envelope(payload: object, schema: ResponseFormatSchema, tool_name: str) -> object:
    payload_object: Final = as_json_object(payload)
    if payload_object is None:
        return payload
    tool_call_payload: Final = _tool_call_envelope_payload(payload_object, tool_name)
    if tool_call_payload is not None:
        return tool_call_payload
    if schema.accepts(payload_object) or _is_schema_object(payload_object, schema):
        return payload
    nested_objects: Final = tuple(as_json_object(value) for value in payload_object.values())
    candidates: Final = tuple(
        nested for nested in nested_objects if nested is not None and _is_schema_object(nested, schema)
    )
    return candidates[0] if len(candidates) == 1 else payload


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
    if not _parsing._completions.is_basemodel_type(response_format):  # pyright: ignore[reportPrivateUsage]  # SDK parser internals
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
