import copy
from abc import ABC, abstractmethod
from typing import List, Optional, Type, Union

from openai.lib import _parsing, _pydantic
from pydantic import BaseModel

from litellm.types.utils import ModelInfoBase


class BaseLLMModelInfo(ABC):
    @abstractmethod
    def get_model_info(
        self,
        model: str,
        existing_model_info: Optional[ModelInfoBase] = None,
    ) -> Optional[ModelInfoBase]:
        pass

    @abstractmethod
    def get_models(self) -> List[str]:
        pass

    @staticmethod
    @abstractmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        pass

    @staticmethod
    @abstractmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        pass


def _dict_to_response_format_helper(
    response_format: dict, ref_template: Optional[str] = None
) -> dict:
    if ref_template is not None and response_format.get("type") == "json_schema":
        # Deep copy to avoid modifying original
        modified_format = copy.deepcopy(response_format)
        schema = modified_format["json_schema"]["schema"]

        # Update all $ref values in the schema
        def update_refs(obj):
            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_path = obj["$ref"]
                    # Extract model name from reference path
                    model_name = ref_path.split("/")[-1]
                    # Apply new template
                    obj["$ref"] = ref_template.format(model=model_name)
                for value in obj.values():
                    update_refs(value)
            elif isinstance(obj, list):
                for item in obj:
                    update_refs(item)

        update_refs(schema)
        return modified_format
    return response_format


def type_to_response_format_param(
    response_format: Optional[Union[Type[BaseModel], dict]],
    ref_template: Optional[str] = None,
) -> Optional[dict]:
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
