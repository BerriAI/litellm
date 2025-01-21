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
        return response_format

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
