import json
from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel
from typing_extensions import override

from litellm.integrations.opentelemetry_utils.base_otel_llm_obs_attributes import (
    BaseLLMObsOTELAttributes,
    safe_set_attribute,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span


class LangfuseLLMObsOTELAttributes(BaseLLMObsOTELAttributes):
    @staticmethod
    @override
    def set_messages(span: "Span", messages: List[Dict[str, Any]]):
        safe_set_attribute(
            span, "langfuse.observation.input", json.dumps({"messages": messages})
        )

    @staticmethod
    @override
    def set_response_output_messages(span: "Span", response_obj):
        safe_set_attribute(
            span,
            "langfuse.observation.output",
            (
                response_obj.model_dump_json()
                if isinstance(response_obj, BaseModel)
                else response_obj
            ),
        )
