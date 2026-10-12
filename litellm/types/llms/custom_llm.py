from typing import TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from litellm.llms.custom_llm import CustomLLM


class CustomLLMItem(TypedDict):
    provider: str
    custom_handler: "CustomLLM"
