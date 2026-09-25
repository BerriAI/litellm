from typing_extensions import TypedDict

from litellm.integrations.custom_logger import CustomLogger


class AdapterItem(TypedDict):
    id: str
    adapter: CustomLogger
