from typing import List, Optional, TypedDict, Union

from .llms.openai import OpenAIRealtimeEvents


class RealtimeResponseTypedDict(TypedDict):
    response: Union[OpenAIRealtimeEvents, List[OpenAIRealtimeEvents]]
    current_output_item_id: Optional[str]
    current_response_id: Optional[str]
