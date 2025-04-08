from typing import List, Optional

from litellm.types.adapter import AdapterItem
from litellm.types.llms.custom_llm import CustomLLMItem

adapters: List[AdapterItem] = []
custom_provider_map: List[CustomLLMItem] = []
_custom_providers: List[str] = (
    []
)  # internal helper util, used to track names of custom providers
disable_hf_tokenizer_download: Optional[bool] = (
    None  # disable huggingface tokenizer download. Defaults to openai clk100
)
global_disable_no_log_param: bool = False
