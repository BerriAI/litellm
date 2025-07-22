from typing import Optional, List, Union
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.types.llms.openai import AllMessageValues

class HerokuChatConfig(BaseConfig):
  def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        headers.update({"Authorization": f"Bearer {api_key}"})
        return headers
  
  def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return f"{api_base}/v1/chat/completions"