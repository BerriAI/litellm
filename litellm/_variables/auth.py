import os
from typing import Optional, Union

from litellm.caching.llm_caching_handler import LLMClientCache

email: Optional[str] = (
    None  # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
token: Optional[str] = (
    None  # Not used anymore, will be removed in next MAJOR release - https://github.com/BerriAI/litellm/discussions/648
)
telemetry = True
max_tokens = 256  # OpenAI Defaults
drop_params = bool(os.getenv("LITELLM_DROP_PARAMS", False))
modify_params = False
retry = True
### AUTH ###
api_key: Optional[str] = None
openai_key: Optional[str] = None
groq_key: Optional[str] = None
databricks_key: Optional[str] = None
openai_like_key: Optional[str] = None
azure_key: Optional[str] = None
anthropic_key: Optional[str] = None
replicate_key: Optional[str] = None
cohere_key: Optional[str] = None
infinity_key: Optional[str] = None
clarifai_key: Optional[str] = None
maritalk_key: Optional[str] = None
ai21_key: Optional[str] = None
ollama_key: Optional[str] = None
openrouter_key: Optional[str] = None
predibase_key: Optional[str] = None
huggingface_key: Optional[str] = None
vertex_project: Optional[str] = None
vertex_location: Optional[str] = None
predibase_tenant_id: Optional[str] = None
togetherai_api_key: Optional[str] = None
cloudflare_api_key: Optional[str] = None
baseten_key: Optional[str] = None
aleph_alpha_key: Optional[str] = None
nlp_cloud_key: Optional[str] = None
snowflake_key: Optional[str] = None
common_cloud_provider_auth_params: dict = {
    "params": ["project", "region_name", "token"],
    "providers": ["vertex_ai", "bedrock", "watsonx", "azure", "vertex_ai_beta"],
}
use_client: bool = False
ssl_verify: Union[str, bool] = True
ssl_certificate: Optional[str] = None
disable_streaming_logging: bool = False
disable_add_transform_inline_image_block: bool = False
safe_memory_mode: bool = False
enable_azure_ad_token_refresh: Optional[bool] = False
in_memory_llm_clients_cache: LLMClientCache = LLMClientCache()
