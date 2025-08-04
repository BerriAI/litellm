from typing import Any, Dict, List, Optional
from litellm.proxy._types import TokenCountResponse
from litellm.proxy.token_counting_factory import TokenCounterInterface


class AnthropicTokenCounter(TokenCounterInterface):
    def supports_provider(
        self, 
        deployment: Optional[Dict[str, Any]] = None,
        from_endpoint: bool = False
    ) -> bool:
        if not from_endpoint:
            return False
            
        if deployment is None:
            return False
            
        full_model = deployment.get("litellm_params", {}).get("model", "")
        is_anthropic_provider = full_model.startswith("anthropic/") or "anthropic" in full_model.lower()
        
        return is_anthropic_provider
    
    async def count_tokens(
        self,
        model_to_use: str,
        messages: Optional[List[Dict[str, Any]]],
        deployment: Optional[Dict[str, Any]] = None,
        request_model: str = "",
    ) -> Optional[TokenCountResponse]:
        from litellm.proxy.utils import count_tokens_with_anthropic_api
        
        result = await count_tokens_with_anthropic_api(
            model_to_use=model_to_use,
            messages=messages,
            deployment=deployment,
        )
        
        if result is not None:
            return TokenCountResponse(
                total_tokens=result["total_tokens"],
                request_model=request_model,
                model_used=model_to_use,
                tokenizer_type=result["tokenizer_used"],
            )
        
        return None