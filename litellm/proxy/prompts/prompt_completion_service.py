from typing import Any, Dict, Optional, Tuple, List, cast
from pydantic import BaseModel, Field
from fastapi import HTTPException

from litellm.integrations.dotprompt import DotpromptManager
from litellm.proxy._types import LitellmUserRoles
from litellm.types.prompts.init_prompts import PromptTemplateBase

from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.proxy.prompts.prompt_registry import PROMPT_HUB

from litellm.integrations.gitlab import GitLabPromptManager


class PromptCompletionRequest(BaseModel):
    prompt_id: str = Field(..., description="Unique ID of the prompt registered in PromptHub.")
    prompt_version: Optional[str] = Field(None, description="Optional version identifier.")
    prompt_variables: Dict[str, Any] = Field(default_factory=dict, description="Key-value mapping for template variables.")


class PromptCompletionResponse(BaseModel):
    prompt_id: str
    prompt_version: Optional[str]
    model: str
    metadata: Dict[str, Any]
    variables: Dict[str, Any]
    completion_text: str
    raw_response: Dict[str, Any]

class PromptCompletionService:
    """
    Service encapsulating all helper logic for generating completions from managed prompts.
    """

    def __init__(self, user_api_key_dict):
        self.user_api_key_dict = user_api_key_dict

    # ------------------------------------------------------------------
    # Access Control
    # ------------------------------------------------------------------
    def validate_access(self, prompt_id: str) -> None:
        prompts: Optional[List[str]] = None

        if self.user_api_key_dict.metadata is not None:
            prompts = cast(
                Optional[List[str]], self.user_api_key_dict.metadata.get("prompts", None)
            )
            if prompts is not None and prompt_id not in prompts:
                raise HTTPException(status_code=400, detail=f"Prompt {prompt_id} not found")
        if self.user_api_key_dict.user_role not in (
                LitellmUserRoles.PROXY_ADMIN,
                LitellmUserRoles.PROXY_ADMIN.value,
        ):
            raise HTTPException(
                status_code=403,
                detail=(
                    f"You are not authorized to access this prompt. "
                    f"Your role - {self.user_api_key_dict.user_role}, Your key's prompts - {prompts}"
                ),
            )

    # ------------------------------------------------------------------
    # Prompt Loading
    # ------------------------------------------------------------------
    def load_prompt_and_callback(
            self, prompt_id: str
    ) -> Tuple[CustomPromptManagement, "PromptTemplateBase", dict]:


        prompt_spec = PROMPT_HUB.get_prompt_by_id(prompt_id)
        if prompt_spec is None:
            raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")

        prompt_callback: Optional[CustomPromptManagement] = PROMPT_HUB.get_prompt_callback_by_id(
            prompt_id
        )
        if prompt_callback is None:
            raise HTTPException(status_code=404, detail=f"No callback found for prompt {prompt_id}")

        prompt_template: Optional[PromptTemplateBase] = None

        if isinstance(prompt_callback, DotpromptManager):
            template = prompt_callback.prompt_manager.get_all_prompts_as_json()
            if template and len(template) == 1:
                tid = list(template.keys())[0]
                prompt_template = PromptTemplateBase(
                    litellm_prompt_id=tid,
                    content=template[tid]["content"],
                    metadata=template[tid]["metadata"],
                )

        elif isinstance(prompt_callback, GitLabPromptManager):
            prompt_json = prompt_spec.model_dump()
            prompt_template = PromptTemplateBase(
                litellm_prompt_id=prompt_json.get("prompt_id", ""),
                content=prompt_json.get("litellm_params", {})
                .get("model_config", {})
                .get("content", ""),
                metadata=prompt_json.get("litellm_params", {})
                .get("model_config", {})
                .get("metadata", {}),
                )

        if not prompt_template:
            raise HTTPException(
                status_code=400, detail=f"Could not load prompt template for {prompt_id}"
            )

        return prompt_callback, prompt_template, prompt_spec

    # ------------------------------------------------------------------
    # Parameter Handling
    # ------------------------------------------------------------------
    def _flatten_config(self, d: dict) -> dict:
        if "config" in d and isinstance(d["config"], dict):
            flattened = {**d, **d["config"]}
            flattened.pop("config", None)
            return flattened
        return d

    def merge_params(
            self, metadata: dict, prompt_spec, user_overrides: Optional[dict]
    ) -> dict:
        """Merge and flatten all parameter sources."""
        base_params = metadata.get("config", {}) or {}
        prompt_params = (
            prompt_spec.litellm_params.get("config", {})
            if hasattr(prompt_spec, "litellm_params") and isinstance(prompt_spec.litellm_params, dict)
            else {}
        )
        user_overrides = user_overrides or {}

        base_params = self._flatten_config(base_params)
        prompt_params = self._flatten_config(prompt_params)
        user_overrides = self._flatten_config(user_overrides)

        merged = {**base_params, **prompt_params, **user_overrides}
        merged.setdefault("stream", False)
        merged["user"] = self.user_api_key_dict.user_id

        # Remove invalid keys
        merged.pop("model", None)
        merged.pop("messages", None)
        return merged

    # ------------------------------------------------------------------
    # LLM Invocation
    # ------------------------------------------------------------------
    async def execute_model_completion(
            self, model: str, completion_prompt: Tuple[str, list], merged_params: dict
    ) -> Any:
        try:
            import litellm
            return await litellm.acompletion(
                model=completion_prompt[0],
                messages=completion_prompt[1],
                **merged_params,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error invoking model: {str(e)}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    async def run_completion(
            self, prompt_id: str, variables: dict, prompt_version: Optional[str], extra_body: dict
    ) -> "PromptCompletionResponse":


        # Step 1: Access control
        self.validate_access(prompt_id)

        # Step 2: Load prompt and callback
        prompt_callback, prompt_template, prompt_spec = self.load_prompt_and_callback(prompt_id)
        metadata = prompt_template.metadata or {}
        model = metadata.get("model")
        if not model:
            raise HTTPException(
                status_code=400, detail=f"Model not specified in metadata for {prompt_id}"
            )

        # Step 3: Build chat completion prompt
        system_prompt = metadata.get("config", {}).get("system_prompt", "You are a helpful assistant.")
        completion_prompt = prompt_callback.get_chat_completion_prompt(
            model=model,
            messages=[{"role": "system", "content": system_prompt}],
            non_default_params=metadata,
            prompt_id=prompt_id,
            prompt_variables=variables,
            dynamic_callback_params={},
            prompt_label=None,
            prompt_version=prompt_version,
        )

        # Step 4: Merge params
        merged_params = self.merge_params(metadata, prompt_spec, extra_body)

        # Step 5: Execute model
        response = await self.execute_model_completion(model, completion_prompt, merged_params)

        # Step 6: Extract text and build response
        completion_text = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )

        return PromptCompletionResponse(
            prompt_id=prompt_id,
            prompt_version=prompt_version,
            model=model,
            metadata=metadata,
            variables=variables,
            completion_text=completion_text,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else response,
        )