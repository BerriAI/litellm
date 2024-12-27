from typing import Literal, Optional, Tuple

import httpx


class OpenAILikeError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(method="POST", url="https://www.litellm.ai")
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class OpenAILikeBase:
    def __init__(self, **kwargs):
        pass

    def _add_endpoint_to_api_base(
        self, api_base: str, endpoint_type: Literal["chat_completions", "embeddings"]
    ) -> str:
        original_url = httpx.URL(api_base)
        base_url = original_url.copy_with(params={})
        path = original_url.path

        if endpoint_type == "chat_completions" and "/chat/completions" not in path:
            modified_url = base_url.join("chat/completions")
        elif endpoint_type == "embeddings" and "/embeddings" not in path:
            modified_url = base_url.join("/embeddings")
        else:
            modified_url = base_url  # Handle other cases if needed

        # Re-add the original query parameters
        api_base = str(modified_url.copy_with(params=original_url.params))

        return api_base

    def _validate_environment(
        self,
        api_key: Optional[str],
        api_base: Optional[str],
        endpoint_type: Literal["chat_completions", "embeddings"],
        headers: Optional[dict],
        custom_endpoint: Optional[bool],
    ) -> Tuple[str, dict]:
        if api_key is None and headers is None:
            raise OpenAILikeError(
                status_code=400,
                message="Missing API Key - A call is being made to LLM Provider but no key is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
            )

        if api_base is None:
            raise OpenAILikeError(
                status_code=400,
                message="Missing API Base - A call is being made to LLM Provider but no api base is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params",
            )

        if headers is None:
            headers = {
                "Content-Type": "application/json",
            }

        if (
            api_key is not None and "Authorization" not in headers
        ):  # [TODO] remove 'validate_environment' from OpenAI base. should use llm providers config for this only.
            headers.update({"Authorization": "Bearer {}".format(api_key)})

        if not custom_endpoint:
            api_base = self._add_endpoint_to_api_base(
                api_base=api_base, endpoint_type=endpoint_type
            )
        return api_base, headers
