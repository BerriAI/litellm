from collections.abc import Mapping, Sequence

from pydantic import JsonValue

from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.mistral.common_utils import get_mistral_api_base
from litellm.llms.openai.responses.count_tokens.transformation import OpenAICountTokensConfig


class MistralCountTokensConfig(OpenAICountTokensConfig):
    def get_mistral_count_tokens_endpoint(self, api_base: str | None = None) -> str:
        return join_api_base(get_mistral_api_base(api_base), "v1/responses/input_tokens")

    def get_required_headers(
        self, api_key: str
    ) -> dict[str, str]:  # mutable-ok: matches OpenAICountTokensConfig contract
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def transform_request_to_count_tokens(
        self,
        model: str,
        input: str | Sequence[object],
        tools: Sequence[Mapping[str, object]] | None = None,
        instructions: str | None = None,
        text: Mapping[str, object] | None = None,
        tool_choice: JsonValue = None,
    ) -> dict[str, object]:  # mutable-ok: matches OpenAICountTokensConfig contract
        return {
            "model": model,
            "input": input,
            **({} if instructions is None else {"instructions": instructions}),
            **(
                {}
                if tools is None
                else {"tools": self._transform_tools_for_responses_api([dict(tool) for tool in tools])}
            ),
            **({} if text is None else {"text": dict(text)}),
            **({} if tool_choice is None else {"tool_choice": tool_choice}),
        }
