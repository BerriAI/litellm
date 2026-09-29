from collections.abc import Mapping
from typing import Final

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.sail.common_utils import (
    chat_request_for_sail,
    extra_body_for_sail,
    json_body,
    params_with_completion_window,
)
from litellm.types.llms.openai import AllMessageValues

_REJECTED_BY_SAIL: Final = frozenset(
    {"stop", "seed", "frequency_penalty", "presence_penalty", "logit_bias", "logprobs", "top_logprobs"}
)
_ACCEPTED_BY_SAIL: Final = ("reasoning_effort", "user", "metadata")


class SailChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: return type fixed by the base interface
        inherited: Final = tuple(
            param for param in super().get_supported_openai_params(model) if param not in _REJECTED_BY_SAIL
        )
        added: Final = tuple(param for param in _ACCEPTED_BY_SAIL if param not in inherited)
        return [*inherited, *added]  # mutable-ok: the base interface returns a list

    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: signature fixed by the base interface
        optional_params: dict,  # mutable-ok: signature fixed by the base interface
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: return type fixed by the base interface
        mapped: Final = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
        normalized: Final = params_with_completion_window(
            {key: non_default_params[key] for key in ("service_tier", "metadata") if key in non_default_params},
            model=model,
            drop_params=drop_params,
        )
        return {**mapped, **normalized}

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: signature fixed by the base interface
        optional_params: dict,  # mutable-ok: signature fixed by the base interface
        litellm_params: dict,  # mutable-ok: signature fixed by the base interface
        headers: dict,  # mutable-ok: signature fixed by the base interface
    ) -> dict:  # mutable-ok: return type fixed by the base interface
        request: Final = chat_request_for_sail(
            super().transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            ),
            model=model,
            drop_params=bool(litellm_params.get("drop_params")),
        )
        return json_body(request)

    def transform_extra_body(
        self,
        extra_body: Mapping[str, object],
        request: Mapping[str, object],
        model: str,
        litellm_params: Mapping[str, object],
    ) -> Mapping[str, object]:
        return extra_body_for_sail(
            extra_body, request.get("metadata"), model=model, drop_params=bool(litellm_params.get("drop_params"))
        )
