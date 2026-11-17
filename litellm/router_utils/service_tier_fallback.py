from typing import Final

from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.hidden_params import get_hidden_params, set_hidden_params


class FlexTierFallback:
    @staticmethod
    def add_header(response: object) -> object:
        hidden: Final = get_hidden_params(response) or {}
        headers: Final = as_str_mapping(hidden.get("additional_headers")) or {}
        set_hidden_params(response, {**hidden, "additional_headers": {**headers, "x-litellm-service-tier": "default"}})
        return response
