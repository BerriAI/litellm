from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str

CATALOG_URL: Final = "https://platform.acedata.cloud/api/v1/models/catalog/"
_CATALOG: Final = TypeAdapter(dict[str, object])
_ITEMS: Final = TypeAdapter(tuple[Mapping[str, object], ...])


def acedatacloud_native_endpoint(model: str, endpoint: str) -> bool:
    entry: Final = litellm.model_cost.get(f"acedatacloud/{model.removeprefix('acedatacloud/')}", {})
    return endpoint in entry.get("supported_endpoints", ())


def acedatacloud_uses_messages(model: str) -> bool:
    return acedatacloud_native_endpoint(model, "/v1/messages") and not acedatacloud_native_endpoint(
        model, "/v1/chat/completions"
    )


class AceDataCloudModelInfo(OpenAIGPTConfig):
    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("ACEDATACLOUD_API_KEY")

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return api_base or "https://api.acedata.cloud/v1"

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: provider discovery returns a list
        if api_base and api_base.rstrip("/") not in (
            "https://api.acedata.cloud",
            "https://api.acedata.cloud/v1",
            "https://api.acedata.cloud/openai",
        ):
            return OpenAIGPTConfig().get_models(
                api_key=api_key or get_secret_str("ACEDATACLOUD_API_KEY") or "", api_base=api_base
            )
        response: Final = litellm.module_level_client.get(url=CATALOG_URL)
        response.raise_for_status()
        catalog: Final = _CATALOG.validate_json(response.content)
        items: Final = _ITEMS.validate_python(catalog["items"])
        return [
            f"acedatacloud/{item['id']}"
            for item in items
            if item.get("modality") == "chat" and isinstance(item.get("id"), str)
        ]
