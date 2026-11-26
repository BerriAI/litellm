import re
from collections.abc import Mapping
from typing import Final

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.model_cost_table import published

_REGION_CODE: Final = re.compile(r"([A-Za-z]{2})-([A-Za-z]{2})")
_V25_MODEL: Final = "prisma-v2.5"
_REALTIME_ENDPOINT: Final = "/v1/realtime"
V25_FIELDS_FLAG: Final = "gnani_v25_request_fields"


def _lists_realtime(row: Mapping[str, object]) -> bool:
    endpoints: Final = row.get("supported_endpoints")
    return row.get("litellm_provider") == "gnani" and isinstance(endpoints, list) and _REALTIME_ENDPOINT in endpoints


def supports_realtime(model: str) -> bool:
    from litellm.utils import catalog_with_runtime_registrations

    name: Final = model.removeprefix("gnani/")
    catalog: Final = catalog_with_runtime_registrations()
    return any(_lists_realtime(catalog.get(key) or {}) for key in (f"gnani/{name}", name))


def normalize_language(hint: str) -> str:
    if hint.lower() == "indic-auto":
        return "indic-auto"
    if len(hint) == 2:
        return f"{hint.lower()}-IN"
    region: Final = _REGION_CODE.fullmatch(hint)
    return f"{region[1].lower()}-{region[2].upper()}" if region else hint


def is_prisma_v25(model: str) -> bool:
    name: Final = model.removeprefix("gnani/")
    if (as_str_mapping(published().get(f"gnani/{name}")) or {}).get("litellm_provider") != "gnani":
        return name == _V25_MODEL
    flags: Final = litellm.get_model_info(model=name, custom_llm_provider="gnani").get("provider_specific_entry")
    return name == _V25_MODEL if flags is None else flags.get(V25_FIELDS_FLAG) == 1
