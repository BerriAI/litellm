import os
from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

STORE_SETTINGS: Final = TypeAdapter(dict[str, object])


def is_lens_tracing_enabled(settings: object, environ: Mapping[str, str] = os.environ) -> bool:
    if environ.get("LITELLM_LENS_URL"):
        return True
    if not isinstance(settings, Mapping):
        return False
    store: Final = STORE_SETTINGS.validate_python(settings).get("store")
    return isinstance(store, Mapping) and STORE_SETTINGS.validate_python(store).get("type") == "lens"
