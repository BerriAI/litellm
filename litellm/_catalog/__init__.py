"""Internal catalog loading and publication."""

import asyncio
import os
from dataclasses import dataclass
from typing import Final

from litellm._catalog._fetch import Bundled, CatalogUnavailable, Remote, Source, fetch
from litellm._catalog._install import Provenance, install
from litellm.litellm_core_utils.offline_mode import is_offline_mode


@dataclass(frozen=True, slots=True)
class Reloaded:
    models_count: int


def is_priced(model: str, provider: str | None = None) -> bool:
    from litellm._catalog import _read

    return _read.is_priced(model, provider)


def reload_source(url: str) -> Source:
    if not url or is_offline_mode() or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP", "").lower() == "true":
        return Bundled()
    return Remote(url)


async def reload(url: str, *, max_attempts: int = 3, timeout: float = 5.0) -> Reloaded | CatalogUnavailable:
    from litellm.litellm_core_utils.get_model_cost_map import git_blob_id
    from litellm.rust_bridge._native import ModelRegistry

    source: Final = reload_source(url)
    result: Final = await asyncio.to_thread(fetch, source, max_attempts=max_attempts, timeout=timeout)
    if isinstance(result, CatalogUnavailable):
        return result
    local: Final = isinstance(source, Bundled)
    count: Final = install(
        result,
        provenance=Provenance(
            source="local" if local else "remote",
            url=None if local else url,
            is_env_forced=local,
            revision=result.registry.source_revision or git_blob_id(ModelRegistry.bundled_bytes()),
            etag=result.registry.source_validator,
        ),
    )
    return Reloaded(count) if count else CatalogUnavailable("Model cost map failed integrity validation")
