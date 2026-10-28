import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

from litellm._catalog import CATALOG, Catalog, Installed, Parsed, Unavailable
from litellm.litellm_core_utils.get_model_cost_map import reload_model_cost_map

_LOCAL_MAP_FLAGS = ("LITELLM_LOCAL_MODEL_COST_MAP", "LITELLM_OFFLINE_MODE")


@contextlib.contextmanager
def _remote_loads_allowed() -> Iterator[None]:
    """The unit conftest forces the bundled map, which `Catalog.load` honors before it reads a URL"""
    with mock.patch.dict(os.environ):
        for flag in _LOCAL_MAP_FLAGS:
            os.environ.pop(flag, None)
        yield


def load_file(path: Path, catalog: Catalog = CATALOG) -> Parsed | Unavailable:
    with _remote_loads_allowed():
        return catalog.load(path.as_uri(), timeout=5, attempts=1)


def reload_from_file(path: Path, catalog: Catalog = CATALOG) -> Installed | Unavailable:
    with _remote_loads_allowed():
        return reload_model_cost_map(path.as_uri(), catalog)
