from __future__ import annotations

import fcntl
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Final

import yaml
from models import LiteLLMParamsBody
from pydantic import BaseModel, ConfigDict, TypeAdapter

PROXY_MODELS_PATH: Final = Path(__file__).resolve().parent / "proxy_models.yaml"
COLUMN_LOCK_DIR: Final = Path(tempfile.gettempdir()) / "litellm-codex-compat-columns"


class Deployment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str
    litellm_params: LiteLLMParamsBody


COLUMNS: Final = TypeAdapter(Mapping[str, tuple[Deployment, ...]])


@cache
def load_columns(path: Path = PROXY_MODELS_PATH) -> Mapping[str, tuple[Deployment, ...]]:
    return COLUMNS.validate_python(yaml.safe_load(path.read_text()))


def column_models(provider: str) -> tuple[str, ...]:
    return tuple(deployment.model_name for deployment in load_columns()[provider])


@contextmanager
def column_lock(provider: str, lock_dir: Path = COLUMN_LOCK_DIR) -> Generator[None]:
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{provider}.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
