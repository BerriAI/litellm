import re
from collections.abc import Mapping
from typing import Final
from urllib.parse import urlsplit

from pydantic import TypeAdapter, ValidationError

PASSTHROUGH_OPERATION_DATA: Final = TypeAdapter(Mapping[str, object])


def normalize_passthrough_operation(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized: Final = value.strip().lower()
    return normalized if re.fullmatch(r"[a-z][a-z0-9_.]{0,63}", normalized) else None


def _mapping(value: object) -> Mapping[str, object]:
    try:
        return PASSTHROUGH_OPERATION_DATA.validate_python(value)
    except ValidationError:
        return {}


def passthrough_header_operation(data: Mapping[str, object]) -> str | None:
    return normalize_passthrough_operation(data.get("passthrough_operation_name"))


def infer_passthrough_operation(data: Mapping[str, object]) -> str:
    inferred: Final = data.get("passthrough_inferred_operation")
    if isinstance(inferred, str) and inferred in ("chat", "embeddings", "text_completion", "litellm.pass_through"):
        return inferred
    url: Final = _mapping(data.get("passthrough_logging_payload")).get("url")
    try:
        path: Final = urlsplit(url if isinstance(url, str) else "").path.lower()
    except ValueError:
        return "litellm.pass_through"
    if path.endswith(
        (
            "/chat/completions",
            "/v1/messages",
            "/responses",
            ":generatecontent",
            ":streamgeneratecontent",
            "/converse",
            "/converse-stream",
            "/chat",
        )
    ):
        return "chat"
    if path.endswith(("/embeddings", ":embedcontent", ":batchembedcontents", "/embed")):
        return "embeddings"
    if path.endswith("/completions"):
        return "text_completion"
    return "litellm.pass_through"


def resolve_passthrough_operation(data: Mapping[str, object]) -> str:
    return passthrough_header_operation(data) or infer_passthrough_operation(data)
