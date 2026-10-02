from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal, TypeAlias

from pydantic import AnyHttpUrl, BaseModel, TypeAdapter, ValidationError

from litellm.secret_managers.main import get_secret_str

LayaCheckpoint: TypeAlias = Literal["english", "multilingual", "typed-decisions"]


def validate_laya_model(value: object) -> LayaCheckpoint:
    try:
        return TypeAdapter(LayaCheckpoint).validate_python(value)
    except ValidationError as exc:
        raise ValueError("Laya model must be 'english', 'multilingual', or 'typed-decisions'") from exc


def validate_laya_request(body: Mapping[str, object]) -> LayaCheckpoint:
    if "custom_body" in body:
        raise ValueError("custom_body is not supported for Laya requests")
    if body.get("stream"):
        raise ValueError("Streaming is not supported for Laya requests")
    return validate_laya_model(body.get("model"))


@dataclass(frozen=True, slots=True)
class LayaConnection:
    api_base: str
    api_key: str | None = field(repr=False)


def validate_laya_api_base(value: str) -> str:
    try:
        url: Final = TypeAdapter(AnyHttpUrl).validate_python(value)
    except ValidationError as exc:
        raise ValueError("Laya api_base must be an HTTP or HTTPS server URL") from exc
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("Laya api_base must not contain credentials, a query, or a fragment")
    return str(url).rstrip("/")


def laya_connection(api_base: str | None = None, api_key: str | None = None) -> LayaConnection:
    base: Final = api_base if api_base is not None else get_secret_str("LAYA_API_BASE")
    if not base:
        raise ValueError("Laya requires api_base or LAYA_API_BASE pointing to a self-hosted server")
    key: Final = api_key if api_base is not None else api_key or get_secret_str("LAYA_API_KEY")
    return LayaConnection(api_base=validate_laya_api_base(base), api_key=key)


class _LayaRouting(BaseModel):
    model: str | None = None


def laya_response_model(response: Mapping[str, object], requested_model: str | None) -> str:
    try:
        routing: Final = TypeAdapter(_LayaRouting).validate_python(response.get("routing") or _LayaRouting())
    except ValidationError:
        return requested_model or "unknown"
    return routing.model or requested_model or "unknown"
