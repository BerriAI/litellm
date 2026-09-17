from __future__ import annotations

from collections.abc import Callable, Coroutine, Mapping, Sequence
from types import MappingProxyType
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from pydantic import TypeAdapter

from litellm.llms.base_llm.ocr.transformation import PROVIDER_NATIVE_RESPONSE_KEY, OCRResponse
from litellm.rust_bridge.bindings import NativeBinding
from litellm.types.utils import CustomPricingLiteLLMParams


class OcrLoggingProtocol(Protocol):
    def update_from_kwargs(
        self,
        *,
        kwargs: dict[str, object],
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        custom_llm_provider: str,
    ) -> object: ...

    def pre_call(self, *, input: str, api_key: str | None, additional_args: dict[str, object]) -> object: ...

    def post_call(self, *, original_response: object, additional_args: dict[str, object]) -> object: ...


def _redact(params: Mapping[str, object], secret_fields: Sequence[str]) -> dict[str, object]:
    return {  # mutable-ok: update_from_kwargs takes dict
        name: "****" if name in secret_fields else value
        for name, value in params.items()
        if name != "proxy_server_request"
    }


def update_logging(
    logger: OcrLoggingProtocol,
    kwargs: Mapping[str, object],
    model: str,
    custom_llm_provider: str,
    optional_params: Mapping[str, object],
    secret_fields: Sequence[str],
    url: str,
) -> None:
    logger.update_from_kwargs(
        kwargs=_redact(kwargs, secret_fields),
        model=model,
        optional_params=_redact(optional_params, secret_fields),
        litellm_params={  # mutable-ok: update_from_kwargs takes dict
            "litellm_call_id": kwargs.get("litellm_call_id"),
            "api_base": url,
            **{
                name: kwargs[name] for name in ("logger_fn", "litellm_request_debug") if name in kwargs
            },  # mutable-ok: splat into the dict above
            **{  # mutable-ok: splat into the dict above
                name: kwargs[name]
                for name in CustomPricingLiteLLMParams.model_fields
                if name in kwargs and kwargs[name] is not None
            },
        },
        custom_llm_provider=custom_llm_provider,
    )


def pre_call(
    logger: OcrLoggingProtocol,
    api_key: str | None,
    body: dict[str, object],
    headers: dict[str, str],
    url: str,
) -> None:
    logger.pre_call(
        input="OCR document processing",
        api_key=api_key,
        additional_args={"complete_input_dict": body, "headers": headers, "api_base": url},
    )


def post_call(
    logger: OcrLoggingProtocol,
    original_response: object,
    body: dict[str, object],
    headers: dict[str, str],
) -> None:
    logger.post_call(
        original_response=original_response,
        additional_args={"complete_input_dict": body, "headers": headers},
    )


class RustOcr(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> OCRResponse: ...


class RustAocr(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> Coroutine[object, object, OCRResponse]: ...


def _as_ocr(value: object) -> RustOcr | None:
    return cast(RustOcr, value) if callable(value) else None


def _as_aocr(value: object) -> RustAocr | None:
    return cast(RustAocr, value) if callable(value) else None


NATIVE_OCR: Final = NativeBinding("ocr", validate=_as_ocr)
NATIVE_AOCR: Final = NativeBinding("aocr", validate=_as_aocr)
_NATIVE_RESPONSE: Final = TypeAdapter(Mapping[str, object])


def read_document(reader: Callable[[], object]) -> bytes:
    value: Final = reader()
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, bytes):
        return value
    raise TypeError(f"OCR file read must return bytes or str, got {type(value)}")


def build_response(response: Mapping[str, object]) -> OCRResponse:
    provider_native_response: Final = response.get(PROVIDER_NATIVE_RESPONSE_KEY)
    normalized: Final = OCRResponse.model_validate(
        MappingProxyType({key: value for key, value in response.items() if key != PROVIDER_NATIVE_RESPONSE_KEY})
    )
    if isinstance(provider_native_response, Mapping):
        normalized.set_provider_native_response(_NATIVE_RESPONSE.validate_python(provider_native_response))
    return normalized
