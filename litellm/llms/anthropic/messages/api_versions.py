import re
from collections.abc import Mapping
from typing import Final

from litellm.exceptions import UnsupportedParamsError

_VERSION_HEADER: Final = "anthropic-version"
_SAMPLING_PARAMS: Final = ("temperature", "top_p", "top_k")
_FIRST_VERSION_WITHOUT_SAMPLING_PARAMS: Final = "2026-10-21"
_DATED_VERSION: Final = re.compile(r"\d{4}-\d{2}-\d{2}")


def requested_api_version(headers: Mapping[str, object]) -> str | None:
    requested: Final = next((value for name, value in headers.items() if name.lower() == _VERSION_HEADER), None)
    return requested.strip() if isinstance(requested, str) else None


def removes_sampling_params(api_version: str | None) -> bool:
    return (
        api_version is not None
        and _DATED_VERSION.fullmatch(api_version) is not None
        and api_version >= _FIRST_VERSION_WITHOUT_SAMPLING_PARAMS
    )


def sampling_params_to_drop(
    params: Mapping[str, object],
    api_version: str | None,
    model: str,
    provider: str,
    drop_params: bool,
) -> tuple[str, ...] | UnsupportedParamsError:
    if not removes_sampling_params(api_version):
        return ()
    sent: Final = tuple(name for name in _SAMPLING_PARAMS if params.get(name) is not None)
    if not sent or drop_params:
        return sent
    named: Final = ", ".join(f"{name}={params[name]}" for name in sent)
    return UnsupportedParamsError(
        message=(
            f"{model} does not support {named} under anthropic-version {api_version}, which removed "
            "temperature, top_p and top_k. To drop unsupported params, set `litellm.drop_params = True`."
        ),
        llm_provider=provider,
        model=model,
    )
