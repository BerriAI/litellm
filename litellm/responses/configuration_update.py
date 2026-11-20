from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast  # noqa: TID251  # an update's settings are forwarded as sent, like the request's

from pydantic import TypeAdapter, ValidationError

from litellm.types.llms.openai import ResponseInputParam, ResponsesAPIOptionalRequestParams

_SETTINGS: Final = TypeAdapter(Mapping[str, object])


@dataclass(frozen=True, slots=True)
class ConfiguredRequest:
    input: str | ResponseInputParam
    responses_api_request: ResponsesAPIOptionalRequestParams


def _is_configuration_update(item: object) -> bool:
    match item:
        case {"type": "configuration_update"}:
            return True
        case _:
            return False


def _settings(value: object) -> Mapping[str, object]:
    try:
        return _SETTINGS.validate_python(value)
    except ValidationError:
        return {}


def _section_updated(
    request: Mapping[str, object], updates: Sequence[Mapping[str, object]], section: str, setting: str
) -> Mapping[str, object]:
    values: Final = tuple(
        value for update in updates if (value := _settings(update.get(section)).get(setting)) is not None
    )
    return {section: {**_settings(request.get(section)), setting: values[-1]}} if values else {}


def with_configuration_updates_applied(
    input: str | ResponseInputParam,
    responses_api_request: ResponsesAPIOptionalRequestParams,
    earlier_items: Sequence[object] = (),
) -> ConfiguredRequest:
    current_items: Final = () if isinstance(input, str) else tuple(input)
    updates: Final = tuple(
        _settings(item) for item in (*earlier_items, *current_items) if _is_configuration_update(item)
    )
    if not updates:
        return ConfiguredRequest(input=input, responses_api_request=responses_api_request)
    request: Final[Mapping[str, object]] = responses_api_request
    return ConfiguredRequest(
        input=input if isinstance(input, str) else [item for item in input if not _is_configuration_update(item)],
        responses_api_request=cast(  # cast-ok: an update's effort and format are forwarded as sent
            ResponsesAPIOptionalRequestParams,
            {
                **request,
                **_section_updated(request, updates, "reasoning", "effort"),
                **_section_updated(request, updates, "text", "format"),
            },
        ),
    )
