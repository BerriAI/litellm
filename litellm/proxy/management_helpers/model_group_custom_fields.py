import math
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import TypeAdapter

CustomMetadataValue: TypeAlias = str | int | float | bool

_OBJECT_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[object, object])
_OBJECT_LIST_ADAPTER: Final = TypeAdapter(list[object])


def _string_tags(tags: object) -> Iterator[str]:
    if isinstance(tags, list):
        yield from (tag for tag in _OBJECT_LIST_ADAPTER.validate_python(tags) if isinstance(tag, str))


def _is_non_finite(value: object) -> bool:
    return isinstance(value, float) and not math.isfinite(value)


def _scalar_metadata(metadata: object) -> Iterator[tuple[str, CustomMetadataValue]]:
    if isinstance(metadata, dict):
        yield from (
            (key, value)
            for key, value in _OBJECT_MAPPING_ADAPTER.validate_python(metadata).items()
            if isinstance(key, str) and isinstance(value, (str, int, float, bool)) and not _is_non_finite(value)
        )


def _custom_tag_values(model_infos: Sequence[Mapping[object, object]]) -> Iterator[str]:
    for model_info in model_infos:
        yield from _string_tags(model_info.get("custom_tags"))


def _custom_metadata_entries(
    model_infos: Sequence[Mapping[object, object]],
) -> Iterator[tuple[str, CustomMetadataValue]]:
    for model_info in model_infos:
        yield from _scalar_metadata(model_info.get("custom_metadata"))


def _merged_custom_tags(model_infos: Sequence[Mapping[object, object]]) -> tuple[str, ...] | None:
    return tuple(dict.fromkeys(_custom_tag_values(model_infos))) or None


def _merged_custom_metadata(
    model_infos: Sequence[Mapping[object, object]],
) -> Mapping[str, CustomMetadataValue] | None:
    entries: Final = tuple(_custom_metadata_entries(model_infos))
    first_seen: Final = MappingProxyType(dict(reversed(entries)))
    return MappingProxyType({key: first_seen[key] for key in dict.fromkeys(key for key, _ in entries)}) or None


def _model_info(deployment: Mapping[str, object]) -> Mapping[object, object]:
    model_info: Final = deployment.get("model_info")
    return _OBJECT_MAPPING_ADAPTER.validate_python(model_info) if isinstance(model_info, dict) else MappingProxyType({})


def model_group_custom_fields(deployments: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    model_infos: Final = tuple(_model_info(deployment) for deployment in deployments)
    return MappingProxyType(
        {
            "custom_tags": _merged_custom_tags(model_infos),
            "custom_metadata": _merged_custom_metadata(model_infos),
        }
    )
