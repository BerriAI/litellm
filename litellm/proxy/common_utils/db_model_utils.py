from collections import deque
from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import BaseModel, TypeAdapter

from litellm._logging import verbose_proxy_logger

_MAPPING: Final = TypeAdapter(Mapping[str, object])
_SEQUENCE: Final = TypeAdapter(Sequence[object])


def strip_env_refs_for_team_model(
    litellm_params: Mapping[str, object], model_info: Mapping[str, object] | BaseModel | None
) -> Mapping[str, object]:
    info: Final = model_info.model_dump() if isinstance(model_info, BaseModel) else model_info
    if info is None or info.get("team_id") is None:
        return litellm_params

    # Private builders keep deep copying linear without recursion and leave caller inputs unchanged
    result: Final = dict[str, object]()
    pending: Final = deque[tuple[Mapping[str, object] | Sequence[object], dict[str, object] | list[object]]](
        ((litellm_params, result),)
    )

    def copy_value(value: object) -> object:
        if isinstance(value, str) and value.startswith("os.environ/"):
            verbose_proxy_logger.warning("Ignoring a proxy setting reference in scoped model parameters")
            return ""
        if isinstance(value, dict):
            child: Final = dict[str, object]()
            pending.append((_MAPPING.validate_python(value), child))
            return child
        if isinstance(value, list):
            items: Final = list[object]()
            pending.append((_SEQUENCE.validate_python(value), items))
            return items
        return value

    while pending:
        source, target = pending.popleft()
        if isinstance(source, Mapping) and isinstance(target, dict):
            target.update((key, copy_value(value)) for key, value in source.items())
        elif isinstance(source, Sequence) and isinstance(target, list):
            target.extend(copy_value(value) for value in source)
    return result
