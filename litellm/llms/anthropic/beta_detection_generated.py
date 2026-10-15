from collections.abc import Mapping, Sequence

from pydantic import JsonValue


def _field(value: JsonValue, name: str) -> JsonValue:
    return value.get(name) if isinstance(value, dict) else None


def _items(value: JsonValue) -> Sequence[JsonValue]:
    return value if isinstance(value, list) else ()


def detect_anthropic_betas(request: Mapping[str, JsonValue]) -> frozenset[str]:
    return frozenset(
        beta
        for used, beta in (
            (
                (
                    _field(request.get("thinking"), "type")
                    in (
                        "enabled",
                        "adaptive",
                    )
                    and _field(request.get("thinking"), "display") in ("updates",)
                ),
                "thinking-display-updates-2026-08-18",
            ),
            (
                any(
                    (
                        _field(message, "role") in ("system",)
                        and any(
                            (
                                _field(block, "type")
                                in (
                                    "tool_addition",
                                    "tool_removal",
                                )
                                and _field(_field(block, "tool"), "type") in ("tool_reference",)
                            )
                            for block in _items(_field(message, "content"))
                        )
                    )
                    for message in _items(request.get("messages"))
                ),
                "mid-conversation-tool-changes-2026-07-01",
            ),
        )
        if used
    )
