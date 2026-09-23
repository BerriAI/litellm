import json
from collections.abc import Mapping
from typing import Final

OWNED_KEYS: Final[frozenset[str]] = frozenset(
    (
        "user_api_key_hash",
        "user_api_key_alias",
        "user_api_key_team_id",
        "user_api_key_user_id",
        "user_api_key_org_id",
        "user_api_key_end_user_id",
        "litellm_api_version",
        "litellm_call_id",
        "litellm_logging_obj",
        "litellm_metadata",
        "litellm_trace_id",
        "proxy_server_request",
        "global_max_parallel_requests",
        "model_info",
    )
)
OWNED_PREFIX: Final = "_litellm_"


def is_owned_key(key: str) -> bool:
    return key in OWNED_KEYS or key.startswith(OWNED_PREFIX)


def parse_request_body(body: object) -> Mapping[str, object] | None:
    if isinstance(body, Mapping):
        return body
    if not isinstance(body, (str, bytes)):
        return None
    try:
        parsed: Final = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def request_body_view(body: Mapping[str, object]) -> Mapping[str, object]:
    extra: Final = body.get("extra_body")
    if not isinstance(extra, Mapping):
        return body
    return {**{key: value for key, value in body.items() if key != "extra_body"}, **extra}


def owned_keys_in(body: Mapping[str, object]) -> tuple[str, ...]:
    view: Final = request_body_view(body)
    top_level: Final = (key for key in view if isinstance(key, str) and is_owned_key(key))
    nested: Final = (
        f"{container}.{key}"
        for container, inner in view.items()
        if isinstance(container, str) and isinstance(inner, Mapping)
        for key in inner
        if isinstance(key, str) and is_owned_key(key)
    )
    return tuple(sorted({*top_level, *nested}))


def loggable_owned_keys(flagged: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                leaf if leaf in OWNED_KEYS else f"{OWNED_PREFIX}*"
                for flagged_key in flagged
                for leaf in (flagged_key.rsplit(".", 1)[-1],)
            }
        )
    )


def log_safe(value: object) -> str:
    return str(value).replace("\r", "\\r").replace("\n", "\\n")
