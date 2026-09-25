from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, TypeAlias, cast

from pydantic import JsonValue
from sentry_sdk.scrubber import EventScrubber
from typing_extensions import ReadOnly, TypedDict

from litellm.constants import SENTRY_DENYLIST, SENTRY_PII_DENYLIST
from litellm.secret_managers.main import str_to_bool

if TYPE_CHECKING:
    from sentry_sdk.types import Event, Hint

EventScrubFn: TypeAlias = "Callable[[Event, Hint], Event]"

FILTERED: Final = "[Filtered]"
SEND_DEFAULT_PII_ENV: Final = "SENTRY_SEND_DEFAULT_PII"

EMAIL_PATTERN: Final = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
SHA256_HEX_PATTERN: Final = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{64}(?![0-9A-Za-z])")
QUOTED_VALUE: Final = r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\""
BARE_VALUE: Final = r"(?!None(?![0-9A-Za-z_]))[^,)\]}\s]+"


class SentryInitOptions(TypedDict):
    dsn: ReadOnly[str | None]
    traces_sample_rate: ReadOnly[float]
    sample_rate: ReadOnly[float]
    send_default_pii: ReadOnly[bool]
    event_scrubber: ReadOnly[EventScrubber]
    before_send: ReadOnly[EventScrubFn]
    before_send_transaction: ReadOnly[EventScrubFn]
    environment: ReadOnly[str]


def build_repr_field_pattern(field_names: Sequence[str]) -> re.Pattern[str]:
    names: Final = "|".join(re.escape(name) for name in field_names)
    return re.compile(
        rf"(?P<field>(?<![0-9A-Za-z_])(?:{names})=|['\"](?:{names})['\"]:\s*)(?P<value>{QUOTED_VALUE}|{BARE_VALUE})",
        re.IGNORECASE,
    )


def build_string_scrubber(send_default_pii: bool) -> Callable[[str], str]:
    field_names: Final = (
        tuple(SENTRY_DENYLIST) if send_default_pii else tuple(SENTRY_DENYLIST) + tuple(SENTRY_PII_DENYLIST)
    )
    field_pattern: Final = build_repr_field_pattern(field_names)
    value_patterns: Final = () if send_default_pii else (EMAIL_PATTERN, SHA256_HEX_PATTERN)

    def scrub(text: str) -> str:
        fields_scrubbed: Final = field_pattern.sub(_filtered_field, text)
        return _substitute_all(value_patterns, fields_scrubbed)

    return scrub


def _filtered_field(match: re.Match[str]) -> str:
    quote: Final = '"' if match.group("value").startswith('"') else "'"
    return f"{match.group('field')}{quote}{FILTERED}{quote}"


def _substitute_all(patterns: Sequence[re.Pattern[str]], text: str) -> str:
    if not patterns:
        return text
    return _substitute_all(patterns[1:], patterns[0].sub(FILTERED, text))


def scrub_json_strings(value: JsonValue, scrub: Callable[[str], str]) -> JsonValue:
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {key: scrub_json_strings(item, scrub) for key, item in value.items()}  # mutable-ok: JSON object
    if isinstance(value, list):
        return [scrub_json_strings(item, scrub) for item in value]  # mutable-ok: JSON array
    return value


def build_event_scrubber(send_default_pii: bool) -> EventScrubFn:
    scrub: Final = build_string_scrubber(send_default_pii)

    def scrub_event(event: Event, _hint: Hint) -> Event:
        json_event: Final = cast("JsonValue", event)  # cast-ok: [LIT006] the SDK serialized the event to JSON already
        return cast("Event", scrub_json_strings(json_event, scrub))  # cast-ok: [LIT006] same JSON shape going back

    return scrub_event


def send_default_pii_from_env(env: Mapping[str, str]) -> bool:
    return str_to_bool(env.get(SEND_DEFAULT_PII_ENV)) is True


def build_sentry_init_options(env: Mapping[str, str]) -> SentryInitOptions:
    send_default_pii: Final = send_default_pii_from_env(env)
    scrub_event: Final = build_event_scrubber(send_default_pii)
    return SentryInitOptions(
        dsn=env.get("SENTRY_DSN"),
        traces_sample_rate=float(env.get("SENTRY_API_TRACE_RATE") or "1.0"),
        sample_rate=float(env.get("SENTRY_API_SAMPLE_RATE") or "1.0"),
        send_default_pii=send_default_pii,
        event_scrubber=EventScrubber(
            denylist=list(SENTRY_DENYLIST),  # mutable-ok: EventScrubber appends pii_denylist onto denylist in place
            pii_denylist=list(SENTRY_PII_DENYLIST),  # mutable-ok: EventScrubber takes List[str]
            recursive=True,
            send_default_pii=send_default_pii,
        ),
        before_send=scrub_event,
        before_send_transaction=scrub_event,
        environment=env.get("SENTRY_ENVIRONMENT", "production"),
    )
