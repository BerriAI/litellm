import re

import sentry_sdk
from sentry_sdk.integrations import Integration
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.utils import capture_internal_exceptions

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
    from sentry_sdk._types import Event


MODULE_RE = r"[a-zA-Z0-9/._:\\-]+"
TYPE_RE = r"[a-zA-Z0-9._:<>,-]+"
HEXVAL_RE = r"[A-Fa-f0-9]+"


FRAME_RE = r"""
^(?P<index>\d+)\.\s
(?P<package>{MODULE_RE})\(
  (?P<retval>{TYPE_RE}\ )?
  ((?P<function>{TYPE_RE})
    (?P<args>\(.*\))?
  )?
  ((?P<constoffset>\ const)?\+0x(?P<offset>{HEXVAL_RE}))?
\)\s
\[0x(?P<retaddr>{HEXVAL_RE})\]$
""".format(
    MODULE_RE=MODULE_RE, HEXVAL_RE=HEXVAL_RE, TYPE_RE=TYPE_RE
)

FRAME_RE = re.compile(FRAME_RE, re.MULTILINE | re.VERBOSE)


class GnuBacktraceIntegration(Integration):
    identifier = "gnu_backtrace"

    @staticmethod
    def setup_once():
        # type: () -> None
        @add_global_event_processor
        def process_gnu_backtrace(event, hint):
            # type: (Event, dict[str, Any]) -> Event
            with capture_internal_exceptions():
                return _process_gnu_backtrace(event, hint)


def _process_gnu_backtrace(event, hint):
    # type: (Event, dict[str, Any]) -> Event
    if sentry_sdk.get_client().get_integration(GnuBacktraceIntegration) is None:
        return event

    exc_info = hint.get("exc_info", None)

    if exc_info is None:
        return event

    exception = event.get("exception", None)

    if exception is None:
        return event

    values = exception.get("values", None)

    if values is None:
        return event

    for exception in values:
        frames = exception.get("stacktrace", {}).get("frames", [])
        if not frames:
            continue

        msg = exception.get("value", None)
        if not msg:
            continue

        additional_frames = []
        new_msg = []

        for line in msg.splitlines():
            match = FRAME_RE.match(line)
            if match:
                additional_frames.append(
                    (
                        int(match.group("index")),
                        {
                            "package": match.group("package") or None,
                            "function": match.group("function") or None,
                            "platform": "native",
                        },
                    )
                )
            else:
                # Put garbage lines back into message, not sure what to do with them.
                new_msg.append(line)

        if additional_frames:
            additional_frames.sort(key=lambda x: -x[0])
            for _, frame in additional_frames:
                frames.append(frame)

            new_msg.append("<stacktrace parsed and removed by GnuBacktraceIntegration>")
            exception["value"] = "\n".join(new_msg)

    return event
