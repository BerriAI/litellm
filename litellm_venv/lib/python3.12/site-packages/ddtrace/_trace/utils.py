from typing import Callable

from ddtrace.propagation.http import HTTPPropagator


def extract_DD_context_from_messages(messages, extract_from_message: Callable):
    ctx = None
    if len(messages) >= 1:
        message = messages[0]
        context_json = extract_from_message(message)
        if context_json is not None:
            child_of = HTTPPropagator.extract(context_json)
            if child_of.trace_id is not None:
                ctx = child_of
    return ctx
