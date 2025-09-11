import ast

import sentry_sdk
from sentry_sdk import serializer
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.utils import walk_exception_chain, iter_stacks

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Optional, Dict, Any, Tuple, List
    from types import FrameType

    from sentry_sdk._types import Event, Hint

try:
    import executing
except ImportError:
    raise DidNotEnable("executing is not installed")

try:
    import pure_eval
except ImportError:
    raise DidNotEnable("pure_eval is not installed")

try:
    # Used implicitly, just testing it's available
    import asttokens  # noqa
except ImportError:
    raise DidNotEnable("asttokens is not installed")


class PureEvalIntegration(Integration):
    identifier = "pure_eval"

    @staticmethod
    def setup_once():
        # type: () -> None

        @add_global_event_processor
        def add_executing_info(event, hint):
            # type: (Event, Optional[Hint]) -> Optional[Event]
            if sentry_sdk.get_client().get_integration(PureEvalIntegration) is None:
                return event

            if hint is None:
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

            for exception, (_exc_type, _exc_value, exc_tb) in zip(
                reversed(values), walk_exception_chain(exc_info)
            ):
                sentry_frames = [
                    frame
                    for frame in exception.get("stacktrace", {}).get("frames", [])
                    if frame.get("function")
                ]
                tbs = list(iter_stacks(exc_tb))
                if len(sentry_frames) != len(tbs):
                    continue

                for sentry_frame, tb in zip(sentry_frames, tbs):
                    sentry_frame["vars"] = (
                        pure_eval_frame(tb.tb_frame) or sentry_frame["vars"]
                    )
            return event


def pure_eval_frame(frame):
    # type: (FrameType) -> Dict[str, Any]
    source = executing.Source.for_frame(frame)
    if not source.tree:
        return {}

    statements = source.statements_at_line(frame.f_lineno)
    if not statements:
        return {}

    scope = stmt = list(statements)[0]
    while True:
        # Get the parent first in case the original statement is already
        # a function definition, e.g. if we're calling a decorator
        # In that case we still want the surrounding scope, not that function
        scope = scope.parent
        if isinstance(scope, (ast.FunctionDef, ast.ClassDef, ast.Module)):
            break

    evaluator = pure_eval.Evaluator.from_frame(frame)
    expressions = evaluator.interesting_expressions_grouped(scope)

    def closeness(expression):
        # type: (Tuple[List[Any], Any]) -> Tuple[int, int]
        # Prioritise expressions with a node closer to the statement executed
        # without being after that statement
        # A higher return value is better - the expression will appear
        # earlier in the list of values and is less likely to be trimmed
        nodes, _value = expression

        def start(n):
            # type: (ast.expr) -> Tuple[int, int]
            return (n.lineno, n.col_offset)

        nodes_before_stmt = [
            node for node in nodes if start(node) < stmt.last_token.end  # type: ignore
        ]
        if nodes_before_stmt:
            # The position of the last node before or in the statement
            return max(start(node) for node in nodes_before_stmt)
        else:
            # The position of the first node after the statement
            # Negative means it's always lower priority than nodes that come before
            # Less negative means closer to the statement and higher priority
            lineno, col_offset = min(start(node) for node in nodes)
            return (-lineno, -col_offset)

    # This adds the first_token and last_token attributes to nodes
    atok = source.asttokens()

    expressions.sort(key=closeness, reverse=True)
    vars = {
        atok.get_text(nodes[0]): value
        for nodes, value in expressions[: serializer.MAX_DATABAG_BREADTH]
    }
    return serializer.serialize(vars, is_vars=True)
