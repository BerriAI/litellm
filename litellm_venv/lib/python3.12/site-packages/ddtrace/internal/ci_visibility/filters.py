from typing import TYPE_CHECKING  # noqa:F401
from typing import Dict  # noqa:F401
from typing import List  # noqa:F401
from typing import Optional  # noqa:F401
from typing import Union  # noqa:F401

import ddtrace
from ddtrace.constants import AUTO_KEEP
from ddtrace.ext import SpanTypes
from ddtrace.ext import ci
from ddtrace.filters import TraceFilter


if TYPE_CHECKING:
    from ddtrace import Span  # noqa:F401


class TraceCiVisibilityFilter(TraceFilter):
    def __init__(self, tags, service):
        # type: (Dict[Union[str, bytes], str], str) -> None
        self._tags = tags
        self._service = service

    def process_trace(self, trace):
        # type: (List[Span]) -> Optional[List[Span]]
        if not trace:
            return trace

        local_root = trace[0]._local_root
        if not local_root or local_root.span_type != SpanTypes.TEST:
            return None

        local_root.context.dd_origin = ci.CI_APP_TEST_ORIGIN
        local_root.context.sampling_priority = AUTO_KEEP
        for span in trace:
            span.set_tags(self._tags)
            span.set_tag_str(ci.LIBRARY_VERSION, ddtrace.__version__)

        return trace
