import abc
import re
from typing import TYPE_CHECKING  # noqa:F401
from typing import List  # noqa:F401
from typing import Optional  # noqa:F401
from typing import Union  # noqa:F401

from ddtrace._trace.processor import TraceProcessor
from ddtrace.ext import http


if TYPE_CHECKING:  # pragma: no cover
    from ddtrace._trace.span import Span  # noqa:F401


class TraceFilter(TraceProcessor):
    @abc.abstractmethod
    def process_trace(self, trace):
        # type: (List[Span]) -> Optional[List[Span]]
        """Processes a trace.

        None can be returned to prevent the trace from being exported.
        """
        pass


class FilterRequestsOnUrl(TraceFilter):
    r"""Filter out traces from incoming http requests based on the request's url.

    This class takes as argument a list of regular expression patterns
    representing the urls to be excluded from tracing. A trace will be excluded
    if its root span contains a ``http.url`` tag and if this tag matches any of
    the provided regular expression using the standard python regexp match
    semantic (https://docs.python.org/3/library/re.html#re.match).

    :param list regexps: a list of regular expressions (or a single string) defining
                         the urls that should be filtered out.

    Examples:
    To filter out http calls to domain api.example.com::

        FilterRequestsOnUrl(r'http://api\\.example\\.com')

    To filter out http calls to all first level subdomains from example.com::

        FilterRequestOnUrl(r'http://.*+\\.example\\.com')

    To filter out calls to both http://test.example.com and http://example.com/healthcheck::

        FilterRequestOnUrl([r'http://test\\.example\\.com', r'http://example\\.com/healthcheck'])
    """

    def __init__(self, regexps: Union[str, List[str]]):
        if isinstance(regexps, str):
            regexps = [regexps]
        self._regexps = [re.compile(regexp) for regexp in regexps]

    def process_trace(self, trace):
        # type: (List[Span]) -> Optional[List[Span]]
        """
        When the filter is registered in the tracer, process_trace is called by
        on each trace before it is sent to the agent, the returned value will
        be fed to the next filter in the list. If process_trace returns None,
        the whole trace is discarded.
        """
        for span in trace:
            url = span.get_tag(http.URL)
            if span.parent_id is None and url is not None:
                for regexp in self._regexps:
                    if regexp.match(url):
                        return None
        return trace
