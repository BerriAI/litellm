from ddtrace import config
from ddtrace._trace.processor import TraceProcessor
from ddtrace.constants import BASE_SERVICE_KEY

from . import schematize_service_name


class BaseServiceProcessor(TraceProcessor):
    def __init__(self):
        self._global_service = schematize_service_name((config.service or "").lower())

    def process_trace(self, trace):
        if not trace:
            return

        traces_to_process = filter(
            lambda x: x.service and x.service.lower() != self._global_service,
            trace,
        )
        any(map(lambda x: self._update_dd_base_service(x), traces_to_process))

        return trace

    def _update_dd_base_service(self, span):
        span.set_tag_str(key=BASE_SERVICE_KEY, value=self._global_service)
