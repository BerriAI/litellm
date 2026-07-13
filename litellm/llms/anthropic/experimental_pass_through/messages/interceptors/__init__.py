from typing import List

from .advisor import AdvisorOrchestrationHandler
from .base import MessagesInterceptor

_interceptors: List[MessagesInterceptor] = [
    AdvisorOrchestrationHandler(),
]


def get_messages_interceptors() -> List[MessagesInterceptor]:
    """Return the list of active MessagesInterceptors.

    Order matters: interceptors are tried in list order; the first one whose
    ``can_handle()`` returns True wins.
    """
    return _interceptors
