from typing import Final

from .advisor import AdvisorOrchestrationHandler
from .base import MessagesInterceptor

_interceptors: Final[list[MessagesInterceptor]] = [
    AdvisorOrchestrationHandler(),
]


def get_messages_interceptors() -> list[MessagesInterceptor]:
    """Return the list of active MessagesInterceptors.

    Order matters: interceptors are tried in list order; the first one whose
    ``can_handle()`` returns True wins.
    """
    return _interceptors
