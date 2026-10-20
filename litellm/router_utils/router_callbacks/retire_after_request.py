from collections.abc import Callable

from litellm.integrations.custom_logger import CustomLogger


class RetireRoutingStateAfterRequest(CustomLogger):
    def __init__(self, retire: Callable[[], None]) -> None:
        super().__init__()
        self._retire = retire

    async def async_log_success_event(
        self, kwargs: object, response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._retire()

    async def async_log_failure_event(
        self, kwargs: object, response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._retire()
