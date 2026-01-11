import asyncio

from .events_writer import AsyncEventsWriter
from .transformer import extract_events_from_log
from .logging import get_logger


logger = get_logger(__name__)


class Callback:
    """
    This class implements the callback interface of LiteLLM, in order to turn
    their "Standard Logging Object" into "Amberflo metering events" and ingest
    them asynchronously.

    See: https://docs.litellm.ai/docs/proxy/logging
    """

    __name__ = "amberflo-callback"

    def __init__(self, writer: AsyncEventsWriter):
        self.writer = writer
        logger.debug("Callback initialized")

    async def __call__(self, *args, **kwargs) -> None:
        """
        The main LiteLLM callback entrypoint for the purpose of handling standard logging objects.
        """

        # Find the standard_logging_object
        log = None
        if args and isinstance(args[0], dict):
            log = args[0].get("standard_logging_object")

        # Process the standard logging object
        if log:
            self._handle_log_object(log)

    async def async_post_call_success_hook(self, *args, **kwargs) -> None:
        """
        This is needed by LiteLLM.
        """
        pass

    def _handle_log_object(self, log):
        logger.debug("Handling log object: %s", log)

        events = extract_events_from_log(log)

        if events:
            # Avoid blocking
            asyncio.create_task(self.writer.async_write(events))
