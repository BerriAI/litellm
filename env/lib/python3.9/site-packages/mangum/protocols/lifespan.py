import asyncio
import enum
import logging
from types import TracebackType
from typing import Optional, Type

from mangum.types import ASGI, LifespanMode, Message
from mangum.exceptions import LifespanUnsupported, LifespanFailure, UnexpectedMessage


class LifespanCycleState(enum.Enum):
    """
    The state of the ASGI `lifespan` connection.

    * **CONNECTING** - Initial state. The ASGI application instance will be run with
    the connection scope containing the `lifespan` type.
    * **STARTUP** - The lifespan startup event has been pushed to the queue to be
    received by the application.
    * **SHUTDOWN** - The lifespan shutdown event has been pushed to the queue to be
    received by the application.
    * **FAILED** - A lifespan failure has been detected, and the connection will be
    closed with an error.
    * **UNSUPPORTED** - An application attempted to send a message before receiving
    the lifepan startup event. If the lifespan argument is "on", then the connection
    will be closed with an error.
    """

    CONNECTING = enum.auto()
    STARTUP = enum.auto()
    SHUTDOWN = enum.auto()
    FAILED = enum.auto()
    UNSUPPORTED = enum.auto()


class LifespanCycle:
    """
    Manages the application cycle for an ASGI `lifespan` connection.

    * **app** - An asynchronous callable that conforms to version 3.0 of the ASGI
    specification. This will usually be an ASGI framework application instance.
    * **lifespan** - A string to configure lifespan support. Choices are `auto`, `on`,
    and `off`. Default is `auto`.
    * **state** - An enumerated `LifespanCycleState` type that indicates the state of
    the ASGI connection.
    * **exception** - An exception raised while handling the ASGI event. This may or
    may not be raised depending on the state.
    * **app_queue** - An asyncio queue (FIFO) containing messages to be received by the
    application.
    * **startup_event** - An asyncio event object used to control the application
    startup flow.
    * **shutdown_event** - An asyncio event object used to control the application
    shutdown flow.
    """

    def __init__(self, app: ASGI, lifespan: LifespanMode) -> None:
        self.app = app
        self.lifespan = lifespan
        self.state: LifespanCycleState = LifespanCycleState.CONNECTING
        self.exception: Optional[BaseException] = None
        self.loop = asyncio.get_event_loop()
        self.app_queue: asyncio.Queue[Message] = asyncio.Queue()
        self.startup_event: asyncio.Event = asyncio.Event()
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.logger = logging.getLogger("mangum.lifespan")

    def __enter__(self) -> None:
        """Runs the event loop for application startup."""
        self.loop.create_task(self.run())
        self.loop.run_until_complete(self.startup())

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Runs the event loop for application shutdown."""
        self.loop.run_until_complete(self.shutdown())

    async def run(self) -> None:
        """Calls the application with the `lifespan` connection scope."""
        try:
            await self.app(
                {"type": "lifespan", "asgi": {"spec_version": "2.0", "version": "3.0"}},
                self.receive,
                self.send,
            )
        except LifespanUnsupported:
            self.logger.info("ASGI 'lifespan' protocol appears unsupported.")
        except (LifespanFailure, UnexpectedMessage) as exc:
            self.exception = exc
        except BaseException as exc:
            self.logger.error("Exception in 'lifespan' protocol.", exc_info=exc)
        finally:
            self.startup_event.set()
            self.shutdown_event.set()

    async def receive(self) -> Message:
        """Awaited by the application to receive ASGI `lifespan` events."""
        if self.state is LifespanCycleState.CONNECTING:

            # Connection established. The next event returned by the queue will be
            # `lifespan.startup` to inform the application that the connection is
            # ready to receive lfiespan messages.
            self.state = LifespanCycleState.STARTUP

        elif self.state is LifespanCycleState.STARTUP:

            # Connection shutting down. The next event returned by the queue will be
            # `lifespan.shutdown` to inform the application that the connection is now
            # closing so that it may perform cleanup.
            self.state = LifespanCycleState.SHUTDOWN

        return await self.app_queue.get()

    async def send(self, message: Message) -> None:
        """Awaited by the application to send ASGI `lifespan` events."""
        message_type = message["type"]
        self.logger.info(
            "%s:  '%s' event received from application.", self.state, message_type
        )

        if self.state is LifespanCycleState.CONNECTING:
            if self.lifespan == "on":
                raise LifespanFailure(
                    "Lifespan connection failed during startup and lifespan is 'on'."
                )

            # If a message is sent before the startup event is received by the
            # application, then assume that lifespan is unsupported.
            self.state = LifespanCycleState.UNSUPPORTED
            raise LifespanUnsupported("Lifespan protocol appears unsupported.")

        if message_type not in (
            "lifespan.startup.complete",
            "lifespan.shutdown.complete",
            "lifespan.startup.failed",
            "lifespan.shutdown.failed",
        ):
            self.state = LifespanCycleState.FAILED
            raise UnexpectedMessage(f"Unexpected '{message_type}' event received.")

        if self.state is LifespanCycleState.STARTUP:
            if message_type == "lifespan.startup.complete":
                self.startup_event.set()
            elif message_type == "lifespan.startup.failed":
                self.state = LifespanCycleState.FAILED
                self.startup_event.set()
                message_value = message.get("message", "")
                raise LifespanFailure(f"Lifespan startup failure. {message_value}")

        elif self.state is LifespanCycleState.SHUTDOWN:
            if message_type == "lifespan.shutdown.complete":
                self.shutdown_event.set()
            elif message_type == "lifespan.shutdown.failed":
                self.state = LifespanCycleState.FAILED
                self.shutdown_event.set()
                message_value = message.get("message", "")
                raise LifespanFailure(f"Lifespan shutdown failure. {message_value}")

    async def startup(self) -> None:
        """Pushes the `lifespan` startup event to the queue and handles errors."""
        self.logger.info("Waiting for application startup.")
        await self.app_queue.put({"type": "lifespan.startup"})
        await self.startup_event.wait()
        if self.state is LifespanCycleState.FAILED:
            raise LifespanFailure(self.exception)

        if not self.exception:
            self.logger.info("Application startup complete.")
        else:
            self.logger.info("Application startup failed.")

    async def shutdown(self) -> None:
        """Pushes the `lifespan` shutdown event to the queue and handles errors."""
        self.logger.info("Waiting for application shutdown.")
        await self.app_queue.put({"type": "lifespan.shutdown"})
        await self.shutdown_event.wait()
        if self.state is LifespanCycleState.FAILED:
            raise LifespanFailure(self.exception)
