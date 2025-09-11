from __future__ import annotations

import asyncio
import codecs
import collections
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Generic,
    Iterable,
    TypeVar,
)

from ..exceptions import ConcurrencyError
from ..frames import OP_BINARY, OP_CONT, OP_TEXT, Frame
from ..typing import Data


__all__ = ["Assembler"]

UTF8Decoder = codecs.getincrementaldecoder("utf-8")

T = TypeVar("T")


class SimpleQueue(Generic[T]):
    """
    Simplified version of :class:`asyncio.Queue`.

    Provides only the subset of functionality needed by :class:`Assembler`.

    """

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.get_waiter: asyncio.Future[None] | None = None
        self.queue: collections.deque[T] = collections.deque()

    def __len__(self) -> int:
        return len(self.queue)

    def put(self, item: T) -> None:
        """Put an item into the queue without waiting."""
        self.queue.append(item)
        if self.get_waiter is not None and not self.get_waiter.done():
            self.get_waiter.set_result(None)

    async def get(self) -> T:
        """Remove and return an item from the queue, waiting if necessary."""
        if not self.queue:
            if self.get_waiter is not None:
                raise ConcurrencyError("get is already running")
            self.get_waiter = self.loop.create_future()
            try:
                await self.get_waiter
            finally:
                self.get_waiter.cancel()
                self.get_waiter = None
        return self.queue.popleft()

    def reset(self, items: Iterable[T]) -> None:
        """Put back items into an empty, idle queue."""
        assert self.get_waiter is None, "cannot reset() while get() is running"
        assert not self.queue, "cannot reset() while queue isn't empty"
        self.queue.extend(items)

    def abort(self) -> None:
        if self.get_waiter is not None and not self.get_waiter.done():
            self.get_waiter.set_exception(EOFError("stream of frames ended"))
        # Clear the queue to avoid storing unnecessary data in memory.
        self.queue.clear()


class Assembler:
    """
    Assemble messages from frames.

    :class:`Assembler` expects only data frames. The stream of frames must
    respect the protocol; if it doesn't, the behavior is undefined.

    Args:
        pause: Called when the buffer of frames goes above the high water mark;
            should pause reading from the network.
        resume: Called when the buffer of frames goes below the low water mark;
            should resume reading from the network.

    """

    # coverage reports incorrectly: "line NN didn't jump to the function exit"
    def __init__(  # pragma: no cover
        self,
        high: int = 16,
        low: int | None = None,
        pause: Callable[[], Any] = lambda: None,
        resume: Callable[[], Any] = lambda: None,
    ) -> None:
        # Queue of incoming messages. Each item is a queue of frames.
        self.frames: SimpleQueue[Frame] = SimpleQueue()

        # We cannot put a hard limit on the size of the queue because a single
        # call to Protocol.data_received() could produce thousands of frames,
        # which must be buffered. Instead, we pause reading when the buffer goes
        # above the high limit and we resume when it goes under the low limit.
        if low is None:
            low = high // 4
        if low < 0:
            raise ValueError("low must be positive or equal to zero")
        if high < low:
            raise ValueError("high must be greater than or equal to low")
        self.high, self.low = high, low
        self.pause = pause
        self.resume = resume
        self.paused = False

        # This flag prevents concurrent calls to get() by user code.
        self.get_in_progress = False

        # This flag marks the end of the connection.
        self.closed = False

    async def get(self, decode: bool | None = None) -> Data:
        """
        Read the next message.

        :meth:`get` returns a single :class:`str` or :class:`bytes`.

        If the message is fragmented, :meth:`get` waits until the last frame is
        received, then it reassembles the message and returns it. To receive
        messages frame by frame, use :meth:`get_iter` instead.

        Args:
            decode: :obj:`False` disables UTF-8 decoding of text frames and
                returns :class:`bytes`. :obj:`True` forces UTF-8 decoding of
                binary frames and returns :class:`str`.

        Raises:
            EOFError: If the stream of frames has ended.
            ConcurrencyError: If two coroutines run :meth:`get` or
                :meth:`get_iter` concurrently.

        """
        if self.closed:
            raise EOFError("stream of frames ended")

        if self.get_in_progress:
            raise ConcurrencyError("get() or get_iter() is already running")

        # Locking with get_in_progress ensures only one coroutine can get here.
        self.get_in_progress = True

        # First frame
        try:
            frame = await self.frames.get()
        except asyncio.CancelledError:
            self.get_in_progress = False
            raise
        self.maybe_resume()
        assert frame.opcode is OP_TEXT or frame.opcode is OP_BINARY
        if decode is None:
            decode = frame.opcode is OP_TEXT
        frames = [frame]

        # Following frames, for fragmented messages
        while not frame.fin:
            try:
                frame = await self.frames.get()
            except asyncio.CancelledError:
                # Put frames already received back into the queue
                # so that future calls to get() can return them.
                self.frames.reset(frames)
                self.get_in_progress = False
                raise
            self.maybe_resume()
            assert frame.opcode is OP_CONT
            frames.append(frame)

        self.get_in_progress = False

        data = b"".join(frame.data for frame in frames)
        if decode:
            return data.decode()
        else:
            return data

    async def get_iter(self, decode: bool | None = None) -> AsyncIterator[Data]:
        """
        Stream the next message.

        Iterating the return value of :meth:`get_iter` asynchronously yields a
        :class:`str` or :class:`bytes` for each frame in the message.

        The iterator must be fully consumed before calling :meth:`get_iter` or
        :meth:`get` again. Else, :exc:`ConcurrencyError` is raised.

        This method only makes sense for fragmented messages. If messages aren't
        fragmented, use :meth:`get` instead.

        Args:
            decode: :obj:`False` disables UTF-8 decoding of text frames and
                returns :class:`bytes`. :obj:`True` forces UTF-8 decoding of
                binary frames and returns :class:`str`.

        Raises:
            EOFError: If the stream of frames has ended.
            ConcurrencyError: If two coroutines run :meth:`get` or
                :meth:`get_iter` concurrently.

        """
        if self.closed:
            raise EOFError("stream of frames ended")

        if self.get_in_progress:
            raise ConcurrencyError("get() or get_iter() is already running")

        # Locking with get_in_progress ensures only one coroutine can get here.
        self.get_in_progress = True

        # First frame
        try:
            frame = await self.frames.get()
        except asyncio.CancelledError:
            self.get_in_progress = False
            raise
        self.maybe_resume()
        assert frame.opcode is OP_TEXT or frame.opcode is OP_BINARY
        if decode is None:
            decode = frame.opcode is OP_TEXT
        if decode:
            decoder = UTF8Decoder()
            yield decoder.decode(frame.data, frame.fin)
        else:
            yield frame.data

        # Following frames, for fragmented messages
        while not frame.fin:
            # We cannot handle asyncio.CancelledError because we don't buffer
            # previous fragments — we're streaming them. Canceling get_iter()
            # here will leave the assembler in a stuck state. Future calls to
            # get() or get_iter() will raise ConcurrencyError.
            frame = await self.frames.get()
            self.maybe_resume()
            assert frame.opcode is OP_CONT
            if decode:
                yield decoder.decode(frame.data, frame.fin)
            else:
                yield frame.data

        self.get_in_progress = False

    def put(self, frame: Frame) -> None:
        """
        Add ``frame`` to the next message.

        Raises:
            EOFError: If the stream of frames has ended.

        """
        if self.closed:
            raise EOFError("stream of frames ended")

        self.frames.put(frame)
        self.maybe_pause()

    def maybe_pause(self) -> None:
        """Pause the writer if queue is above the high water mark."""
        # Check for "> high" to support high = 0
        if len(self.frames) > self.high and not self.paused:
            self.paused = True
            self.pause()

    def maybe_resume(self) -> None:
        """Resume the writer if queue is below the low water mark."""
        # Check for "<= low" to support low = 0
        if len(self.frames) <= self.low and self.paused:
            self.paused = False
            self.resume()

    def close(self) -> None:
        """
        End the stream of frames.

        Callling :meth:`close` concurrently with :meth:`get`, :meth:`get_iter`,
        or :meth:`put` is safe. They will raise :exc:`EOFError`.

        """
        if self.closed:
            return

        self.closed = True

        # Unblock get() or get_iter().
        self.frames.abort()
