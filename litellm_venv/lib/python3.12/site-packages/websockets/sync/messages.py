from __future__ import annotations

import codecs
import queue
import threading
from typing import Iterator, cast

from ..exceptions import ConcurrencyError
from ..frames import OP_BINARY, OP_CONT, OP_TEXT, Frame
from ..typing import Data


__all__ = ["Assembler"]

UTF8Decoder = codecs.getincrementaldecoder("utf-8")


class Assembler:
    """
    Assemble messages from frames.

    """

    def __init__(self) -> None:
        # Serialize reads and writes -- except for reads via synchronization
        # primitives provided by the threading and queue modules.
        self.mutex = threading.Lock()

        # We create a latch with two events to synchronize the production of
        # frames and the consumption of messages (or frames) without a buffer.
        # This design requires a switch between the library thread and the user
        # thread for each message; that shouldn't be a performance bottleneck.

        # put() sets this event to tell get() that a message can be fetched.
        self.message_complete = threading.Event()
        # get() sets this event to let put() that the message was fetched.
        self.message_fetched = threading.Event()

        # This flag prevents concurrent calls to get() by user code.
        self.get_in_progress = False
        # This flag prevents concurrent calls to put() by library code.
        self.put_in_progress = False

        # Decoder for text frames, None for binary frames.
        self.decoder: codecs.IncrementalDecoder | None = None

        # Buffer of frames belonging to the same message.
        self.chunks: list[Data] = []

        # When switching from "buffering" to "streaming", we use a thread-safe
        # queue for transferring frames from the writing thread (library code)
        # to the reading thread (user code). We're buffering when chunks_queue
        # is None and streaming when it's a SimpleQueue. None is a sentinel
        # value marking the end of the message, superseding message_complete.

        # Stream data from frames belonging to the same message.
        self.chunks_queue: queue.SimpleQueue[Data | None] | None = None

        # This flag marks the end of the connection.
        self.closed = False

    def get(self, timeout: float | None = None) -> Data:
        """
        Read the next message.

        :meth:`get` returns a single :class:`str` or :class:`bytes`.

        If the message is fragmented, :meth:`get` waits until the last frame is
        received, then it reassembles the message and returns it. To receive
        messages frame by frame, use :meth:`get_iter` instead.

        Args:
            timeout: If a timeout is provided and elapses before a complete
                message is received, :meth:`get` raises :exc:`TimeoutError`.

        Raises:
            EOFError: If the stream of frames has ended.
            ConcurrencyError: If two threads run :meth:`get` or :meth:`get_iter`
                concurrently.
            TimeoutError: If a timeout is provided and elapses before a
                complete message is received.

        """
        with self.mutex:
            if self.closed:
                raise EOFError("stream of frames ended")

            if self.get_in_progress:
                raise ConcurrencyError("get() or get_iter() is already running")

            self.get_in_progress = True

        # If the message_complete event isn't set yet, release the lock to
        # allow put() to run and eventually set it.
        # Locking with get_in_progress ensures only one thread can get here.
        completed = self.message_complete.wait(timeout)

        with self.mutex:
            self.get_in_progress = False

            # Waiting for a complete message timed out.
            if not completed:
                raise TimeoutError(f"timed out in {timeout:.1f}s")

            # get() was unblocked by close() rather than put().
            if self.closed:
                raise EOFError("stream of frames ended")

            assert self.message_complete.is_set()
            self.message_complete.clear()

            joiner: Data = b"" if self.decoder is None else ""
            # mypy cannot figure out that chunks have the proper type.
            message: Data = joiner.join(self.chunks)  # type: ignore

            self.chunks = []
            assert self.chunks_queue is None

            assert not self.message_fetched.is_set()
            self.message_fetched.set()

            return message

    def get_iter(self) -> Iterator[Data]:
        """
        Stream the next message.

        Iterating the return value of :meth:`get_iter` yields a :class:`str` or
        :class:`bytes` for each frame in the message.

        The iterator must be fully consumed before calling :meth:`get_iter` or
        :meth:`get` again. Else, :exc:`ConcurrencyError` is raised.

        This method only makes sense for fragmented messages. If messages aren't
        fragmented, use :meth:`get` instead.

        Raises:
            EOFError: If the stream of frames has ended.
            ConcurrencyError: If two threads run :meth:`get` or :meth:`get_iter`
                concurrently.

        """
        with self.mutex:
            if self.closed:
                raise EOFError("stream of frames ended")

            if self.get_in_progress:
                raise ConcurrencyError("get() or get_iter() is already running")

            chunks = self.chunks
            self.chunks = []
            self.chunks_queue = cast(
                # Remove quotes around type when dropping Python < 3.9.
                "queue.SimpleQueue[Data | None]",
                queue.SimpleQueue(),
            )

            # Sending None in chunk_queue supersedes setting message_complete
            # when switching to "streaming". If message is already complete
            # when the switch happens, put() didn't send None, so we have to.
            if self.message_complete.is_set():
                self.chunks_queue.put(None)

            self.get_in_progress = True

        # Locking with get_in_progress ensures only one thread can get here.
        chunk: Data | None
        for chunk in chunks:
            yield chunk
        while (chunk := self.chunks_queue.get()) is not None:
            yield chunk

        with self.mutex:
            self.get_in_progress = False

            # get_iter() was unblocked by close() rather than put().
            if self.closed:
                raise EOFError("stream of frames ended")

            assert self.message_complete.is_set()
            self.message_complete.clear()

            assert self.chunks == []
            self.chunks_queue = None

            assert not self.message_fetched.is_set()
            self.message_fetched.set()

    def put(self, frame: Frame) -> None:
        """
        Add ``frame`` to the next message.

        When ``frame`` is the final frame in a message, :meth:`put` waits until
        the message is fetched, which can be achieved by calling :meth:`get` or
        by fully consuming the return value of :meth:`get_iter`.

        :meth:`put` assumes that the stream of frames respects the protocol. If
        it doesn't, the behavior is undefined.

        Raises:
            EOFError: If the stream of frames has ended.
            ConcurrencyError: If two threads run :meth:`put` concurrently.

        """
        with self.mutex:
            if self.closed:
                raise EOFError("stream of frames ended")

            if self.put_in_progress:
                raise ConcurrencyError("put is already running")

            if frame.opcode is OP_TEXT:
                self.decoder = UTF8Decoder(errors="strict")
            elif frame.opcode is OP_BINARY:
                self.decoder = None
            else:
                assert frame.opcode is OP_CONT

            data: Data
            if self.decoder is not None:
                data = self.decoder.decode(frame.data, frame.fin)
            else:
                data = frame.data

            if self.chunks_queue is None:
                self.chunks.append(data)
            else:
                self.chunks_queue.put(data)

            if not frame.fin:
                return

            # Message is complete. Wait until it's fetched to return.

            assert not self.message_complete.is_set()
            self.message_complete.set()

            if self.chunks_queue is not None:
                self.chunks_queue.put(None)

            assert not self.message_fetched.is_set()

            self.put_in_progress = True

        # Release the lock to allow get() to run and eventually set the event.
        # Locking with put_in_progress ensures only one coroutine can get here.
        self.message_fetched.wait()

        with self.mutex:
            self.put_in_progress = False

            # put() was unblocked by close() rather than get() or get_iter().
            if self.closed:
                raise EOFError("stream of frames ended")

            assert self.message_fetched.is_set()
            self.message_fetched.clear()

            self.decoder = None

    def close(self) -> None:
        """
        End the stream of frames.

        Callling :meth:`close` concurrently with :meth:`get`, :meth:`get_iter`,
        or :meth:`put` is safe. They will raise :exc:`EOFError`.

        """
        with self.mutex:
            if self.closed:
                return

            self.closed = True

            # Unblock get or get_iter.
            if self.get_in_progress:
                self.message_complete.set()
                if self.chunks_queue is not None:
                    self.chunks_queue.put(None)

            # Unblock put().
            if self.put_in_progress:
                self.message_fetched.set()
