import os
from collections import deque

from sentry_sdk._compat import PY311
from sentry_sdk.utils import filename_for_module

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentry_sdk._lru_cache import LRUCache
    from types import FrameType
    from typing import Deque
    from typing import List
    from typing import Optional
    from typing import Sequence
    from typing import Tuple
    from typing_extensions import TypedDict

    ThreadId = str

    ProcessedStack = List[int]

    ProcessedFrame = TypedDict(
        "ProcessedFrame",
        {
            "abs_path": str,
            "filename": Optional[str],
            "function": str,
            "lineno": int,
            "module": Optional[str],
        },
    )

    ProcessedThreadMetadata = TypedDict(
        "ProcessedThreadMetadata",
        {"name": str},
    )

    FrameId = Tuple[
        str,  # abs_path
        int,  # lineno
        str,  # function
    ]
    FrameIds = Tuple[FrameId, ...]

    # The exact value of this id is not very meaningful. The purpose
    # of this id is to give us a compact and unique identifier for a
    # raw stack that can be used as a key to a dictionary so that it
    # can be used during the sampled format generation.
    StackId = Tuple[int, int]

    ExtractedStack = Tuple[StackId, FrameIds, List[ProcessedFrame]]
    ExtractedSample = Sequence[Tuple[ThreadId, ExtractedStack]]

# The default sampling frequency to use. This is set at 101 in order to
# mitigate the effects of lockstep sampling.
DEFAULT_SAMPLING_FREQUENCY = 101


# We want to impose a stack depth limit so that samples aren't too large.
MAX_STACK_DEPTH = 128


if PY311:

    def get_frame_name(frame):
        # type: (FrameType) -> str
        return frame.f_code.co_qualname

else:

    def get_frame_name(frame):
        # type: (FrameType) -> str

        f_code = frame.f_code
        co_varnames = f_code.co_varnames

        # co_name only contains the frame name.  If the frame was a method,
        # the class name will NOT be included.
        name = f_code.co_name

        # if it was a method, we can get the class name by inspecting
        # the f_locals for the `self` argument
        try:
            if (
                # the co_varnames start with the frame's positional arguments
                # and we expect the first to be `self` if its an instance method
                co_varnames
                and co_varnames[0] == "self"
                and "self" in frame.f_locals
            ):
                for cls in type(frame.f_locals["self"]).__mro__:
                    if name in cls.__dict__:
                        return "{}.{}".format(cls.__name__, name)
        except (AttributeError, ValueError):
            pass

        # if it was a class method, (decorated with `@classmethod`)
        # we can get the class name by inspecting the f_locals for the `cls` argument
        try:
            if (
                # the co_varnames start with the frame's positional arguments
                # and we expect the first to be `cls` if its a class method
                co_varnames
                and co_varnames[0] == "cls"
                and "cls" in frame.f_locals
            ):
                for cls in frame.f_locals["cls"].__mro__:
                    if name in cls.__dict__:
                        return "{}.{}".format(cls.__name__, name)
        except (AttributeError, ValueError):
            pass

        # nothing we can do if it is a staticmethod (decorated with @staticmethod)

        # we've done all we can, time to give up and return what we have
        return name


def frame_id(raw_frame):
    # type: (FrameType) -> FrameId
    return (raw_frame.f_code.co_filename, raw_frame.f_lineno, get_frame_name(raw_frame))


def extract_frame(fid, raw_frame, cwd):
    # type: (FrameId, FrameType, str) -> ProcessedFrame
    abs_path = raw_frame.f_code.co_filename

    try:
        module = raw_frame.f_globals["__name__"]
    except Exception:
        module = None

    # namedtuples can be many times slower when initialing
    # and accessing attribute so we opt to use a tuple here instead
    return {
        # This originally was `os.path.abspath(abs_path)` but that had
        # a large performance overhead.
        #
        # According to docs, this is equivalent to
        # `os.path.normpath(os.path.join(os.getcwd(), path))`.
        # The `os.getcwd()` call is slow here, so we precompute it.
        #
        # Additionally, since we are using normalized path already,
        # we skip calling `os.path.normpath` entirely.
        "abs_path": os.path.join(cwd, abs_path),
        "module": module,
        "filename": filename_for_module(module, abs_path) or None,
        "function": fid[2],
        "lineno": raw_frame.f_lineno,
    }


def extract_stack(
    raw_frame,  # type: Optional[FrameType]
    cache,  # type: LRUCache
    cwd,  # type: str
    max_stack_depth=MAX_STACK_DEPTH,  # type: int
):
    # type: (...) -> ExtractedStack
    """
    Extracts the stack starting the specified frame. The extracted stack
    assumes the specified frame is the top of the stack, and works back
    to the bottom of the stack.

    In the event that the stack is more than `MAX_STACK_DEPTH` frames deep,
    only the first `MAX_STACK_DEPTH` frames will be returned.
    """

    raw_frames = deque(maxlen=max_stack_depth)  # type: Deque[FrameType]

    while raw_frame is not None:
        f_back = raw_frame.f_back
        raw_frames.append(raw_frame)
        raw_frame = f_back

    frame_ids = tuple(frame_id(raw_frame) for raw_frame in raw_frames)
    frames = []
    for i, fid in enumerate(frame_ids):
        frame = cache.get(fid)
        if frame is None:
            frame = extract_frame(fid, raw_frames[i], cwd)
            cache.set(fid, frame)
        frames.append(frame)

    # Instead of mapping the stack into frame ids and hashing
    # that as a tuple, we can directly hash the stack.
    # This saves us from having to generate yet another list.
    # Additionally, using the stack as the key directly is
    # costly because the stack can be large, so we pre-hash
    # the stack, and use the hash as the key as this will be
    # needed a few times to improve performance.
    #
    # To Reduce the likelihood of hash collisions, we include
    # the stack depth. This means that only stacks of the same
    # depth can suffer from hash collisions.
    stack_id = len(raw_frames), hash(frame_ids)

    return stack_id, frame_ids, frames
