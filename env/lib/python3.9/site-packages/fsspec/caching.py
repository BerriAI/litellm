import collections
import functools
import logging
import math
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("fsspec")


class BaseCache:
    """Pass-though cache: doesn't keep anything, calls every time

    Acts as base class for other cachers

    Parameters
    ----------
    blocksize: int
        How far to read ahead in numbers of bytes
    fetcher: func
        Function of the form f(start, end) which gets bytes from remote as
        specified
    size: int
        How big this file is
    """

    name = "none"

    def __init__(self, blocksize, fetcher, size):
        self.blocksize = blocksize
        self.fetcher = fetcher
        self.size = size

    def _fetch(self, start, stop):
        if start is None:
            start = 0
        if stop is None:
            stop = self.size
        if start >= self.size or start >= stop:
            return b""
        return self.fetcher(start, stop)


class MMapCache(BaseCache):
    """memory-mapped sparse file cache

    Opens temporary file, which is filled blocks-wise when data is requested.
    Ensure there is enough disc space in the temporary location.

    This cache method might only work on posix
    """

    name = "mmap"

    def __init__(self, blocksize, fetcher, size, location=None, blocks=None):
        super().__init__(blocksize, fetcher, size)
        self.blocks = set() if blocks is None else blocks
        self.location = location
        self.cache = self._makefile()

    def _makefile(self):
        import mmap
        import tempfile

        if self.size == 0:
            return bytearray()

        # posix version
        if self.location is None or not os.path.exists(self.location):
            if self.location is None:
                fd = tempfile.TemporaryFile()
                self.blocks = set()
            else:
                fd = open(self.location, "wb+")
            fd.seek(self.size - 1)
            fd.write(b"1")
            fd.flush()
        else:
            fd = open(self.location, "rb+")

        return mmap.mmap(fd.fileno(), self.size)

    def _fetch(self, start, end):
        logger.debug(f"MMap cache fetching {start}-{end}")
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b""
        start_block = start // self.blocksize
        end_block = end // self.blocksize
        need = [i for i in range(start_block, end_block + 1) if i not in self.blocks]
        while need:
            # TODO: not a for loop so we can consolidate blocks later to
            # make fewer fetch calls; this could be parallel
            i = need.pop(0)
            sstart = i * self.blocksize
            send = min(sstart + self.blocksize, self.size)
            logger.debug(f"MMap get block #{i} ({sstart}-{send}")
            self.cache[sstart:send] = self.fetcher(sstart, send)
            self.blocks.add(i)

        return self.cache[start:end]

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state["cache"]
        return state

    def __setstate__(self, state):
        # Restore instance attributes
        self.__dict__.update(state)
        self.cache = self._makefile()


class ReadAheadCache(BaseCache):
    """Cache which reads only when we get beyond a block of data

    This is a much simpler version of BytesCache, and does not attempt to
    fill holes in the cache or keep fragments alive. It is best suited to
    many small reads in a sequential order (e.g., reading lines from a file).
    """

    name = "readahead"

    def __init__(self, blocksize, fetcher, size):
        super().__init__(blocksize, fetcher, size)
        self.cache = b""
        self.start = 0
        self.end = 0

    def _fetch(self, start, end):
        if start is None:
            start = 0
        if end is None or end > self.size:
            end = self.size
        if start >= self.size or start >= end:
            return b""
        l = end - start
        if start >= self.start and end <= self.end:
            # cache hit
            return self.cache[start - self.start : end - self.start]
        elif self.start <= start < self.end:
            # partial hit
            part = self.cache[start - self.start :]
            l -= len(part)
            start = self.end
        else:
            # miss
            part = b""
        end = min(self.size, end + self.blocksize)
        self.cache = self.fetcher(start, end)  # new block replaces old
        self.start = start
        self.end = self.start + len(self.cache)
        return part + self.cache[:l]


class FirstChunkCache(BaseCache):
    """Caches the first block of a file only

    This may be useful for file types where the metadata is stored in the header,
    but is randomly accessed.
    """

    name = "first"

    def __init__(self, blocksize, fetcher, size):
        super().__init__(blocksize, fetcher, size)
        self.cache = None

    def _fetch(self, start, end):
        start = start or 0
        end = end or self.size
        if start < self.blocksize:
            if self.cache is None:
                if end > self.blocksize:
                    data = self.fetcher(0, end)
                    self.cache = data[: self.blocksize]
                    return data[start:]
                self.cache = self.fetcher(0, self.blocksize)
            part = self.cache[start:end]
            if end > self.blocksize:
                part += self.fetcher(self.blocksize, end)
            return part
        else:
            return self.fetcher(start, end)


class BlockCache(BaseCache):
    """
    Cache holding memory as a set of blocks.

    Requests are only ever made ``blocksize`` at a time, and are
    stored in an LRU cache. The least recently accessed block is
    discarded when more than ``maxblocks`` are stored.

    Parameters
    ----------
    blocksize : int
        The number of bytes to store in each block.
        Requests are only ever made for ``blocksize``, so this
        should balance the overhead of making a request against
        the granularity of the blocks.
    fetcher : Callable
    size : int
        The total size of the file being cached.
    maxblocks : int
        The maximum number of blocks to cache for. The maximum memory
        use for this cache is then ``blocksize * maxblocks``.
    """

    name = "blockcache"

    def __init__(self, blocksize, fetcher, size, maxblocks=32):
        super().__init__(blocksize, fetcher, size)
        self.nblocks = math.ceil(size / blocksize)
        self.maxblocks = maxblocks
        self._fetch_block_cached = functools.lru_cache(maxblocks)(self._fetch_block)

    def __repr__(self):
        return "<BlockCache blocksize={}, size={}, nblocks={}>".format(
            self.blocksize, self.size, self.nblocks
        )

    def cache_info(self):
        """
        The statistics on the block cache.

        Returns
        -------
        NamedTuple
            Returned directly from the LRU Cache used internally.
        """
        return self._fetch_block_cached.cache_info()

    def __getstate__(self):
        state = self.__dict__
        del state["_fetch_block_cached"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fetch_block_cached = functools.lru_cache(state["maxblocks"])(
            self._fetch_block
        )

    def _fetch(self, start, end):
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b""

        # byte position -> block numbers
        start_block_number = start // self.blocksize
        end_block_number = end // self.blocksize

        # these are cached, so safe to do multiple calls for the same start and end.
        for block_number in range(start_block_number, end_block_number + 1):
            self._fetch_block_cached(block_number)

        return self._read_cache(
            start,
            end,
            start_block_number=start_block_number,
            end_block_number=end_block_number,
        )

    def _fetch_block(self, block_number):
        """
        Fetch the block of data for `block_number`.
        """
        if block_number > self.nblocks:
            raise ValueError(
                "'block_number={}' is greater than the number of blocks ({})".format(
                    block_number, self.nblocks
                )
            )

        start = block_number * self.blocksize
        end = start + self.blocksize
        logger.info("BlockCache fetching block %d", block_number)
        block_contents = super()._fetch(start, end)
        return block_contents

    def _read_cache(self, start, end, start_block_number, end_block_number):
        """
        Read from our block cache.

        Parameters
        ----------
        start, end : int
            The start and end byte positions.
        start_block_number, end_block_number : int
            The start and end block numbers.
        """
        start_pos = start % self.blocksize
        end_pos = end % self.blocksize

        if start_block_number == end_block_number:
            block = self._fetch_block_cached(start_block_number)
            return block[start_pos:end_pos]

        else:
            # read from the initial
            out = []
            out.append(self._fetch_block_cached(start_block_number)[start_pos:])

            # intermediate blocks
            # Note: it'd be nice to combine these into one big request. However
            # that doesn't play nicely with our LRU cache.
            for block_number in range(start_block_number + 1, end_block_number):
                out.append(self._fetch_block_cached(block_number))

            # final block
            out.append(self._fetch_block_cached(end_block_number)[:end_pos])

            return b"".join(out)


class BytesCache(BaseCache):
    """Cache which holds data in a in-memory bytes object

    Implements read-ahead by the block size, for semi-random reads progressing
    through the file.

    Parameters
    ----------
    trim: bool
        As we read more data, whether to discard the start of the buffer when
        we are more than a blocksize ahead of it.
    """

    name = "bytes"

    def __init__(self, blocksize, fetcher, size, trim=True):
        super().__init__(blocksize, fetcher, size)
        self.cache = b""
        self.start = None
        self.end = None
        self.trim = trim

    def _fetch(self, start, end):
        # TODO: only set start/end after fetch, in case it fails?
        # is this where retry logic might go?
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b""
        if (
            self.start is not None
            and start >= self.start
            and self.end is not None
            and end < self.end
        ):
            # cache hit: we have all the required data
            offset = start - self.start
            return self.cache[offset : offset + end - start]

        if self.blocksize:
            bend = min(self.size, end + self.blocksize)
        else:
            bend = end

        if bend == start or start > self.size:
            return b""

        if (self.start is None or start < self.start) and (
            self.end is None or end > self.end
        ):
            # First read, or extending both before and after
            self.cache = self.fetcher(start, bend)
            self.start = start
        elif start < self.start:
            if self.end - end > self.blocksize:
                self.cache = self.fetcher(start, bend)
                self.start = start
            else:
                new = self.fetcher(start, self.start)
                self.start = start
                self.cache = new + self.cache
        elif bend > self.end:
            if self.end > self.size:
                pass
            elif end - self.end > self.blocksize:
                self.cache = self.fetcher(start, bend)
                self.start = start
            else:
                new = self.fetcher(self.end, bend)
                self.cache = self.cache + new

        self.end = self.start + len(self.cache)
        offset = start - self.start
        out = self.cache[offset : offset + end - start]
        if self.trim:
            num = (self.end - self.start) // (self.blocksize + 1)
            if num > 1:
                self.start += self.blocksize * num
                self.cache = self.cache[self.blocksize * num :]
        return out

    def __len__(self):
        return len(self.cache)


class AllBytes(BaseCache):
    """Cache entire contents of the file"""

    name = "all"

    def __init__(self, blocksize=None, fetcher=None, size=None, data=None):
        super().__init__(blocksize, fetcher, size)
        if data is None:
            data = self.fetcher(0, self.size)
        self.data = data

    def _fetch(self, start, end):
        return self.data[start:end]


class KnownPartsOfAFile(BaseCache):
    """
    Cache holding known file parts.

    Parameters
    ----------
    blocksize: int
        How far to read ahead in numbers of bytes
    fetcher: func
        Function of the form f(start, end) which gets bytes from remote as
        specified
    size: int
        How big this file is
    data: dict
        A dictionary mapping explicit `(start, stop)` file-offset tuples
        with known bytes.
    strict: bool, default True
        Whether to fetch reads that go beyond a known byte-range boundary.
        If `False`, any read that ends outside a known part will be zero
        padded. Note that zero padding will not be used for reads that
        begin outside a known byte-range.
    """

    name = "parts"

    def __init__(self, blocksize, fetcher, size, data={}, strict=True, **_):
        super(KnownPartsOfAFile, self).__init__(blocksize, fetcher, size)
        self.strict = strict

        # simple consolidation of contiguous blocks
        if data:
            old_offsets = sorted(data.keys())
            offsets = [old_offsets[0]]
            blocks = [data.pop(old_offsets[0])]
            for start, stop in old_offsets[1:]:
                start0, stop0 = offsets[-1]
                if start == stop0:
                    offsets[-1] = (start0, stop)
                    blocks[-1] += data.pop((start, stop))
                else:
                    offsets.append((start, stop))
                    blocks.append(data.pop((start, stop)))

            self.data = dict(zip(offsets, blocks))
        else:
            self.data = data

    def _fetch(self, start, stop):
        out = b""
        for (loc0, loc1), data in self.data.items():
            # If self.strict=False, use zero-padded data
            # for reads beyond the end of a "known" buffer
            if loc0 <= start < loc1:
                off = start - loc0
                out = data[off : off + stop - start]
                if not self.strict or loc0 <= stop <= loc1:
                    # The request is within a known range, or
                    # it begins within a known range, and we
                    # are allowed to pad reads beyond the
                    # buffer with zero
                    out += b"\x00" * (stop - start - len(out))
                    return out
                else:
                    # The request ends outside a known range,
                    # and we are being "strict" about reads
                    # beyond the buffer
                    start = loc1
                    break

        # We only get here if there is a request outside the
        # known parts of the file. In an ideal world, this
        # should never happen
        if self.fetcher is None:
            # We cannot fetch the data, so raise an error
            raise ValueError(f"Read is outside the known file parts: {(start, stop)}. ")
        # We can fetch the data, but should warn the user
        # that this may be slow
        warnings.warn(
            f"Read is outside the known file parts: {(start, stop)}. "
            f"IO/caching performance may be poor!"
        )
        logger.debug(f"KnownPartsOfAFile cache fetching {start}-{stop}")
        return out + super()._fetch(start, stop)


class UpdatableLRU:
    """
    Custom implementation of LRU cache that allows updating keys

    Used by BackgroudBlockCache
    """

    CacheInfo = collections.namedtuple(
        "CacheInfo", ["hits", "misses", "maxsize", "currsize"]
    )

    def __init__(self, func, max_size=128):
        self._cache = collections.OrderedDict()
        self._func = func
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    def __call__(self, *args):
        with self._lock:
            if args in self._cache:
                self._cache.move_to_end(args)
                self._hits += 1
                return self._cache[args]

        result = self._func(*args)

        with self._lock:
            self._cache[args] = result
            self._misses += 1
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

        return result

    def is_key_cached(self, *args):
        with self._lock:
            return args in self._cache

    def add_key(self, result, *args):
        with self._lock:
            self._cache[args] = result
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def cache_info(self):
        with self._lock:
            return self.CacheInfo(
                maxsize=self._max_size,
                currsize=len(self._cache),
                hits=self._hits,
                misses=self._misses,
            )


class BackgroundBlockCache(BaseCache):
    """
    Cache holding memory as a set of blocks with pre-loading of
    the next block in the background.

    Requests are only ever made ``blocksize`` at a time, and are
    stored in an LRU cache. The least recently accessed block is
    discarded when more than ``maxblocks`` are stored. If the
    next block is not in cache, it is loaded in a separate thread
    in non-blocking way.

    Parameters
    ----------
    blocksize : int
        The number of bytes to store in each block.
        Requests are only ever made for ``blocksize``, so this
        should balance the overhead of making a request against
        the granularity of the blocks.
    fetcher : Callable
    size : int
        The total size of the file being cached.
    maxblocks : int
        The maximum number of blocks to cache for. The maximum memory
        use for this cache is then ``blocksize * maxblocks``.
    """

    name = "background"

    def __init__(self, blocksize, fetcher, size, maxblocks=32):
        super().__init__(blocksize, fetcher, size)
        self.nblocks = math.ceil(size / blocksize)
        self.maxblocks = maxblocks
        self._fetch_block_cached = UpdatableLRU(self._fetch_block, maxblocks)

        self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._fetch_future_block_number = None
        self._fetch_future = None
        self._fetch_future_lock = threading.Lock()

    def __repr__(self):
        return "<BackgroundBlockCache blocksize={}, size={}, nblocks={}>".format(
            self.blocksize, self.size, self.nblocks
        )

    def cache_info(self):
        """
        The statistics on the block cache.

        Returns
        -------
        NamedTuple
            Returned directly from the LRU Cache used internally.
        """
        return self._fetch_block_cached.cache_info()

    def __getstate__(self):
        state = self.__dict__
        del state["_fetch_block_cached"]
        del state["_thread_executor"]
        del state["_fetch_future_block_number"]
        del state["_fetch_future"]
        del state["_fetch_future_lock"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fetch_block_cached = UpdatableLRU(self._fetch_block, state["maxblocks"])
        self._thread_executor = ThreadPoolExecutor(max_workers=1)
        self._fetch_future_block_number = None
        self._fetch_future = None
        self._fetch_future_lock = threading.Lock()

    def _fetch(self, start, end):
        if start is None:
            start = 0
        if end is None:
            end = self.size
        if start >= self.size or start >= end:
            return b""

        # byte position -> block numbers
        start_block_number = start // self.blocksize
        end_block_number = end // self.blocksize

        fetch_future_block_number = None
        fetch_future = None
        with self._fetch_future_lock:
            # Background thread is running. Check we we can or must join it.
            if self._fetch_future is not None:
                if self._fetch_future.done():
                    logger.info("BlockCache joined background fetch without waiting.")
                    self._fetch_block_cached.add_key(
                        self._fetch_future.result(), self._fetch_future_block_number
                    )
                    # Cleanup the fetch variables. Done with fetching the block.
                    self._fetch_future_block_number = None
                    self._fetch_future = None
                else:
                    # Must join if we need the block for the current fetch
                    must_join = bool(
                        start_block_number
                        <= self._fetch_future_block_number
                        <= end_block_number
                    )
                    if must_join:
                        # Copy to the local variables to release lock
                        # before waiting for result
                        fetch_future_block_number = self._fetch_future_block_number
                        fetch_future = self._fetch_future

                        # Cleanup the fetch variables. Have a local copy.
                        self._fetch_future_block_number = None
                        self._fetch_future = None

        # Need to wait for the future for the current read
        if fetch_future is not None:
            logger.info("BlockCache waiting for background fetch.")
            # Wait until result and put it in cache
            self._fetch_block_cached.add_key(
                fetch_future.result(), fetch_future_block_number
            )

        # these are cached, so safe to do multiple calls for the same start and end.
        for block_number in range(start_block_number, end_block_number + 1):
            self._fetch_block_cached(block_number)

        # fetch next block in the background if nothing is running in the background,
        # the block is within file and it is not already cached
        end_block_plus_1 = end_block_number + 1
        with self._fetch_future_lock:
            if (
                self._fetch_future is None
                and end_block_plus_1 <= self.nblocks
                and not self._fetch_block_cached.is_key_cached(end_block_plus_1)
            ):
                self._fetch_future_block_number = end_block_plus_1
                self._fetch_future = self._thread_executor.submit(
                    self._fetch_block, end_block_plus_1, "async"
                )

        return self._read_cache(
            start,
            end,
            start_block_number=start_block_number,
            end_block_number=end_block_number,
        )

    def _fetch_block(self, block_number, log_info="sync"):
        """
        Fetch the block of data for `block_number`.
        """
        if block_number > self.nblocks:
            raise ValueError(
                "'block_number={}' is greater than the number of blocks ({})".format(
                    block_number, self.nblocks
                )
            )

        start = block_number * self.blocksize
        end = start + self.blocksize
        logger.info("BlockCache fetching block (%s) %d", log_info, block_number)
        block_contents = super()._fetch(start, end)
        return block_contents

    def _read_cache(self, start, end, start_block_number, end_block_number):
        """
        Read from our block cache.

        Parameters
        ----------
        start, end : int
            The start and end byte positions.
        start_block_number, end_block_number : int
            The start and end block numbers.
        """
        start_pos = start % self.blocksize
        end_pos = end % self.blocksize

        if start_block_number == end_block_number:
            block = self._fetch_block_cached(start_block_number)
            return block[start_pos:end_pos]

        else:
            # read from the initial
            out = []
            out.append(self._fetch_block_cached(start_block_number)[start_pos:])

            # intermediate blocks
            # Note: it'd be nice to combine these into one big request. However
            # that doesn't play nicely with our LRU cache.
            for block_number in range(start_block_number + 1, end_block_number):
                out.append(self._fetch_block_cached(block_number))

            # final block
            out.append(self._fetch_block_cached(end_block_number)[:end_pos])

            return b"".join(out)


caches = {
    # one custom case
    None: BaseCache,
}


def register_cache(cls, clobber=False):
    """'Register' cache implementation.

    Parameters
    ----------
    clobber: bool, optional
        If set to True (default is False) - allow to overwrite existing
        entry.

    Raises
    ------
    ValueError
    """
    name = cls.name
    if not clobber and name in caches:
        raise ValueError(f"Cache with name {name!r} is already known: {caches[name]}")
    caches[name] = cls


for c in (
    BaseCache,
    MMapCache,
    BytesCache,
    ReadAheadCache,
    BlockCache,
    FirstChunkCache,
    AllBytes,
    KnownPartsOfAFile,
    BackgroundBlockCache,
):
    register_cache(c)
