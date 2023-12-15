from __future__ import annotations

import inspect
import logging
import os
import tempfile
import time
import weakref
from shutil import rmtree
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from fsspec import AbstractFileSystem, filesystem
from fsspec.callbacks import _DEFAULT_CALLBACK
from fsspec.compression import compr
from fsspec.core import BaseCache, MMapCache
from fsspec.exceptions import BlocksizeMismatchError
from fsspec.implementations.cache_mapper import create_cache_mapper
from fsspec.implementations.cache_metadata import CacheMetadata
from fsspec.spec import AbstractBufferedFile
from fsspec.utils import infer_compression

if TYPE_CHECKING:
    from fsspec.implementations.cache_mapper import AbstractCacheMapper

logger = logging.getLogger("fsspec.cached")


class CachingFileSystem(AbstractFileSystem):
    """Locally caching filesystem, layer over any other FS

    This class implements chunk-wise local storage of remote files, for quick
    access after the initial download. The files are stored in a given
    directory with hashes of URLs for the filenames. If no directory is given,
    a temporary one is used, which should be cleaned up by the OS after the
    process ends. The files themselves are sparse (as implemented in
    :class:`~fsspec.caching.MMapCache`), so only the data which is accessed
    takes up space.

    Restrictions:

    - the block-size must be the same for each access of a given file, unless
      all blocks of the file have already been read
    - caching can only be applied to file-systems which produce files
      derived from fsspec.spec.AbstractBufferedFile ; LocalFileSystem is also
      allowed, for testing
    """

    protocol: ClassVar[str | tuple[str, ...]] = ("blockcache", "cached")

    def __init__(
        self,
        target_protocol=None,
        cache_storage="TMP",
        cache_check=10,
        check_files=False,
        expiry_time=604800,
        target_options=None,
        fs=None,
        same_names: bool | None = None,
        compression=None,
        cache_mapper: AbstractCacheMapper | None = None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        target_protocol: str (optional)
            Target filesystem protocol. Provide either this or ``fs``.
        cache_storage: str or list(str)
            Location to store files. If "TMP", this is a temporary directory,
            and will be cleaned up by the OS when this process ends (or later).
            If a list, each location will be tried in the order given, but
            only the last will be considered writable.
        cache_check: int
            Number of seconds between reload of cache metadata
        check_files: bool
            Whether to explicitly see if the UID of the remote file matches
            the stored one before using. Warning: some file systems such as
            HTTP cannot reliably give a unique hash of the contents of some
            path, so be sure to set this option to False.
        expiry_time: int
            The time in seconds after which a local copy is considered useless.
            Set to falsy to prevent expiry. The default is equivalent to one
            week.
        target_options: dict or None
            Passed to the instantiation of the FS, if fs is None.
        fs: filesystem instance
            The target filesystem to run against. Provide this or ``protocol``.
        same_names: bool (optional)
            By default, target URLs are hashed using a ``HashCacheMapper`` so
            that files from different backends with the same basename do not
            conflict. If this argument is ``true``, a ``BasenameCacheMapper``
            is used instead. Other cache mapper options are available by using
            the ``cache_mapper`` keyword argument. Only one of this and
            ``cache_mapper`` should be specified.
        compression: str (optional)
            To decompress on download. Can be 'infer' (guess from the URL name),
            one of the entries in ``fsspec.compression.compr``, or None for no
            decompression.
        cache_mapper: AbstractCacheMapper (optional)
            The object use to map from original filenames to cached filenames.
            Only one of this and ``same_names`` should be specified.
        """
        super().__init__(**kwargs)
        if fs is None and target_protocol is None:
            raise ValueError(
                "Please provide filesystem instance(fs) or target_protocol"
            )
        if not (fs is None) ^ (target_protocol is None):
            raise ValueError(
                "Both filesystems (fs) and target_protocol may not be both given."
            )
        if cache_storage == "TMP":
            tempdir = tempfile.mkdtemp()
            storage = [tempdir]
            weakref.finalize(self, self._remove_tempdir, tempdir)
        else:
            if isinstance(cache_storage, str):
                storage = [cache_storage]
            else:
                storage = cache_storage
        os.makedirs(storage[-1], exist_ok=True)
        self.storage = storage
        self.kwargs = target_options or {}
        self.cache_check = cache_check
        self.check_files = check_files
        self.expiry = expiry_time
        self.compression = compression

        if same_names is not None and cache_mapper is not None:
            raise ValueError(
                "Cannot specify both same_names and cache_mapper in "
                "CachingFileSystem.__init__"
            )
        if cache_mapper is not None:
            self._mapper = cache_mapper
        else:
            self._mapper = create_cache_mapper(
                same_names if same_names is not None else False
            )

        self.target_protocol = (
            target_protocol
            if isinstance(target_protocol, str)
            else (fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0])
        )
        self._metadata = CacheMetadata(self.storage)
        self.load_cache()
        self.fs = fs if fs is not None else filesystem(target_protocol, **self.kwargs)

        def _strip_protocol(path):
            # acts as a method, since each instance has a difference target
            return self.fs._strip_protocol(type(self)._strip_protocol(path))

        self._strip_protocol: Callable = _strip_protocol

    @staticmethod
    def _remove_tempdir(tempdir):
        try:
            rmtree(tempdir)
        except Exception:
            pass

    def _mkcache(self):
        os.makedirs(self.storage[-1], exist_ok=True)

    def load_cache(self):
        """Read set of stored blocks from file"""
        self._metadata.load()
        self._mkcache()
        self.last_cache = time.time()

    def save_cache(self):
        """Save set of stored blocks from file"""
        self._mkcache()
        self._metadata.save()
        self.last_cache = time.time()

    def _check_cache(self):
        """Reload caches if time elapsed or any disappeared"""
        self._mkcache()
        if not self.cache_check:
            # explicitly told not to bother checking
            return
        timecond = time.time() - self.last_cache > self.cache_check
        existcond = all(os.path.exists(storage) for storage in self.storage)
        if timecond or not existcond:
            self.load_cache()

    def _check_file(self, path):
        """Is path in cache and still valid"""
        path = self._strip_protocol(path)
        self._check_cache()
        return self._metadata.check_file(path, self)

    def clear_cache(self):
        """Remove all files and metadata from the cache

        In the case of multiple cache locations, this clears only the last one,
        which is assumed to be the read/write one.
        """
        rmtree(self.storage[-1])
        self.load_cache()

    def clear_expired_cache(self, expiry_time=None):
        """Remove all expired files and metadata from the cache

        In the case of multiple cache locations, this clears only the last one,
        which is assumed to be the read/write one.

        Parameters
        ----------
        expiry_time: int
            The time in seconds after which a local copy is considered useless.
            If not defined the default is equivalent to the attribute from the
            file caching instantiation.
        """

        if not expiry_time:
            expiry_time = self.expiry

        self._check_cache()

        expired_files, writable_cache_empty = self._metadata.clear_expired(expiry_time)
        for fn in expired_files:
            if os.path.exists(fn):
                os.remove(fn)

        if writable_cache_empty:
            rmtree(self.storage[-1])
            self.load_cache()

    def pop_from_cache(self, path):
        """Remove cached version of given file

        Deletes local copy of the given (remote) path. If it is found in a cache
        location which is not the last, it is assumed to be read-only, and
        raises PermissionError
        """
        path = self._strip_protocol(path)
        fn = self._metadata.pop_file(path)
        if fn is not None:
            os.remove(fn)

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        """Wrap the target _open

        If the whole file exists in the cache, just open it locally and
        return that.

        Otherwise, open the file on the target FS, and make it have a mmap
        cache pointing to the location which we determine, in our cache.
        The ``blocks`` instance is shared, so as the mmap cache instance
        updates, so does the entry in our ``cached_files`` attribute.
        We monkey-patch this file, so that when it closes, we call
        ``close_and_update`` to save the state of the blocks.
        """
        path = self._strip_protocol(path)

        path = self.fs._strip_protocol(path)
        if "r" not in mode:
            return self.fs._open(
                path,
                mode=mode,
                block_size=block_size,
                autocommit=autocommit,
                cache_options=cache_options,
                **kwargs,
            )
        detail = self._check_file(path)
        if detail:
            # file is in cache
            detail, fn = detail
            hash, blocks = detail["fn"], detail["blocks"]
            if blocks is True:
                # stored file is complete
                logger.debug("Opening local copy of %s" % path)
                return open(fn, mode)
            # TODO: action where partial file exists in read-only cache
            logger.debug("Opening partially cached copy of %s" % path)
        else:
            hash = self._mapper(path)
            fn = os.path.join(self.storage[-1], hash)
            blocks = set()
            detail = {
                "original": path,
                "fn": hash,
                "blocks": blocks,
                "time": time.time(),
                "uid": self.fs.ukey(path),
            }
            self._metadata.update_file(path, detail)
            logger.debug("Creating local sparse file for %s" % path)

        # call target filesystems open
        self._mkcache()
        f = self.fs._open(
            path,
            mode=mode,
            block_size=block_size,
            autocommit=autocommit,
            cache_options=cache_options,
            cache_type="none",
            **kwargs,
        )
        if self.compression:
            comp = (
                infer_compression(path)
                if self.compression == "infer"
                else self.compression
            )
            f = compr[comp](f, mode="rb")
        if "blocksize" in detail:
            if detail["blocksize"] != f.blocksize:
                raise BlocksizeMismatchError(
                    "Cached file must be reopened with same block"
                    "size as original (old: %i, new %i)"
                    "" % (detail["blocksize"], f.blocksize)
                )
        else:
            detail["blocksize"] = f.blocksize
        f.cache = MMapCache(f.blocksize, f._fetch_range, f.size, fn, blocks)
        close = f.close
        f.close = lambda: self.close_and_update(f, close)
        self.save_cache()
        return f

    def hash_name(self, path: str, *args: Any) -> str:
        # Kept for backward compatibility with downstream libraries.
        # Ignores extra arguments, previously same_name boolean.
        return self._mapper(path)

    def close_and_update(self, f, close):
        """Called when a file is closing, so store the set of blocks"""
        if f.closed:
            return
        path = self._strip_protocol(f.path)
        self._metadata.on_close_cached_file(f, path)
        try:
            logger.debug("going to save")
            self.save_cache()
            logger.debug("saved")
        except OSError:
            logger.debug("Cache saving failed while closing file")
        except NameError:
            logger.debug("Cache save failed due to interpreter shutdown")
        close()
        f.closed = True

    def __getattribute__(self, item):
        if item in [
            "load_cache",
            "_open",
            "save_cache",
            "close_and_update",
            "__init__",
            "__getattribute__",
            "__reduce__",
            "_make_local_details",
            "open",
            "cat",
            "cat_file",
            "get",
            "read_block",
            "tail",
            "head",
            "_check_file",
            "_check_cache",
            "_mkcache",
            "clear_cache",
            "clear_expired_cache",
            "pop_from_cache",
            "_mkcache",
            "local_file",
            "_paths_from_path",
            "get_mapper",
            "open_many",
            "commit_many",
            "hash_name",
            "__hash__",
            "__eq__",
            "to_json",
        ]:
            # all the methods defined in this class. Note `open` here, since
            # it calls `_open`, but is actually in superclass
            return lambda *args, **kw: getattr(type(self), item).__get__(self)(
                *args, **kw
            )
        if item in ["__reduce_ex__"]:
            raise AttributeError
        if item in ["_cache"]:
            # class attributes
            return getattr(type(self), item)
        if item == "__class__":
            return type(self)
        d = object.__getattribute__(self, "__dict__")
        fs = d.get("fs", None)  # fs is not immediately defined
        if item in d:
            return d[item]
        elif fs is not None:
            if item in fs.__dict__:
                # attribute of instance
                return fs.__dict__[item]
            # attributed belonging to the target filesystem
            cls = type(fs)
            m = getattr(cls, item)
            if (inspect.isfunction(m) or inspect.isdatadescriptor(m)) and (
                not hasattr(m, "__self__") or m.__self__ is None
            ):
                # instance method
                return m.__get__(fs, cls)
            return m  # class method or attribute
        else:
            # attributes of the superclass, while target is being set up
            return super().__getattribute__(item)

    def __eq__(self, other):
        """Test for equality."""
        if self is other:
            return True
        if not isinstance(other, type(self)):
            return False
        return (
            self.storage == other.storage
            and self.kwargs == other.kwargs
            and self.cache_check == other.cache_check
            and self.check_files == other.check_files
            and self.expiry == other.expiry
            and self.compression == other.compression
            and self._mapper == other._mapper
            and self.target_protocol == other.target_protocol
        )

    def __hash__(self):
        """Calculate hash."""
        return (
            hash(tuple(self.storage))
            ^ hash(str(self.kwargs))
            ^ hash(self.cache_check)
            ^ hash(self.check_files)
            ^ hash(self.expiry)
            ^ hash(self.compression)
            ^ hash(self._mapper)
            ^ hash(self.target_protocol)
        )

    def to_json(self):
        """Calculate JSON representation.

        Not implemented yet for CachingFileSystem.
        """
        raise NotImplementedError(
            "CachingFileSystem JSON representation not implemented"
        )


class WholeFileCacheFileSystem(CachingFileSystem):
    """Caches whole remote files on first access

    This class is intended as a layer over any other file system, and
    will make a local copy of each file accessed, so that all subsequent
    reads are local. This is similar to ``CachingFileSystem``, but without
    the block-wise functionality and so can work even when sparse files
    are not allowed. See its docstring for definition of the init
    arguments.

    The class still needs access to the remote store for listing files,
    and may refresh cached files.
    """

    protocol = "filecache"
    local_file = True

    def open_many(self, open_files):
        paths = [of.path for of in open_files]
        if "r" in open_files.mode:
            self._mkcache()
        else:
            return [
                LocalTempFile(self.fs, path, mode=open_files.mode) for path in paths
            ]

        if self.compression:
            raise NotImplementedError
        details = [self._check_file(sp) for sp in paths]
        downpath = [p for p, d in zip(paths, details) if not d]
        downfn0 = [
            os.path.join(self.storage[-1], self._mapper(p))
            for p, d in zip(paths, details)
        ]  # keep these path names for opening later
        downfn = [fn for fn, d in zip(downfn0, details) if not d]
        if downpath:
            # skip if all files are already cached and up to date
            self.fs.get(downpath, downfn)

            # update metadata - only happens when downloads are successful
            newdetail = [
                {
                    "original": path,
                    "fn": self._mapper(path),
                    "blocks": True,
                    "time": time.time(),
                    "uid": self.fs.ukey(path),
                }
                for path in downpath
            ]
            for path, detail in zip(downpath, newdetail):
                self._metadata.update_file(path, detail)
            self.save_cache()

        def firstpart(fn):
            # helper to adapt both whole-file and simple-cache
            return fn[1] if isinstance(fn, tuple) else fn

        return [
            open(firstpart(fn0) if fn0 else fn1, mode=open_files.mode)
            for fn0, fn1 in zip(details, downfn0)
        ]

    def commit_many(self, open_files):
        self.fs.put([f.fn for f in open_files], [f.path for f in open_files])
        [f.close() for f in open_files]
        for f in open_files:
            # in case autocommit is off, and so close did not already delete
            try:
                os.remove(f.name)
            except FileNotFoundError:
                pass

    def _make_local_details(self, path):
        hash = self._mapper(path)
        fn = os.path.join(self.storage[-1], hash)
        detail = {
            "original": path,
            "fn": hash,
            "blocks": True,
            "time": time.time(),
            "uid": self.fs.ukey(path),
        }
        self._metadata.update_file(path, detail)
        logger.debug("Copying %s to local cache" % path)
        return fn

    def cat(
        self,
        path,
        recursive=False,
        on_error="raise",
        callback=_DEFAULT_CALLBACK,
        **kwargs,
    ):
        paths = self.expand_path(
            path, recursive=recursive, maxdepth=kwargs.get("maxdepth", None)
        )
        getpaths = []
        storepaths = []
        fns = []
        out = {}
        for p in paths.copy():
            try:
                detail = self._check_file(p)
                if not detail:
                    fn = self._make_local_details(p)
                    getpaths.append(p)
                    storepaths.append(fn)
                else:
                    detail, fn = detail if isinstance(detail, tuple) else (None, detail)
                fns.append(fn)
            except Exception as e:
                if on_error == "raise":
                    raise
                if on_error == "return":
                    out[p] = e
                paths.remove(p)

        if getpaths:
            self.fs.get(getpaths, storepaths)
            self.save_cache()

        callback.set_size(len(paths))
        for p, fn in zip(paths, fns):
            with open(fn, "rb") as f:
                out[p] = f.read()
            callback.relative_update(1)
        if isinstance(path, str) and len(paths) == 1 and recursive is False:
            out = out[paths[0]]
        return out

    def _open(self, path, mode="rb", **kwargs):
        path = self._strip_protocol(path)
        if "r" not in mode:
            return LocalTempFile(self, path, mode=mode)
        detail = self._check_file(path)
        if detail:
            detail, fn = detail
            _, blocks = detail["fn"], detail["blocks"]
            if blocks is True:
                logger.debug("Opening local copy of %s" % path)

                # In order to support downstream filesystems to be able to
                # infer the compression from the original filename, like
                # the `TarFileSystem`, let's extend the `io.BufferedReader`
                # fileobject protocol by adding a dedicated attribute
                # `original`.
                f = open(fn, mode)
                f.original = detail.get("original")
                return f
            else:
                raise ValueError(
                    "Attempt to open partially cached file %s"
                    "as a wholly cached file" % path
                )
        else:
            fn = self._make_local_details(path)
        kwargs["mode"] = mode

        # call target filesystems open
        self._mkcache()
        if self.compression:
            with self.fs._open(path, **kwargs) as f, open(fn, "wb") as f2:
                if isinstance(f, AbstractBufferedFile):
                    # want no type of caching if just downloading whole thing
                    f.cache = BaseCache(0, f.cache.fetcher, f.size)
                comp = (
                    infer_compression(path)
                    if self.compression == "infer"
                    else self.compression
                )
                f = compr[comp](f, mode="rb")
                data = True
                while data:
                    block = getattr(f, "blocksize", 5 * 2**20)
                    data = f.read(block)
                    f2.write(data)
        else:
            self.fs.get_file(path, fn)
        self.save_cache()
        return self._open(path, mode)


class SimpleCacheFileSystem(WholeFileCacheFileSystem):
    """Caches whole remote files on first access

    This class is intended as a layer over any other file system, and
    will make a local copy of each file accessed, so that all subsequent
    reads are local. This implementation only copies whole files, and
    does not keep any metadata about the download time or file details.
    It is therefore safer to use in multi-threaded/concurrent situations.

    This is the only of the caching filesystems that supports write: you will
    be given a real local open file, and upon close and commit, it will be
    uploaded to the target filesystem; the writability or the target URL is
    not checked until that time.

    """

    protocol = "simplecache"
    local_file = True

    def __init__(self, **kwargs):
        kw = kwargs.copy()
        for key in ["cache_check", "expiry_time", "check_files"]:
            kw[key] = False
        super().__init__(**kw)
        for storage in self.storage:
            if not os.path.exists(storage):
                os.makedirs(storage, exist_ok=True)

    def _check_file(self, path):
        self._check_cache()
        sha = self._mapper(path)
        for storage in self.storage:
            fn = os.path.join(storage, sha)
            if os.path.exists(fn):
                return fn

    def save_cache(self):
        pass

    def load_cache(self):
        pass

    def _open(self, path, mode="rb", **kwargs):
        path = self._strip_protocol(path)

        if "r" not in mode:
            return LocalTempFile(self, path, mode=mode)
        fn = self._check_file(path)
        if fn:
            return open(fn, mode)

        sha = self._mapper(path)
        fn = os.path.join(self.storage[-1], sha)
        logger.debug("Copying %s to local cache" % path)
        kwargs["mode"] = mode

        self._mkcache()
        if self.compression:
            with self.fs._open(path, **kwargs) as f, open(fn, "wb") as f2:
                if isinstance(f, AbstractBufferedFile):
                    # want no type of caching if just downloading whole thing
                    f.cache = BaseCache(0, f.cache.fetcher, f.size)
                comp = (
                    infer_compression(path)
                    if self.compression == "infer"
                    else self.compression
                )
                f = compr[comp](f, mode="rb")
                data = True
                while data:
                    block = getattr(f, "blocksize", 5 * 2**20)
                    data = f.read(block)
                    f2.write(data)
        else:
            self.fs.get_file(path, fn)
        return self._open(path, mode)


class LocalTempFile:
    """A temporary local file, which will be uploaded on commit"""

    def __init__(self, fs, path, fn=None, mode="wb", autocommit=True, seek=0):
        if fn:
            self.fn = fn
            self.fh = open(fn, mode)
        else:
            fd, self.fn = tempfile.mkstemp()
            self.fh = open(fd, mode)
        self.mode = mode
        if seek:
            self.fh.seek(seek)
        self.path = path
        self.fs = fs
        self.closed = False
        self.autocommit = autocommit

    def __reduce__(self):
        # always open in rb+ to allow continuing writing at a location
        return (
            LocalTempFile,
            (self.fs, self.path, self.fn, "rb+", self.autocommit, self.tell()),
        )

    def __enter__(self):
        return self.fh

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.closed:
            return
        self.fh.close()
        self.closed = True
        if self.autocommit:
            self.commit()

    def discard(self):
        self.fh.close()
        os.remove(self.fn)

    def commit(self):
        self.fs.put(self.fn, self.path)
        try:
            os.remove(self.fn)
        except (PermissionError, FileNotFoundError):
            # file path may be held by new version of the file on windows
            pass

    @property
    def name(self):
        return self.fn

    def __getattr__(self, item):
        return getattr(self.fh, item)
