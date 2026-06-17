import json
import os
import sqlite3
import time
from contextlib import closing
from typing import Callable, Optional, Tuple, Union

from .base_cache import BaseCache


def _encode(value: object) -> Tuple[Union[str, bytes], str]:
    if isinstance(value, bytes):
        return value, "b"
    if isinstance(value, str):
        return value, "s"
    return json.dumps(value), "j"


def _decode(stored: Union[str, bytes], mode: str) -> object:
    if mode == "j":
        decoded: object = json.loads(stored)
        return decoded
    return stored


class _SqliteCache:
    """Persistent key-value store backing DiskCache, on stdlib sqlite3.

    Values are stored as JSON text (or raw text/bytes), never pickled, so the
    read path cannot deserialize executable payloads. Exposes the small surface
    DiskCache relies on: set/get/pop/clear with relative-seconds expiry.
    """

    def __init__(
        self, directory: str, time_fn: Callable[[], float] = time.time
    ) -> None:
        self._time = time_fn
        os.makedirs(directory, exist_ok=True)
        self._path = os.path.join(directory, "cache.db")
        with closing(self._connect()) as con, con:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, value, mode TEXT NOT NULL, expire_time REAL)"
            )

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._path, timeout=60.0)
        con.execute("PRAGMA busy_timeout=60000")
        return con

    def _live_value(
        self, row: Optional[Tuple[Union[str, bytes], str, Optional[float]]]
    ) -> object:
        if row is None:
            return None
        stored, mode, expire_time = row
        if expire_time is not None and expire_time <= self._time():
            return None
        return _decode(stored, mode)

    def set(self, key: str, value: object, expire: Optional[float] = None) -> None:
        encoded, mode = _encode(value)
        expire_time = None if expire is None else self._time() + expire
        with closing(self._connect()) as con, con:
            con.execute(
                "INSERT OR REPLACE INTO cache(key, value, mode, expire_time) "
                "VALUES (?, ?, ?, ?)",
                (key, encoded, mode, expire_time),
            )

    def get(self, key: str) -> object:
        with closing(self._connect()) as con:
            row = con.execute(
                "SELECT value, mode, expire_time FROM cache WHERE key = ?", (key,)
            ).fetchone()
        return self._live_value(row)

    def pop(self, key: str) -> object:
        with closing(self._connect()) as con, con:
            row = con.execute(
                "SELECT value, mode, expire_time FROM cache WHERE key = ?", (key,)
            ).fetchone()
            con.execute("DELETE FROM cache WHERE key = ?", (key,))
        return self._live_value(row)

    def clear(self) -> None:
        with closing(self._connect()) as con, con:
            con.execute("DELETE FROM cache")


class DiskCache(BaseCache):
    def __init__(
        self,
        disk_cache_dir: Optional[str] = None,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.disk_cache = _SqliteCache(
            disk_cache_dir or ".litellm_cache", time_fn=time_fn
        )

    def set_cache(self, key, value, **kwargs):
        if "ttl" in kwargs:
            self.disk_cache.set(key, value, expire=kwargs["ttl"])
        else:
            self.disk_cache.set(key, value)

    async def async_set_cache(self, key, value, **kwargs):
        self.set_cache(key=key, value=value, **kwargs)

    async def async_set_cache_pipeline(self, cache_list, **kwargs):
        for cache_key, cache_value in cache_list:
            if "ttl" in kwargs:
                self.set_cache(key=cache_key, value=cache_value, ttl=kwargs["ttl"])
            else:
                self.set_cache(key=cache_key, value=cache_value)

    def get_cache(self, key, **kwargs):
        original_cached_response = self.disk_cache.get(key)
        if original_cached_response:
            try:
                cached_response = json.loads(original_cached_response)  # type: ignore
            except Exception:
                cached_response = original_cached_response
            return cached_response
        return None

    def batch_get_cache(self, keys: list, **kwargs):
        return_val = []
        for k in keys:
            val = self.get_cache(key=k, **kwargs)
            return_val.append(val)
        return return_val

    def increment_cache(self, key, value: int, **kwargs) -> int:
        # get the value
        init_value = self.get_cache(key=key) or 0
        value = init_value + value  # type: ignore
        self.set_cache(key, value, **kwargs)
        return value

    async def async_get_cache(self, key, **kwargs):
        return self.get_cache(key=key, **kwargs)

    async def async_batch_get_cache(self, keys: list, **kwargs):
        return_val = []
        for k in keys:
            val = self.get_cache(key=k, **kwargs)
            return_val.append(val)
        return return_val

    async def async_increment(self, key, value: int, **kwargs) -> int:
        # get the value
        init_value = await self.async_get_cache(key=key) or 0
        value = init_value + value  # type: ignore
        await self.async_set_cache(key, value, **kwargs)
        return value

    def flush_cache(self):
        self.disk_cache.clear()

    async def disconnect(self):
        pass

    def delete_cache(self, key):
        self.disk_cache.pop(key)
