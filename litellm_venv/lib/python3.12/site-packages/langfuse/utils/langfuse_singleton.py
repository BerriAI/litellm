import threading
from typing import Optional

import httpx

from langfuse import Langfuse
from langfuse.types import MaskFunction


class LangfuseSingleton:
    _instance = None
    _lock = threading.Lock()
    _langfuse: Optional[Langfuse] = None

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(LangfuseSingleton, cls).__new__(cls)
        return cls._instance

    def get(
        self,
        *,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
        release: Optional[str] = None,
        debug: Optional[bool] = None,
        threads: Optional[int] = None,
        flush_at: Optional[int] = None,
        flush_interval: Optional[int] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        httpx_client: Optional[httpx.Client] = None,
        sdk_integration: Optional[str] = None,
        enabled: Optional[bool] = None,
        sample_rate: Optional[float] = None,
        mask: Optional[MaskFunction] = None,
        environment: Optional[str] = None,
    ) -> Langfuse:
        if self._langfuse:
            return self._langfuse

        with self._lock:
            if self._langfuse:
                return self._langfuse

            langfuse_init_args = {
                "public_key": public_key,
                "secret_key": secret_key,
                "host": host,
                "release": release,
                "debug": debug,
                "threads": threads,
                "flush_at": flush_at,
                "flush_interval": flush_interval,
                "max_retries": max_retries,
                "timeout": timeout,
                "httpx_client": httpx_client,
                "sdk_integration": sdk_integration,
                "enabled": enabled,
                "sample_rate": sample_rate,
                "mask": mask,
                "environment": environment,
            }

            self._langfuse = Langfuse(
                **{k: v for k, v in langfuse_init_args.items() if v is not None}
            )

            return self._langfuse

    def reset(self) -> None:
        with self._lock:
            if self._langfuse:
                self._langfuse.shutdown()

            self._langfuse = None
