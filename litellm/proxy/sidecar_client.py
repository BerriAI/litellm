"""
LiteLLM Rust Sidecar Client

Forwards HTTP requests through the Rust sidecar for improved performance
under high concurrency. The sidecar provides:
- Pre-warmed connection pools (no per-request SSL/TCP overhead)
- Lock-free metrics aggregation  
- Zero GIL contention for I/O forwarding

The sidecar is optional — if unavailable, requests fall back to the
normal Python HTTP path.
"""

import asyncio
import json
import os
import subprocess
from typing import Optional, Union

import aiohttp

from litellm._logging import verbose_proxy_logger


class SidecarClient:
    """Client for communicating with the Rust sidecar process."""

    def __init__(
        self,
        sidecar_port: int = 8787,
        sidecar_binary: Optional[str] = None,
        auto_start: bool = False,
    ):
        self.sidecar_port = sidecar_port
        self.sidecar_url = f"http://127.0.0.1:{sidecar_port}"
        self.sidecar_binary = sidecar_binary
        self.auto_start = auto_start
        self._session: Optional[aiohttp.ClientSession] = None
        self._process: Optional[subprocess.Popen] = None
        self._healthy = False

    async def initialize(self):
        """Initialize the sidecar client and optionally start the sidecar process."""
        connector = aiohttp.TCPConnector(
            limit=0,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=90,
            enable_cleanup_closed=True,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None, connect=5),
        )

        if self.auto_start and self.sidecar_binary:
            await self._start_sidecar()

        self._healthy = await self._check_health()
        if self._healthy:
            verbose_proxy_logger.info(f"Sidecar client connected to {self.sidecar_url}")
        else:
            verbose_proxy_logger.warning(
                f"Sidecar not available at {self.sidecar_url}, will use fallback"
            )

    async def _start_sidecar(self):
        """Start the sidecar binary as a subprocess."""
        if self.sidecar_binary and os.path.isfile(self.sidecar_binary):
            env = os.environ.copy()
            env["SIDECAR_PORT"] = str(self.sidecar_port)
            self._process = subprocess.Popen(
                [self.sidecar_binary],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(50):
                await asyncio.sleep(0.1)
                if await self._check_health():
                    return
            verbose_proxy_logger.error("Sidecar failed to start within 5 seconds")

    async def _check_health(self) -> bool:
        """Check if the sidecar is healthy."""
        if self._session is None:
            return False
        try:
            async with self._session.get(
                f"{self.sidecar_url}/health", timeout=aiohttp.ClientTimeout(total=2)
            ) as resp:
                return resp.status == 200
        except Exception:
            return False

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    async def forward_request(
        self,
        provider_url: str,
        api_key: str,
        request_body: Union[dict, str, bytes],
        path: str = "/v1/chat/completions",
        timeout: int = 300,
        stream: bool = False,
    ) -> aiohttp.ClientResponse:
        """Forward a request through the sidecar to the provider."""
        if self._session is None:
            raise RuntimeError("SidecarClient not initialized")

        headers = {
            "X-LiteLLM-Provider-URL": provider_url,
            "X-LiteLLM-API-Key": api_key,
            "X-LiteLLM-Timeout": str(timeout),
            "X-LiteLLM-Stream": "true" if stream else "false",
            "X-LiteLLM-Path": path,
            "Content-Type": "application/json",
        }

        if isinstance(request_body, dict):
            body = json.dumps(request_body).encode()
        elif isinstance(request_body, str):
            body = request_body.encode()
        else:
            body = request_body

        resp = await self._session.post(
            f"{self.sidecar_url}/forward",
            data=body,
            headers=headers,
        )
        return resp

    async def close(self):
        """Shut down the sidecar client and optionally the sidecar process."""
        if self._session:
            await self._session.close()
            self._session = None
        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, self._process.wait),
                    timeout=5,
                )
            except asyncio.TimeoutError:
                self._process.kill()
            self._process = None
        self._healthy = False


# Global sidecar client instance
_sidecar_client: Optional[SidecarClient] = None


def get_sidecar_client() -> Optional[SidecarClient]:
    """Get the global sidecar client instance."""
    return _sidecar_client


async def init_sidecar_client(
    port: int = 8787,
    binary: Optional[str] = None,
    auto_start: bool = False,
) -> SidecarClient:
    """Initialize the global sidecar client."""
    global _sidecar_client
    _sidecar_client = SidecarClient(
        sidecar_port=port,
        sidecar_binary=binary,
        auto_start=auto_start,
    )
    await _sidecar_client.initialize()
    return _sidecar_client
