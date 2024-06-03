import litellm
import httpx, asyncio, traceback, os
from typing import Optional, Union, Mapping, Any
from litellm._logging import verbose_logger

# from litellm import print_verbose

# https://www.python-httpx.org/advanced/timeouts
_DEFAULT_TIMEOUT = httpx.Timeout(timeout=5.0, connect=5.0)


def str_to_bool(s: Union[str, bool]) -> bool:
    if isinstance(s, str):
        return s.lower() == "true"
    else:
        return s


class AsyncHTTPHandler:
    def __init__(
        self,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        concurrent_limit=1000,
    ):
        async_proxy_mounts = None
        # Check if the HTTP_PROXY and HTTPS_PROXY environment variables are set and use them accordingly.
        http_proxy = os.getenv("HTTP_PROXY", None)
        https_proxy = os.getenv("HTTPS_PROXY", None)
        no_proxy = os.getenv("NO_PROXY", None)
        ssl_verify = str_to_bool(os.getenv("SSL_VERIFY", True))
        cert = os.getenv(
            "SSL_CERTIFICATE", litellm.ssl_certificate
        )  # /path/to/client.pem

        if http_proxy is not None and https_proxy is not None:
            async_proxy_mounts = {
                "http://": httpx.AsyncHTTPTransport(proxy=httpx.Proxy(url=http_proxy)),
                "https://": httpx.AsyncHTTPTransport(
                    proxy=httpx.Proxy(url=https_proxy)
                ),
            }
            # assume no_proxy is a list of comma separated urls
            if no_proxy is not None and isinstance(no_proxy, str):
                no_proxy_urls = no_proxy.split(",")

                for url in no_proxy_urls:  # set no-proxy support for specific urls
                    async_proxy_mounts[url] = None  # type: ignore

        verbose_logger.info(
            f"\nSSL VERIFY VALUE!! ={ssl_verify}, os.getenv('SSL_VERIFY')={os.getenv('SSL_VERIFY')}\n"
        )
        if timeout is None:
            timeout = _DEFAULT_TIMEOUT
        # Create a client with a connection pool
        self.client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=concurrent_limit,
                max_keepalive_connections=concurrent_limit,
            ),
            verify=ssl_verify,
            mounts=async_proxy_mounts,
            cert=cert,
        )

    async def close(self):
        # Close the client when you're done with it
        await self.client.aclose()

    async def __aenter__(self):
        return self.client

    async def __aexit__(self):
        # close the client when exiting
        await self.client.aclose()

    async def get(
        self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None
    ):
        response = await self.client.get(url, params=params, headers=headers)
        return response

    async def post(
        self,
        url: str,
        data: Optional[Union[dict, str]] = None,  # type: ignore
        json: Optional[dict] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
    ):
        try:
            req = self.client.build_request(
                "POST", url, data=data, json=json, params=params, headers=headers  # type: ignore
            )
            response = await self.client.send(req, stream=stream)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise
        except Exception as e:
            raise

    def __del__(self) -> None:
        try:
            asyncio.get_running_loop().create_task(self.close())
        except Exception:
            pass


class HTTPHandler:
    def __init__(
        self,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        concurrent_limit=1000,
        client: Optional[httpx.Client] = None,
    ):
        if timeout is None:
            timeout = _DEFAULT_TIMEOUT

        # Check if the HTTP_PROXY and HTTPS_PROXY environment variables are set and use them accordingly.
        http_proxy = os.getenv("HTTP_PROXY", None)
        https_proxy = os.getenv("HTTPS_PROXY", None)
        no_proxy = os.getenv("NO_PROXY", None)
        ssl_verify = bool(os.getenv("SSL_VERIFY", litellm.ssl_verify))
        cert = os.getenv(
            "SSL_CERTIFICATE", litellm.ssl_certificate
        )  # /path/to/client.pem

        sync_proxy_mounts = None
        if http_proxy is not None and https_proxy is not None:
            sync_proxy_mounts = {
                "http://": httpx.HTTPTransport(proxy=httpx.Proxy(url=http_proxy)),
                "https://": httpx.HTTPTransport(proxy=httpx.Proxy(url=https_proxy)),
            }
            # assume no_proxy is a list of comma separated urls
            if no_proxy is not None and isinstance(no_proxy, str):
                no_proxy_urls = no_proxy.split(",")

                for url in no_proxy_urls:  # set no-proxy support for specific urls
                    sync_proxy_mounts[url] = None  # type: ignore

        if client is None:
            # Create a client with a connection pool
            self.client = httpx.Client(
                timeout=timeout,
                limits=httpx.Limits(
                    max_connections=concurrent_limit,
                    max_keepalive_connections=concurrent_limit,
                ),
                verify=ssl_verify,
                mounts=sync_proxy_mounts,
                cert=cert,
            )
        else:
            self.client = client

    def close(self):
        # Close the client when you're done with it
        self.client.close()

    def get(
        self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None
    ):
        response = self.client.get(url, params=params, headers=headers)
        return response

    def post(
        self,
        url: str,
        data: Optional[Union[dict, str]] = None,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        stream: bool = False,
    ):
        req = self.client.build_request(
            "POST", url, data=data, params=params, headers=headers  # type: ignore
        )
        response = self.client.send(req, stream=stream)
        return response

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
