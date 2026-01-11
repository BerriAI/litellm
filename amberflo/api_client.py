import aiohttp
from datetime import datetime
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from .blob_client import BlobWriter
from .utils import get_env, make_key
from .logging import get_logger


logger = get_logger(__name__)


class ApiClient(BlobWriter):
    """
    API client for sending data to Amberflo's ingestion endpoint.
    """

    def __init__(
        self,
        api_key=None,
        endpoint="https://ingest.amberflo.io",
    ):
        self.endpoint = get_env("AFLO_API_ENDPOINT", default=endpoint)

        api_key = get_env("AFLO_API_KEY", default=api_key, required=True)

        headers = {"x-api-key": api_key, "Content-Encoding": "gzip"}

        self.session = aiohttp.ClientSession(headers=headers)

        logger.debug("Initialized API client: endpoint: %s", self.endpoint)

    async def put_object(self, key: str, body: bytes) -> None:
        logger.debug("Attempting write to API: %s", key)

        await self._send_with_retry(key, body)

    def make_key(self, timestamp: datetime) -> str:
        return make_key(timestamp, None)

    @retry(
        reraise=True,
        wait=wait_random_exponential(multiplier=2, min=4, max=30),
        stop=stop_after_attempt(5),
    )
    async def _send_with_retry(self, key, data):
        async with self.session.post(self.endpoint, data=data) as r:
            status = r.status

            if r.ok:
                logger.debug("Wrote to API: %s: %s", key, status)
            else:
                response = await r.text()

                error = f"API call failed for key: {key}: {status}, {response}"

                # retry only on server errors
                if status < 500:
                    logger.error(error)
                else:
                    raise RuntimeError(error)
