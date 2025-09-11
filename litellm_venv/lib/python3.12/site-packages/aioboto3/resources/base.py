import logging
import warnings

from boto3.resources.base import ServiceResource

logger = logging.getLogger(__name__)


class AIOBoto3ServiceResource(ServiceResource):
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.meta.client.__aexit__(exc_type, exc_val, exc_tb)

    def close(self):
        warnings.warn("This should not be called anymore", DeprecationWarning)
        return self.meta.client.close()
