from datetime import datetime

from azure.storage.blob import ContentSettings
from azure.storage.blob.aio import BlobServiceClient, ExponentialRetry

from .blob_client import BlobWriter
from .utils import get_env, make_key
from .logging import get_logger


logger = get_logger(__name__)


class AzureBlobClient(BlobWriter):
    """
    Wrapper around the Azure Blob Storage client.
    """

    def __init__(
        self,
        connection_string=None,
        account_name=None,
        account_key=None,
        container_name=None,
        path=None,
    ):
        self.container = str(
            get_env("AFLO_CONTAINER_NAME", container_name, required=True)
        )
        self.path = get_env("AFLO_PATH", path)

        # https://learn.microsoft.com/en-us/python/api/azure-storage-blob/azure.storage.blob.aio.exponentialretry
        retry = ExponentialRetry(initial_backoff=2, increment_base=3, retry_total=3)

        connection_string = get_env(
            "AZURE_STORAGE_CONNECTION_STRING", connection_string
        )

        if connection_string:
            logger.debug("Initializing Azure Blob service from connection string")
            self.blob_service = BlobServiceClient.from_connection_string(
                connection_string, retry_policy=retry
            )

        else:
            logger.debug(
                "Initializing Azure Blob service from account name and key pair"
            )

            account_name = get_env("AZURE_STORAGE_ACCOUNT_NAME", account_name)
            account_key = get_env("AZURE_STORAGE_ACCOUNT_KEY", account_key)

            account_url = f"https://{account_name}.blob.core.windows.net"

            self.blob_service = BlobServiceClient(
                account_url=account_url, credential=account_key, retry_policy=retry
            )

        logger.debug(
            "Initialized Azure Blob client: container: '%s', path: '%s'",
            self.container,
            self.path,
        )

    async def put_object(self, key: str, body: bytes) -> None:
        logger.debug(
            f"Attempting write to Azure Blob container {self.container}: {key}"
        )

        blob_client = self.blob_service.get_blob_client(
            container=self.container, blob=key
        )

        content_settings = ContentSettings(
            content_type="application/json",
            content_encoding="gzip",
        )

        r = await blob_client.upload_blob(
            body,
            overwrite=True,
            content_settings=content_settings,
        )
        logger.debug("r: %s", r)

        logger.debug(f"Wrote to Azure Blob container {self.container}: {key}")

    def make_key(self, timestamp: datetime) -> str:
        return make_key(timestamp, self.path)
