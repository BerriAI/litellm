from typing import Final

from litellm.types.integrations.custom_logger import StandardCustomLoggerInitParams

AZURE_SENTINEL_MAX_PAYLOAD_SIZE_BYTES: Final = 1_000_000


class AzureSentinelInitParams(StandardCustomLoggerInitParams):
    """
    Params for initializing an Azure Sentinel logger on litellm
    """
