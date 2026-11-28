import asyncio
from types import MappingProxyType
from typing import Final

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.kafka import KafkaLogger


async def close_kafka_loggers(timeout: float) -> None:
    from litellm.litellm_core_utils.litellm_logging import (
        _in_memory_loggers,  # pyright: ignore[reportPrivateUsage]  # startup caches named callbacks here
    )

    callbacks: Final = (*litellm.logging_callback_manager.get_all_callbacks(), *_in_memory_loggers)
    loggers: Final = MappingProxyType(
        {id(callback): callback for callback in callbacks if isinstance(callback, KafkaLogger)}
    )
    results: Final = await asyncio.gather(
        *(logger.aclose(timeout) for logger in loggers.values()), return_exceptions=True
    )
    if any(isinstance(result, BaseException) for result in results):
        verbose_proxy_logger.warning("Kafka logging cleanup failed; continuing proxy shutdown")
