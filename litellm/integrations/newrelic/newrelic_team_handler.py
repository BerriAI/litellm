"""
New Relic Team Handler

Used to get the NewRelicMetricsLogger for a given request.
Handles Key/Team Based New Relic metrics, following the same pattern as DataDogHandler.
"""

from typing import TYPE_CHECKING, Final

from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import StandardCallbackDynamicParams

from .newrelic_metrics import NewRelicMetricsLogger

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import DynamicLoggingCache


class NewRelicLoggingConfig(TypedDict):
    newrelic_api_key: ReadOnly[str | None]
    newrelic_region: ReadOnly[str | None]


class NewRelicHandler:
    @staticmethod
    def get_newrelic_logger_for_request(
        standard_callback_dynamic_params: StandardCallbackDynamicParams,
        in_memory_dynamic_logger_cache: "DynamicLoggingCache",
    ) -> NewRelicMetricsLogger:
        """
        Get a team-scoped NewRelicMetricsLogger for a given request.

        Resolves and caches per-team NewRelicMetricsLogger instances using
        DynamicLoggingCache, keyed by the team's New Relic credentials. Each unique
        set of credentials gets its own logger instance with its own batch/flush loop.

        Note: This handler is only called when a team-scoped newrelic_api_key is
        present. The trace logger for the ``newrelic`` callback (OTel v2 / legacy
        agent) is managed separately by _init_custom_logger_compatible_class via
        _in_memory_loggers.
        """
        _credentials: Final = NewRelicHandler.get_dynamic_newrelic_logging_config(
            standard_callback_dynamic_params=standard_callback_dynamic_params,
        )

        temp_newrelic_logger = in_memory_dynamic_logger_cache.get_cache(
            credentials=_credentials, service_name="newrelic"
        )

        if temp_newrelic_logger is None:
            temp_newrelic_logger = NewRelicHandler._create_newrelic_logger_from_credentials(
                credentials=_credentials,
                in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
            )

        return temp_newrelic_logger

    @staticmethod
    def _create_newrelic_logger_from_credentials(
        credentials: NewRelicLoggingConfig,
        in_memory_dynamic_logger_cache: "DynamicLoggingCache",
    ) -> NewRelicMetricsLogger:
        newrelic_logger: Final = NewRelicMetricsLogger(
            newrelic_api_key=credentials.get("newrelic_api_key") or "",
            newrelic_region=credentials.get("newrelic_region"),
        )
        in_memory_dynamic_logger_cache.set_cache(
            credentials=credentials,
            service_name="newrelic",
            logging_obj=newrelic_logger,
        )
        verbose_logger.debug("New Relic: Created and cached new NewRelicMetricsLogger for team-scoped credentials")
        return newrelic_logger

    @staticmethod
    def get_dynamic_newrelic_logging_config(
        standard_callback_dynamic_params: StandardCallbackDynamicParams,
    ) -> NewRelicLoggingConfig:
        return NewRelicLoggingConfig(
            newrelic_api_key=standard_callback_dynamic_params.get("newrelic_api_key"),
            newrelic_region=standard_callback_dynamic_params.get("newrelic_region"),
        )

    @staticmethod
    def _dynamic_newrelic_credentials_are_passed(
        standard_callback_dynamic_params: StandardCallbackDynamicParams,
    ) -> bool:
        return standard_callback_dynamic_params.get("newrelic_api_key") is not None
