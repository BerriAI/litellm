"""
DataDog Team Handler

Used to get the DataDogLogger for a given request.
Handles Key/Team Based Datadog Logging, following the same pattern as LangFuseHandler.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, TypedDict

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import StandardCallbackDynamicParams

from .datadog import DataDogLogger

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import DynamicLoggingCache
else:
    DynamicLoggingCache = Any


class DatadogLoggingConfig(TypedDict):
    dd_api_key: Optional[str]
    dd_site: Optional[str]
    dd_agent_host: Optional[str]
    dd_agent_port: Optional[str]


class DataDogHandler:
    @staticmethod
    def get_datadog_logger_for_request(
        standard_callback_dynamic_params: StandardCallbackDynamicParams,
        in_memory_dynamic_logger_cache: DynamicLoggingCache,
    ) -> DataDogLogger:
        """
        Get a team-scoped DataDogLogger for a given request.

        Resolves and caches per-team DataDogLogger instances using DynamicLoggingCache,
        keyed by the team's DD credentials. Each unique set of credentials gets its own
        logger instance with its own batch/flush loop.

        Note: This handler is only called when team-scoped DD credentials are present.
        The global (env-var based) DataDogLogger is managed separately by
        _init_custom_logger_compatible_class via _in_memory_loggers.
        """
        _credentials = DataDogHandler.get_dynamic_datadog_logging_config(
            standard_callback_dynamic_params=standard_callback_dynamic_params,
        )
        credentials_dict = dict(_credentials)

        # check if datadog logger is already cached
        temp_datadog_logger = in_memory_dynamic_logger_cache.get_cache(
            credentials=credentials_dict, service_name="datadog"
        )

        # if not cached, create a new datadog logger and cache it
        if temp_datadog_logger is None:
            temp_datadog_logger = DataDogHandler._create_datadog_logger_from_credentials(
                credentials=credentials_dict,
                in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
            )

        return temp_datadog_logger

    @staticmethod
    def _create_datadog_logger_from_credentials(
        credentials: Dict,
        in_memory_dynamic_logger_cache: DynamicLoggingCache,
    ) -> DataDogLogger:
        """
        Create a DataDogLogger from the credentials and cache it.
        """
        # When the destination is caller-supplied (dd_agent_host/dd_site), never fall back to the
        # proxy's DD_API_KEY env var, otherwise it would be sent to a team-controlled host.
        allow_env_credentials = credentials.get("dd_agent_host") is None and credentials.get("dd_site") is None
        datadog_logger = DataDogLogger(
            dd_api_key=credentials.get("dd_api_key"),
            dd_site=credentials.get("dd_site"),
            dd_agent_host=credentials.get("dd_agent_host"),
            dd_agent_port=credentials.get("dd_agent_port"),
            allow_env_credentials=allow_env_credentials,
        )
        in_memory_dynamic_logger_cache.set_cache(
            credentials=credentials,
            service_name="datadog",
            logging_obj=datadog_logger,
        )
        verbose_logger.debug("Datadog: Created and cached new DataDogLogger for team-scoped credentials")
        return datadog_logger

    @staticmethod
    def get_dynamic_datadog_logging_config(
        standard_callback_dynamic_params: StandardCallbackDynamicParams,
    ) -> DatadogLoggingConfig:
        """
        Get the Datadog logging config for a given request from dynamic params.
        """
        return DatadogLoggingConfig(
            dd_api_key=standard_callback_dynamic_params.get("dd_api_key"),
            dd_site=standard_callback_dynamic_params.get("dd_site"),
            dd_agent_host=standard_callback_dynamic_params.get("dd_agent_host"),
            dd_agent_port=standard_callback_dynamic_params.get("dd_agent_port"),
        )

    @staticmethod
    def _dynamic_datadog_credentials_are_passed(
        standard_callback_dynamic_params: StandardCallbackDynamicParams,
    ) -> bool:
        """
        Check if dynamic Datadog credentials are passed in standard_callback_dynamic_params.
        """
        if (
            standard_callback_dynamic_params.get("dd_api_key") is not None
            or standard_callback_dynamic_params.get("dd_site") is not None
            or standard_callback_dynamic_params.get("dd_agent_host") is not None
        ):
            return True
        return False
