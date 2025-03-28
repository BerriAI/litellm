"""
This file contains the LangFuseHandler class

Used to get the LangFuseLogger for a given request

Handles Key/Team Based Langfuse Logging
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from litellm.litellm_core_utils.litellm_logging import StandardCallbackDynamicParams

from .langfuse import LangFuseLogger, LangfuseLoggingConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import DynamicLoggingCache
else:
    DynamicLoggingCache = Any


class LangFuseHandler:

    @staticmethod
    def get_langfuse_logger_for_request(
        standard_callback_dynamic_params: StandardCallbackDynamicParams,
        in_memory_dynamic_logger_cache: DynamicLoggingCache,
        globalLangfuseLogger: Optional[LangFuseLogger] = None,
    ) -> LangFuseLogger:
        """
        This function is used to get the LangFuseLogger for a given request

        1. If dynamic credentials are passed
            - check if a LangFuseLogger is cached for the dynamic credentials
            - if cached LangFuseLogger is not found, create a new LangFuseLogger and cache it

        2. If dynamic credentials are not passed return the globalLangfuseLogger

        """
        temp_langfuse_logger: Optional[LangFuseLogger] = globalLangfuseLogger
        if (
            LangFuseHandler._dynamic_langfuse_credentials_are_passed(
                standard_callback_dynamic_params
            )
            is False
        ):
            return LangFuseHandler._return_global_langfuse_logger(
                globalLangfuseLogger=globalLangfuseLogger,
                in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
            )

        # get langfuse logging config to use for this request, based on standard_callback_dynamic_params
        _credentials = LangFuseHandler.get_dynamic_langfuse_logging_config(
            globalLangfuseLogger=globalLangfuseLogger,
            standard_callback_dynamic_params=standard_callback_dynamic_params,
        )
        credentials_dict = dict(_credentials)

        # check if langfuse logger is already cached
        temp_langfuse_logger = in_memory_dynamic_logger_cache.get_cache(
            credentials=credentials_dict, service_name="langfuse"
        )

        # if not cached, create a new langfuse logger and cache it
        if temp_langfuse_logger is None:
            temp_langfuse_logger = (
                LangFuseHandler._create_langfuse_logger_from_credentials(
                    credentials=credentials_dict,
                    in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
                )
            )

        return temp_langfuse_logger

    @staticmethod
    def _return_global_langfuse_logger(
        globalLangfuseLogger: Optional[LangFuseLogger],
        in_memory_dynamic_logger_cache: DynamicLoggingCache,
    ) -> LangFuseLogger:
        """
        Returns the Global LangfuseLogger set on litellm

        (this is the default langfuse logger - used when no dynamic credentials are passed)

        If no Global LangfuseLogger is set, it will check in_memory_dynamic_logger_cache for a cached LangFuseLogger
        This function is used to return the globalLangfuseLogger if it exists, otherwise it will check in_memory_dynamic_logger_cache for a cached LangFuseLogger
        """
        if globalLangfuseLogger is not None:
            return globalLangfuseLogger

        credentials_dict: Dict[str, Any] = (
            {}
        )  # the global langfuse logger uses Environment Variables, there are no dynamic credentials
        globalLangfuseLogger = in_memory_dynamic_logger_cache.get_cache(
            credentials=credentials_dict,
            service_name="langfuse",
        )
        if globalLangfuseLogger is None:
            globalLangfuseLogger = (
                LangFuseHandler._create_langfuse_logger_from_credentials(
                    credentials=credentials_dict,
                    in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
                )
            )
        return globalLangfuseLogger

    @staticmethod
    def _create_langfuse_logger_from_credentials(
        credentials: Dict,
        in_memory_dynamic_logger_cache: DynamicLoggingCache,
    ) -> LangFuseLogger:
        """
        This function is used to
        1. create a LangFuseLogger from the credentials
        2. cache the LangFuseLogger to prevent re-creating it for the same credentials
        """

        langfuse_logger = LangFuseLogger(
            langfuse_public_key=credentials.get("langfuse_public_key"),
            langfuse_secret=credentials.get("langfuse_secret"),
            langfuse_host=credentials.get("langfuse_host"),
        )
        in_memory_dynamic_logger_cache.set_cache(
            credentials=credentials,
            service_name="langfuse",
            logging_obj=langfuse_logger,
        )
        return langfuse_logger

    @staticmethod
    def get_dynamic_langfuse_logging_config(
        standard_callback_dynamic_params: StandardCallbackDynamicParams,
        globalLangfuseLogger: Optional[LangFuseLogger] = None,
    ) -> LangfuseLoggingConfig:
        """
        This function is used to get the Langfuse logging config to use for a given request.

        It checks if the dynamic parameters are provided in the standard_callback_dynamic_params and uses them to get the Langfuse logging config.

        If no dynamic parameters are provided, it uses the `globalLangfuseLogger` values
        """
        # only use dynamic params if langfuse credentials are passed dynamically
        return LangfuseLoggingConfig(
            langfuse_secret=standard_callback_dynamic_params.get("langfuse_secret")
            or standard_callback_dynamic_params.get("langfuse_secret_key"),
            langfuse_public_key=standard_callback_dynamic_params.get(
                "langfuse_public_key"
            ),
            langfuse_host=standard_callback_dynamic_params.get("langfuse_host"),
        )

    @staticmethod
    def _dynamic_langfuse_credentials_are_passed(
        standard_callback_dynamic_params: StandardCallbackDynamicParams,
    ) -> bool:
        """
        This function is used to check if the dynamic langfuse credentials are passed in standard_callback_dynamic_params

        Returns:
            bool: True if the dynamic langfuse credentials are passed, False otherwise
        """

        if (
            standard_callback_dynamic_params.get("langfuse_host") is not None
            or standard_callback_dynamic_params.get("langfuse_public_key") is not None
            or standard_callback_dynamic_params.get("langfuse_secret") is not None
            or standard_callback_dynamic_params.get("langfuse_secret_key") is not None
        ):
            return True
        return False
