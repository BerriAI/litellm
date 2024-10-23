"""
This file contains the LangFuseHandler class

Used to get the LangFuseLogger for a given request

Handles Key/Team Based Langfuse Logging
"""

from typing import Dict, Optional

from litellm.litellm_core_utils.litellm_logging import (
    DynamicLoggingCache,
    StandardCallbackDynamicParams,
)

from .langfuse import LangFuseLogger, LangfuseLoggingConfig


class LangFuseHandler:

    @staticmethod
    def get_langfuse_logger_for_request(
        standard_callback_dynamic_params: StandardCallbackDynamicParams,
        in_memory_dynamic_logger_cache: DynamicLoggingCache,
        globalLangfuseLogger: Optional[LangFuseLogger] = None,
    ) -> LangFuseLogger:
        temp_langfuse_logger: Optional[LangFuseLogger] = globalLangfuseLogger
        if (
            LangFuseHandler._dynamic_langfuse_credentials_are_passed(
                standard_callback_dynamic_params
            )
            is False
        ):
            if temp_langfuse_logger is None:
                credentials_dict = {}
                temp_langfuse_logger = in_memory_dynamic_logger_cache.get_cache(
                    credentials=credentials_dict,
                    service_name="langfuse",
                )
                if temp_langfuse_logger is None:
                    temp_langfuse_logger = LangFuseHandler.create_langfuse_logger_from_credentials(
                        credentials=credentials_dict,
                        in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
                    )

            return temp_langfuse_logger

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
                LangFuseHandler.create_langfuse_logger_from_credentials(
                    credentials=credentials_dict,
                    in_memory_dynamic_logger_cache=in_memory_dynamic_logger_cache,
                )
            )

        return temp_langfuse_logger

    @staticmethod
    def create_langfuse_logger_from_credentials(
        credentials: Dict,
        in_memory_dynamic_logger_cache: DynamicLoggingCache,
    ) -> LangFuseLogger:
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
