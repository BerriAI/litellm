from typing import Callable, List, Set, Type, Union

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.additional_logging_utils import AdditionalLoggingUtils
from litellm.integrations.custom_logger import CustomLogger


class LoggingCallbackManager:
    """
    A centralized class that allows easy add / remove callbacks for litellm.

    Goals of this class:
    - Prevent adding duplicate callbacks / success_callback / failure_callback
    - Keep a reasonable MAX_CALLBACKS limit (this ensures callbacks don't exponentially grow and consume CPU Resources)
    """

    # healthy maximum number of callbacks - unlikely someone needs more than 20
    MAX_CALLBACKS = 30

    def add_litellm_input_callback(self, callback: Union[CustomLogger, str]):
        """
        Add a input callback to litellm.input_callback
        """
        self._safe_add_callback_to_list(
            callback=callback, parent_list=litellm.input_callback
        )

    def add_litellm_service_callback(
        self, callback: Union[CustomLogger, str, Callable]
    ):
        """
        Add a service callback to litellm.service_callback
        """
        self._safe_add_callback_to_list(
            callback=callback, parent_list=litellm.service_callback
        )

    def add_litellm_callback(self, callback: Union[CustomLogger, str, Callable]):
        """
        Add a callback to litellm.callbacks

        Ensures no duplicates are added.
        """
        self._safe_add_callback_to_list(
            callback=callback, parent_list=litellm.callbacks  # type: ignore
        )

    def add_litellm_success_callback(
        self, callback: Union[CustomLogger, str, Callable]
    ):
        """
        Add a success callback to `litellm.success_callback`
        """
        self._safe_add_callback_to_list(
            callback=callback, parent_list=litellm.success_callback
        )

    def add_litellm_failure_callback(
        self, callback: Union[CustomLogger, str, Callable]
    ):
        """
        Add a failure callback to `litellm.failure_callback`
        """
        self._safe_add_callback_to_list(
            callback=callback, parent_list=litellm.failure_callback
        )

    def add_litellm_async_success_callback(
        self, callback: Union[CustomLogger, Callable, str]
    ):
        """
        Add a success callback to litellm._async_success_callback
        """
        self._safe_add_callback_to_list(
            callback=callback, parent_list=litellm._async_success_callback
        )

    def add_litellm_async_failure_callback(
        self, callback: Union[CustomLogger, Callable, str]
    ):
        """
        Add a failure callback to litellm._async_failure_callback
        """
        self._safe_add_callback_to_list(
            callback=callback, parent_list=litellm._async_failure_callback
        )

    def remove_callback_from_list_by_object(self, callback_list, obj):
        """
        Remove callbacks that are methods of a particular object (e.g., router cleanup)
        """
        if not isinstance(callback_list, list):  # Not list -> do nothing
            return

        remove_list = [
            c for c in callback_list if hasattr(c, "__self__") and c.__self__ == obj
        ]

        for c in remove_list:
            callback_list.remove(c)

    def _add_string_callback_to_list(
        self, callback: str, parent_list: List[Union[CustomLogger, Callable, str]]
    ):
        """
        Add a string callback to a list, if the callback is already in the list, do not add it again.
        """
        if callback not in parent_list:
            parent_list.append(callback)
        else:
            verbose_logger.debug(
                f"Callback {callback} already exists in {parent_list}, not adding again.."
            )

    def _check_callback_list_size(
        self, parent_list: List[Union[CustomLogger, Callable, str]]
    ) -> bool:
        """
        Check if adding another callback would exceed MAX_CALLBACKS
        Returns True if safe to add, False if would exceed limit
        """
        if len(parent_list) >= self.MAX_CALLBACKS:
            verbose_logger.warning(
                f"Cannot add callback - would exceed MAX_CALLBACKS limit of {self.MAX_CALLBACKS}. Current callbacks: {len(parent_list)}"
            )
            return False
        return True

    def _safe_add_callback_to_list(
        self,
        callback: Union[CustomLogger, Callable, str],
        parent_list: List[Union[CustomLogger, Callable, str]],
    ):
        """
        Safe add a callback to a list, if the callback is already in the list, do not add it again.

        Ensures no duplicates are added for `str`, `Callable`, and `CustomLogger` callbacks.
        """
        # Check max callbacks limit first
        if not self._check_callback_list_size(parent_list):
            return

        if isinstance(callback, str):
            self._add_string_callback_to_list(
                callback=callback, parent_list=parent_list
            )
        elif isinstance(callback, CustomLogger):
            self._add_custom_logger_to_list(
                custom_logger=callback,
                parent_list=parent_list,
            )
        elif callable(callback):
            self._add_callback_function_to_list(
                callback=callback, parent_list=parent_list
            )

    def _add_callback_function_to_list(
        self, callback: Callable, parent_list: List[Union[CustomLogger, Callable, str]]
    ):
        """
        Add a callback function to a list, if the callback is already in the list, do not add it again.
        """
        # Check if the function already exists in the list by comparing function objects
        if callback not in parent_list:
            parent_list.append(callback)
        else:
            verbose_logger.debug(
                f"Callback function {callback.__name__} already exists in {parent_list}, not adding again.."
            )

    def _add_custom_logger_to_list(
        self,
        custom_logger: CustomLogger,
        parent_list: List[Union[CustomLogger, Callable, str]],
    ):
        """
        Add a custom logger to a list, if another instance of the same custom logger exists in the list, do not add it again.
        """
        # Check if an instance of the same class already exists in the list
        custom_logger_key = self._get_custom_logger_key(custom_logger)
        custom_logger_type_name = type(custom_logger).__name__
        for existing_logger in parent_list:
            if (
                isinstance(existing_logger, CustomLogger)
                and self._get_custom_logger_key(existing_logger) == custom_logger_key
            ):
                verbose_logger.debug(
                    f"Custom logger of type {custom_logger_type_name}, key: {custom_logger_key} already exists in {parent_list}, not adding again.."
                )
                return
        parent_list.append(custom_logger)

    def _get_custom_logger_key(self, custom_logger: CustomLogger):
        """
        Get a unique key for a custom logger that considers only fundamental instance variables

        Returns:
            str: A unique key combining the class name and fundamental instance variables (str, bool, int)
        """
        key_parts = [type(custom_logger).__name__]

        # Add only fundamental type instance variables to the key
        for attr_name, attr_value in vars(custom_logger).items():
            if not attr_name.startswith("_"):  # Skip private attributes
                if isinstance(attr_value, (str, bool, int)):
                    key_parts.append(f"{attr_name}={attr_value}")

        return "-".join(key_parts)

    def _reset_all_callbacks(self):
        """
        Reset all callbacks to an empty list

        Note: this is an internal function and should be used sparingly.
        """
        litellm.input_callback = []
        litellm.success_callback = []
        litellm.failure_callback = []
        litellm._async_success_callback = []
        litellm._async_failure_callback = []
        litellm.callbacks = []

    def _get_all_callbacks(self) -> List[Union[CustomLogger, Callable, str]]:
        """
        Get all callbacks from litellm.callbacks, litellm.success_callback, litellm.failure_callback, litellm._async_success_callback, litellm._async_failure_callback
        """
        return (
            litellm.callbacks
            + litellm.success_callback
            + litellm.failure_callback
            + litellm._async_success_callback
            + litellm._async_failure_callback
        )

    def get_active_additional_logging_utils_from_custom_logger(
        self,
    ) -> Set[AdditionalLoggingUtils]:
        """
        Get all custom loggers that are instances of the given class type

        Args:
            class_type: The class type to match against (e.g., AdditionalLoggingUtils)

        Returns:
            Set[CustomLogger]: Set of custom loggers that are instances of the given class type
        """
        all_callbacks = self._get_all_callbacks()
        matched_callbacks: Set[AdditionalLoggingUtils] = set()
        for callback in all_callbacks:
            if isinstance(callback, CustomLogger) and isinstance(
                callback, AdditionalLoggingUtils
            ):
                matched_callbacks.add(callback)
        return matched_callbacks

    def get_custom_loggers_for_type(
        self, callback_type: Type[CustomLogger]
    ) -> List[CustomLogger]:
        """
        Get all custom loggers that are instances of the given class type
        """
        return [c for c in self._get_all_callbacks() if isinstance(c, callback_type)]
