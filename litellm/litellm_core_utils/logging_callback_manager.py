from typing import Callable, List, Union

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger


class LoggingCallbackManager:
    """
    A centralized class that allows easy add / remove callbacks for litellm.

    Goals of this class:
    - Prevent adding duplicate callbacks / success_callback / failure_callback
    - Keep a reasonable MAX_CALLBACKS limit (this ensures callbacks don't exponentially grow and consume CPU Resources)
    """

    # healthy maximum number of callbacks - unlikely someone needs more than 20
    MAX_CALLBACKS = 20

    def add_input_callback(self, callback: Union[CustomLogger, str]):
        """
        Add a input callback to litellm.input_callback
        """
        if isinstance(callback, str):
            self._add_string_callback_to_list(
                callback=callback, parent_list=litellm.input_callback
            )
        pass

    def add_sync_success_callback(self, callback: Union[CustomLogger, str, Callable]):
        """
        Add a success callback to litellm.success_callback
        """
        pass

    def add_sync_failure_callback(self, callback: Union[CustomLogger, str, Callable]):
        """
        Add a failure callback to litellm.failure_callback
        """
        pass

    def add_async_success_callback(self, callback: Union[CustomLogger, Callable, str]):
        """
        Add a success callback to litellm._async_success_callback
        """
        if isinstance(callback, str):
            self._add_string_callback_to_list(
                callback=callback, parent_list=litellm._async_success_callback
            )
        elif isinstance(callback, Callable):
            self._add_callback_function_to_list(
                callback=callback, parent_list=litellm._async_success_callback
            )

    def add_async_failure_callback(self, callback: Union[CustomLogger, Callable, str]):
        """
        Add a failure callback to litellm._async_failure_callback
        """
        if isinstance(callback, str):
            self._add_string_callback_to_list(
                callback=callback, parent_list=litellm._async_failure_callback
            )
        elif isinstance(callback, Callable):
            self._add_callback_function_to_list(
                callback=callback, parent_list=litellm._async_failure_callback
            )

    def add_success_callback_sync_and_async(self, callback: Union[CustomLogger, str]):
        """
        Add a success callback to litellm.success_callback and litellm._async_success_callback
        """
        self.add_sync_success_callback(callback)
        self.add_async_success_callback(callback)
        pass

    def add_failure_callback_sync_and_async(self, callback: Union[CustomLogger, str]):
        """
        Add a failure callback to litellm.failure_callback and litellm._async_failure_callback
        """
        self.add_sync_failure_callback(callback)
        self.add_async_failure_callback(callback)

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
