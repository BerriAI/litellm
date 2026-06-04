from typing import List, Optional

import litellm
from litellm._logging import verbose_logger
from litellm.constants import X_LITELLM_DISABLE_CALLBACKS
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.llm_request_utils import (
    get_proxy_server_request_headers,
)
from litellm.proxy._types import CommonProxyErrors
from litellm.types.utils import StandardCallbackDynamicParams


class EnterpriseCallbackControls:
    @staticmethod
    def is_callback_disabled_dynamically(
            callback: litellm.CALLBACK_TYPES, 
            litellm_params: dict,
            standard_callback_dynamic_params: StandardCallbackDynamicParams
        ) -> bool:
            """
            Check if a callback is disabled via the x-litellm-disable-callbacks header or via `litellm_disabled_callbacks` in standard_callback_dynamic_params.
            
            Args:
                callback: The callback to check (can be string, CustomLogger instance, or callable)
                litellm_params: Parameters containing proxy server request info
                
            Returns:
                bool: True if the callback should be disabled, False otherwise
            """
            from litellm.litellm_core_utils.custom_logger_registry import (
                CustomLoggerRegistry,
            )

            try:
                disabled_callbacks = EnterpriseCallbackControls.get_disabled_callbacks(litellm_params, standard_callback_dynamic_params)
                verbose_logger.debug(f"Dynamically disabled callbacks from {X_LITELLM_DISABLE_CALLBACKS}: {disabled_callbacks}")
                verbose_logger.debug(f"Checking if {callback} is disabled via headers. Disable callbacks from headers: {disabled_callbacks}")
                if disabled_callbacks is not None:
                    #########################################################
                    # premium user check
                    #########################################################
                    if not EnterpriseCallbackControls._should_allow_dynamic_callback_disabling():
                        return False
                    #########################################################
                    if isinstance(callback, str):
                        if callback.lower() in disabled_callbacks:
                            verbose_logger.debug(f"Not logging to {callback} because it is disabled via {X_LITELLM_DISABLE_CALLBACKS}")
                            return True
                    elif isinstance(callback, CustomLogger):
                        # get the string name of the callback
                        callback_str = CustomLoggerRegistry.get_callback_str_from_class_type(callback.__class__)
                        if callback_str is not None and callback_str.lower() in disabled_callbacks:
                            verbose_logger.debug(f"Not logging to {callback_str} because it is disabled via {X_LITELLM_DISABLE_CALLBACKS}")
                            return True
                return False
            except Exception as e:
                verbose_logger.debug(
                    f"Error checking disabled callbacks header: {str(e)}"
                )
                return False
    @staticmethod
    def get_disabled_callbacks(litellm_params: dict, standard_callback_dynamic_params: StandardCallbackDynamicParams) -> Optional[List[str]]:
        """
        Get the disabled callbacks from the standard callback dynamic params.
        """

        #########################################################
        # check if disabled via headers
        #########################################################
        request_headers = get_proxy_server_request_headers(litellm_params)
        disabled_callbacks = request_headers.get(X_LITELLM_DISABLE_CALLBACKS, None)
        if disabled_callbacks is not None:
            disabled_callbacks = set([cb.strip().lower() for cb in disabled_callbacks.split(",")])
            return list(disabled_callbacks)
        

        #########################################################
        # check if disabled via request body
        #########################################################
        if standard_callback_dynamic_params.get("litellm_disabled_callbacks", None) is not None:
            return standard_callback_dynamic_params.get("litellm_disabled_callbacks", None)
        
        return None
    
    @staticmethod
    def _should_allow_dynamic_callback_disabling():
        import litellm
        from litellm.proxy.proxy_server import premium_user

        # Check if admin has disabled this feature
        if litellm.allow_dynamic_callback_disabling is not True:
            verbose_logger.debug("Dynamic callback disabling is disabled by admin via litellm.allow_dynamic_callback_disabling")
            return False
        
        if premium_user:
            return True
        verbose_logger.warning(f"Disabling callbacks using request headers is an enterprise feature. {CommonProxyErrors.not_premium_user.value}")
        return False