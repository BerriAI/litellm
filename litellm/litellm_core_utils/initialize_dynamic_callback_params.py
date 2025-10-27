from typing import Dict, Optional

from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import StandardCallbackDynamicParams


def initialize_standard_callback_dynamic_params(
    kwargs: Optional[Dict] = None,
) -> StandardCallbackDynamicParams:
    """
    Initialize the standard callback dynamic params from the kwargs

    checks if langfuse_secret_key, gcs_bucket_name in kwargs and sets the corresponding attributes in StandardCallbackDynamicParams
    """

    standard_callback_dynamic_params = StandardCallbackDynamicParams()
    if kwargs:
        _supported_callback_params = (
            StandardCallbackDynamicParams.__annotations__.keys()
        )

        for param in _supported_callback_params:
            if param in kwargs:
                _param_value = kwargs.pop(param)
                if (
                    _param_value is not None
                    and isinstance(_param_value, str)
                    and "os.environ/" in _param_value
                ):
                    _param_value = get_secret_str(secret_name=_param_value)
                standard_callback_dynamic_params[param] = _param_value  # type: ignore

    return standard_callback_dynamic_params
