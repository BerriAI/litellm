"""
This file is used to store the state variables of the proxy server.

Example: `spend_logs_row_count` is used to store the number of rows in the `LiteLLM_SpendLogs` table.
"""

from typing import Any, Literal

from litellm.proxy._types import ProxyStateVariables


class ProxyState:
    """
    Proxy state class has get/set methods for Proxy state variables.
    """

    # Note: mypy does not recognize when we fetch ProxyStateVariables.annotations.keys(), so we also need to add the valid keys here
    valid_keys_literal = Literal["spend_logs_row_count"]

    def __init__(self) -> None:
        self.proxy_state_variables: ProxyStateVariables = ProxyStateVariables(
            spend_logs_row_count=0,
        )

    def get_proxy_state_variable(
        self,
        variable_name: valid_keys_literal,
    ) -> Any:
        return self.proxy_state_variables.get(variable_name, None)

    def set_proxy_state_variable(
        self,
        variable_name: valid_keys_literal,
        value: Any,
    ) -> None:
        self.proxy_state_variables[variable_name] = value
