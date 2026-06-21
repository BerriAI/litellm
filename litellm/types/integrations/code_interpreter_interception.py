"""
Type definitions for Code Interpreter Interception integration.
"""

from typing import List, TypedDict


class CodeInterpreterInterceptionConfig(TypedDict, total=False):
    """
    Configuration parameters for CodeInterpreterInterceptionLogger.

    Used in proxy_config.yaml under litellm_settings:
        litellm_settings:
          code_interpreter_interception_params:
            enabled: true
            enabled_providers: ["openai"]
            sandbox_tool_name: "my_e2b_sandbox"
    """

    enabled: bool
    enabled_providers: List[str]
    sandbox_tool_name: str
