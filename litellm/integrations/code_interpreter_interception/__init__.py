"""
Code Interpreter Interception Module

Converts the native OpenAI Responses ``code_interpreter`` tool into a function
tool, runs the model-emitted code in a sandbox, and feeds the result back into
the agentic loop.
"""

from litellm.integrations.code_interpreter_interception.handler import (
    CodeInterpreterInterceptionLogger,
)

__all__ = [
    "CodeInterpreterInterceptionLogger",
]
