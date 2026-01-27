"""
LiteLLM Proxy Skills - Database-backed skills storage and execution

This module provides:
- Database-backed skills storage (alternative to Anthropic's cloud-based skills API)
- Skill content extraction and prompt injection
- Sandboxed code execution for skills
- Automatic code execution handler

Main components:
- handler.py: LiteLLMSkillsHandler - database CRUD operations
- transformation.py: LiteLLMSkillsTransformationHandler - SDK transformation layer
- prompt_injection.py: SkillPromptInjectionHandler - SKILL.md extraction and injection
- sandbox_executor.py: SkillsSandboxExecutor - Docker sandbox execution
- code_execution.py: CodeExecutionHandler - automatic agentic loop
"""

from litellm.llms.litellm_proxy.skills.code_execution import (
    LITELLM_CODE_EXECUTION_TOOL,
    CodeExecutionHandler,
    LiteLLMInternalTools,
    add_code_execution_tool,
    code_execution_handler,
    get_litellm_code_execution_tool,
    has_code_execution_tool,
)
from litellm.llms.litellm_proxy.skills.constants import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_SANDBOX_TIMEOUT,
)
from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler
from litellm.llms.litellm_proxy.skills.prompt_injection import (
    SkillPromptInjectionHandler,
)
from litellm.llms.litellm_proxy.skills.sandbox_executor import SkillsSandboxExecutor
from litellm.llms.litellm_proxy.skills.transformation import (
    LiteLLMSkillsTransformationHandler,
)

__all__ = [
    "LiteLLMSkillsHandler",
    "LiteLLMSkillsTransformationHandler",
    "SkillPromptInjectionHandler",
    "SkillsSandboxExecutor",
    "CodeExecutionHandler",
    "LiteLLMInternalTools",
    "LITELLM_CODE_EXECUTION_TOOL",
    "get_litellm_code_execution_tool",
    "code_execution_handler",
    "has_code_execution_tool",
    "add_code_execution_tool",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_SANDBOX_TIMEOUT",
]
