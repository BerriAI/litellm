"""
LiteLLM Skills Hook - Proxy integration for skills

This module provides the CustomLogger hook for skills processing.
The actual skill logic is in litellm/llms/litellm_proxy/skills/.

Usage:
    from litellm.proxy.hooks.litellm_skills import SkillsInjectionHook
    
    # Register hook in proxy
    litellm.callbacks.append(SkillsInjectionHook())
"""

# Re-export from the SDK location for convenience
from litellm.llms.litellm_proxy.skills import (
    LITELLM_CODE_EXECUTION_TOOL,
    CodeExecutionHandler,
    LiteLLMInternalTools,
    SkillPromptInjectionHandler,
    SkillsSandboxExecutor,
    code_execution_handler,
    get_litellm_code_execution_tool,
)
from litellm.proxy.hooks.litellm_skills.main import (
    SkillsInjectionHook,
    skills_injection_hook,
)

__all__ = [
    "SkillsInjectionHook",
    "skills_injection_hook",
    "CodeExecutionHandler",
    "LiteLLMInternalTools",
    "LITELLM_CODE_EXECUTION_TOOL",
    "get_litellm_code_execution_tool",
    "code_execution_handler",
    "SkillPromptInjectionHandler",
    "SkillsSandboxExecutor",
]
