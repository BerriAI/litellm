"""
Automatic Code Execution Handler for LiteLLM Skills

When `litellm_code_execution` tool is present, this handler automatically:
1. Makes the LLM call
2. Executes any code the model generates
3. Continues the conversation with results
4. Returns final response with generated files inline (base64)

This mimics Anthropic's behavior where code execution happens automatically.
Generated files are returned directly in the response - no separate storage needed.
"""

import base64
import json
from enum import Enum
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger


class LiteLLMInternalTools(str, Enum):
    """
    Enum for internal LiteLLM tools that are injected into requests.
    
    These tools are handled automatically by LiteLLM hooks and are not
    passed to the underlying LLM provider directly.
    """
    CODE_EXECUTION = "litellm_code_execution"


def get_litellm_code_execution_tool() -> Dict[str, Any]:
    """
    Returns the litellm_code_execution tool definition in OpenAI format.
    
    This tool enables automatic code execution in a sandboxed environment
    when skills include executable Python code.
    """
    return {
        "type": "function",
        "function": {
            "name": LiteLLMInternalTools.CODE_EXECUTION.value,
            "description": "Execute Python code in a sandboxed environment. Use this to run code that generates files, processes data, or performs computations. Generated files will be returned directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    }


def get_litellm_code_execution_tool_anthropic() -> Dict[str, Any]:
    """
    Returns the litellm_code_execution tool definition in Anthropic/messages API format.
    
    This tool enables automatic code execution in a sandboxed environment
    when skills include executable Python code.
    """
    return {
        "name": LiteLLMInternalTools.CODE_EXECUTION.value,
        "description": "Execute Python code in a sandboxed environment. Use this to run code that generates files, processes data, or performs computations. Generated files will be returned directly.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    }


# Singleton tool definition for backwards compatibility
LITELLM_CODE_EXECUTION_TOOL = get_litellm_code_execution_tool()


class CodeExecutionHandler:
    """
    Handles automatic code execution for LiteLLM skills.
    
    When enabled, this handler intercepts LLM responses with code execution
    tool calls, executes them in a sandbox, and continues the conversation
    automatically until completion.
    """
    
    def __init__(
        self,
        max_iterations: Optional[int] = None,
        sandbox_timeout: Optional[int] = None,
    ):
        from litellm.llms.litellm_proxy.skills.constants import (
            DEFAULT_MAX_ITERATIONS,
            DEFAULT_SANDBOX_TIMEOUT,
        )
        
        self.max_iterations = max_iterations or DEFAULT_MAX_ITERATIONS
        self.sandbox_timeout = sandbox_timeout or DEFAULT_SANDBOX_TIMEOUT
    
    async def execute_with_code_execution(
        self,
        model: str,
        messages: List[Dict],
        tools: List[Dict],
        skill_files: Dict[str, bytes],
        skill_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute an LLM call with automatic code execution handling.
        
        This method:
        1. Makes the initial LLM call
        2. If model calls litellm_code_execution, executes the code
        3. Continues conversation with results
        4. Repeats until model stops calling tools
        5. Returns final response with generated files inline
        
        Args:
            model: Model to use
            messages: Initial messages
            tools: Tools including litellm_code_execution
            skill_files: Dict of skill files for execution
            skill_id: Optional skill ID for tracking
            **kwargs: Additional args for litellm.acompletion
            
        Returns:
            Dict with:
            - response: Final LLM response
            - files: List of generated files with content (base64)
            - execution_results: List of code execution results
        """
        import litellm
        from litellm.llms.litellm_proxy.skills.sandbox_executor import (
            SkillsSandboxExecutor,
        )
        
        current_messages = list(messages)
        generated_files: List[Dict[str, Any]] = []  # Files returned directly
        execution_results: List[Dict] = []
        
        executor = SkillsSandboxExecutor(timeout=self.sandbox_timeout)
        response: Any = None  # Initialize to avoid possibly unbound error
        
        for iteration in range(self.max_iterations):
            verbose_logger.debug(
                f"CodeExecutionHandler: Iteration {iteration + 1}/{self.max_iterations}"
            )
            
            # Make LLM call
            response = await litellm.acompletion(
                model=model,
                messages=current_messages,
                tools=tools,
                **kwargs,
            )
            
            assistant_message = response.choices[0].message  # type: ignore
            stop_reason = response.choices[0].finish_reason  # type: ignore
            
            # Build assistant message for conversation history
            assistant_msg_dict: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_message.content,
            }
            if assistant_message.tool_calls:
                assistant_msg_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            current_messages.append(assistant_msg_dict)
            
            # Check if we're done (no tool calls or not tool_calls finish reason)
            if stop_reason != "tool_calls" or not assistant_message.tool_calls:
                verbose_logger.debug(
                    f"CodeExecutionHandler: Completed after {iteration + 1} iterations"
                )
                return {
                    "response": response,
                    "files": generated_files,  # Files returned directly with base64 content
                    "execution_results": execution_results,
                    "messages": current_messages,
                }
            
            # Handle tool calls
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                
                if tool_name == LiteLLMInternalTools.CODE_EXECUTION.value:
                    # Execute code in sandbox
                    try:
                        args = json.loads(tool_call.function.arguments)
                        code = args.get("code", "")
                        
                        verbose_logger.debug(
                            f"CodeExecutionHandler: Executing code ({len(code)} chars)"
                        )
                        
                        exec_result = executor.execute(
                            code=code,
                            skill_files=skill_files,
                        )

                        verbose_logger.debug(
                            f"CodeExecutionHandler: Execution result: {exec_result}"
                        )
                        
                        execution_results.append({
                            "iteration": iteration,
                            "success": exec_result["success"],
                            "output": exec_result["output"],
                            "error": exec_result["error"],
                            "files": [f["name"] for f in exec_result["files"]],
                        })
                        
                        # Build tool result content
                        tool_result = exec_result["output"] or ""
                        
                        # Collect generated files (returned directly, no storage)
                        if exec_result["files"]:
                            tool_result += "\n\nGenerated files:"
                            for f in exec_result["files"]:
                                file_content = base64.b64decode(f["content_base64"])
                                # Add to generated files list (returned in response)
                                generated_files.append({
                                    "name": f["name"],
                                    "mime_type": f["mime_type"],
                                    "content_base64": f["content_base64"],
                                    "size": len(file_content),
                                })
                                tool_result += f"\n- {f['name']} ({len(file_content)} bytes)"
                                
                                verbose_logger.debug(
                                    f"CodeExecutionHandler: Generated file {f['name']} ({len(file_content)} bytes)"
                                )
                        
                        if exec_result["error"]:
                            tool_result += f"\n\nError:\n{exec_result['error']}"
                        
                    except Exception as e:
                        tool_result = f"Code execution failed: {str(e)}"
                        execution_results.append({
                            "iteration": iteration,
                            "success": False,
                            "error": str(e),
                        })
                    
                    # Add tool result to messages
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    })
                else:
                    # Non-code-execution tool - pass through
                    # In a full implementation, this would call other tool handlers
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Tool '{tool_name}' not handled by code execution handler",
                    })
        
        # Max iterations reached
        verbose_logger.warning(
            f"CodeExecutionHandler: Max iterations ({self.max_iterations}) reached"
        )
        return {
            "response": response,
            "files": generated_files,
            "execution_results": execution_results,
            "messages": current_messages,
            "max_iterations_reached": True,
        }


def has_code_execution_tool(tools: Optional[List[Dict]]) -> bool:
    """Check if litellm_code_execution tool is in the tools list."""
    if not tools:
        return False
    for tool in tools:
        func = tool.get("function", {})
        if func.get("name") == LiteLLMInternalTools.CODE_EXECUTION.value:
            return True
    return False


def add_code_execution_tool(tools: Optional[List[Dict]]) -> List[Dict]:
    """Add litellm_code_execution tool if not already present."""
    tools = tools or []
    if not has_code_execution_tool(tools):
        tools.append(LITELLM_CODE_EXECUTION_TOOL)
    return tools


# Global handler instance
code_execution_handler = CodeExecutionHandler()

