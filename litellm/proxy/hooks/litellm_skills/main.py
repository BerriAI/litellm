"""
Skills Injection Hook for LiteLLM Proxy

Main hook that orchestrates skill processing:
- Fetches skills from LiteLLM DB
- Injects SKILL.md content into system prompt
- Adds litellm_code_execution tool for automatic code execution
- Handles agentic loop internally when litellm_code_execution is called

For non-Anthropic models (e.g., Bedrock, OpenAI, etc.):
- Skills are converted to OpenAI-style tools
- Skill file content (SKILL.md) is extracted and injected into the system prompt
- litellm_code_execution tool is added - when model calls it, LiteLLM handles
  execution automatically and returns final response with file_ids

Usage:
    # Simple - LiteLLM handles everything automatically via proxy
    # The container parameter triggers the SkillsInjectionHook
    response = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Create a bouncing ball GIF"}],
        container={"skills": [{"skill_id": "litellm:skill_abc123"}]},
    )
    # Response includes file_ids for generated files
"""

import base64
import json
from typing import Any, Dict, List, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.litellm_proxy.skills.prompt_injection import (
    SkillPromptInjectionHandler,
)
from litellm.proxy._types import LiteLLM_SkillsTable, UserAPIKeyAuth
from litellm.types.utils import CallTypes, CallTypesLiteral


class SkillsInjectionHook(CustomLogger):
    """
    Pre/Post-call hook that processes skills from container.skills parameter.

    Pre-call (async_pre_call_hook):
    - Skills with 'litellm:' prefix are fetched from LiteLLM DB
    - For Anthropic models: native skills pass through, LiteLLM skills converted to tools
    - For non-Anthropic models: LiteLLM skills are converted to tools + execute_code tool
    
    Post-call (async_post_call_success_deployment_hook):
    - If response has litellm_code_execution tool call, automatically execute code
    - Continue conversation loop until model gives final response
    - Return response with generated files inline
    
    This hook is called automatically by litellm during completion calls.
    """

    def __init__(self, **kwargs):
        from litellm.llms.litellm_proxy.skills.constants import (
            DEFAULT_MAX_ITERATIONS,
            DEFAULT_SANDBOX_TIMEOUT,
        )
        
        self.optional_params = kwargs
        self.prompt_handler = SkillPromptInjectionHandler()
        self.max_iterations = kwargs.get("max_iterations", DEFAULT_MAX_ITERATIONS)
        self.sandbox_timeout = kwargs.get("sandbox_timeout", DEFAULT_SANDBOX_TIMEOUT)
        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Process skills from container.skills before the LLM call.

        1. Check if container.skills exists in request
        2. Separate skills by prefix (litellm: vs native)
        3. Fetch LiteLLM skills from database
        4. For Anthropic: keep native skills in container
        5. For non-Anthropic: convert LiteLLM skills to tools, inject content, add execute_code
        """
        # Only process completion-type calls
        if call_type not in ["completion", "acompletion", "anthropic_messages"]:
            return data

        container = data.get("container")
        if not container or not isinstance(container, dict):
            return data

        skills = container.get("skills")
        if not skills or not isinstance(skills, list):
            return data

        verbose_proxy_logger.debug(f"SkillsInjectionHook: Processing {len(skills)} skills")

        litellm_skills: List[LiteLLM_SkillsTable] = []
        anthropic_skills: List[Dict[str, Any]] = []

        # Separate skills by prefix
        for skill in skills:
            if not isinstance(skill, dict):
                continue

            skill_id = skill.get("skill_id", "")
            if skill_id.startswith("litellm_"):
                # Fetch from LiteLLM DB
                db_skill = await self._fetch_skill_from_db(skill_id)
                if db_skill:
                    litellm_skills.append(db_skill)
                else:
                    verbose_proxy_logger.warning(
                        f"SkillsInjectionHook: Skill '{skill_id}' not found in LiteLLM DB"
                    )
            else:
                # Native Anthropic skill - pass through
                anthropic_skills.append(skill)

        # Check if using messages API spec (anthropic_messages call type)
        # Messages API always uses Anthropic-style tool format
        use_anthropic_format = call_type == "anthropic_messages"

        if len(litellm_skills) > 0:
            data = self._process_for_messages_api(
                data=data,
                litellm_skills=litellm_skills,
                use_anthropic_format=use_anthropic_format,
            )

        return data


    def _process_for_messages_api(
        self,
        data: dict,
        litellm_skills: List[LiteLLM_SkillsTable],
        use_anthropic_format: bool = True,
    ) -> dict:
        """
        Process skills for messages API (Anthropic format tools).
        
        - Converts skills to Anthropic-style tools (name, description, input_schema)
        - Extracts and injects SKILL.md content into system prompt
        - Adds litellm_code_execution tool for code execution
        - Stores skill files in metadata for sandbox execution
        """
        from litellm.llms.litellm_proxy.skills.code_execution import (
            get_litellm_code_execution_tool_anthropic,
        )
        
        tools = data.get("tools", [])
        skill_contents: List[str] = []
        all_skill_files: Dict[str, Dict[str, bytes]] = {}
        all_module_paths: List[str] = []

        for skill in litellm_skills:
            # Convert skill to Anthropic-style tool
            tools.append(self.prompt_handler.convert_skill_to_anthropic_tool(skill))
            
            # Extract skill content from file if available
            content = self.prompt_handler.extract_skill_content(skill)
            if content:
                skill_contents.append(content)
            
            # Extract all files for code execution
            skill_files = self.prompt_handler.extract_all_files(skill)
            if skill_files:
                all_skill_files[skill.skill_id] = skill_files
                for path in skill_files.keys():
                    if path.endswith(".py"):
                        all_module_paths.append(path)

        if tools:
            data["tools"] = tools

        # Inject skill content into system prompt
        # For Anthropic messages API, use top-level 'system' param instead of messages array
        if skill_contents:
            data = self.prompt_handler.inject_skill_content_to_messages(
                data, skill_contents, use_anthropic_format=use_anthropic_format
            )

        # Add litellm_code_execution tool if we have skill files
        if all_skill_files:
            code_exec_tool = get_litellm_code_execution_tool_anthropic()
            data["tools"] = data.get("tools", []) + [code_exec_tool]
            
            # Store skill files in litellm_metadata for automatic code execution
            data["litellm_metadata"] = data.get("litellm_metadata", {})
            data["litellm_metadata"]["_skill_files"] = all_skill_files
            data["litellm_metadata"]["_litellm_code_execution_enabled"] = True

        # Remove container (not supported by underlying providers)
        data.pop("container", None)

        verbose_proxy_logger.debug(
            f"SkillsInjectionHook: Messages API - converted {len(litellm_skills)} skills to Anthropic tools, "
            f"injected {len(skill_contents)} skill contents, "
            f"added litellm_code_execution tool with {len(all_module_paths)} modules"
        )

        return data

    def _process_non_anthropic_model(
        self,
        data: dict,
        litellm_skills: List[LiteLLM_SkillsTable],
    ) -> dict:
        """
        Process skills for non-Anthropic models (OpenAI format tools).
        
        - Converts skills to OpenAI-style tools
        - Extracts and injects SKILL.md content
        - Adds execute_code tool for code execution
        - Stores skill files in metadata for sandbox execution
        """
        tools = data.get("tools", [])
        skill_contents: List[str] = []
        all_skill_files: Dict[str, Dict[str, bytes]] = {}
        all_module_paths: List[str] = []

        for skill in litellm_skills:
            # Convert skill to OpenAI-style tool
            tools.append(self.prompt_handler.convert_skill_to_tool(skill))
            
            # Extract skill content from file if available
            content = self.prompt_handler.extract_skill_content(skill)
            if content:
                skill_contents.append(content)
            
            # Extract all files for code execution
            skill_files = self.prompt_handler.extract_all_files(skill)
            if skill_files:
                all_skill_files[skill.skill_id] = skill_files
                # Collect Python module paths
                for path in skill_files.keys():
                    if path.endswith(".py"):
                        all_module_paths.append(path)

        if tools:
            data["tools"] = tools

        # Inject skill content into system prompt
        if skill_contents:
            data = self.prompt_handler.inject_skill_content_to_messages(data, skill_contents)

        # Add litellm_code_execution tool if we have skill files
        if all_skill_files:
            from litellm.llms.litellm_proxy.skills.code_execution import (
                get_litellm_code_execution_tool,
            )
            data["tools"] = data.get("tools", []) + [get_litellm_code_execution_tool()]
            
            # Store skill files in litellm_metadata for automatic code execution
            # Using litellm_metadata instead of metadata to avoid conflicts with user metadata
            data["litellm_metadata"] = data.get("litellm_metadata", {})
            data["litellm_metadata"]["_skill_files"] = all_skill_files
            data["litellm_metadata"]["_litellm_code_execution_enabled"] = True

        # Remove container for non-Anthropic (they don't support it)
        data.pop("container", None)

        verbose_proxy_logger.debug(
            f"SkillsInjectionHook: Non-Anthropic model - converted {len(litellm_skills)} skills to tools, "
            f"injected {len(skill_contents)} skill contents, "
            f"added execute_code tool with {len(all_module_paths)} modules"
        )

        return data

    async def _fetch_skill_from_db(self, skill_id: str) -> Optional[LiteLLM_SkillsTable]:
        """
        Fetch a skill from the LiteLLM database.

        Args:
            skill_id: The skill ID (without 'litellm:' prefix)

        Returns:
            LiteLLM_SkillsTable or None if not found
        """
        try:
            from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler

            return await LiteLLMSkillsHandler.fetch_skill_from_db(skill_id)
        except Exception as e:
            verbose_proxy_logger.warning(
                f"SkillsInjectionHook: Error fetching skill {skill_id}: {e}"
            )
            return None

    def _is_anthropic_model(self, model: str) -> bool:
        """
        Check if the model is an Anthropic model using get_llm_provider.

        Args:
            model: The model name/identifier

        Returns:
            True if Anthropic model, False otherwise
        """
        try:
            from litellm.litellm_core_utils.get_llm_provider_logic import (
                get_llm_provider,
            )

            _, custom_llm_provider, _, _ = get_llm_provider(model=model)
            return custom_llm_provider == "anthropic"
        except Exception:
            # Fallback to simple check if get_llm_provider fails
            return "claude" in model.lower() or model.lower().startswith("anthropic/")

    async def async_post_call_success_deployment_hook(
        self,
        request_data: dict,
        response: Any,
        call_type: Optional[CallTypes],
    ) -> Optional[Any]:
        """
        Post-call hook to handle automatic code execution.
        
        Handles both OpenAI format (response.choices) and Anthropic/messages API 
        format (response["content"]).
        
        If the response contains a tool call (litellm_code_execution or skill tool):
        1. Execute the code in sandbox
        2. Add result to messages
        3. Make another LLM call
        4. Repeat until model gives final response
        5. Return modified response with generated files
        """
        from litellm.llms.litellm_proxy.skills.code_execution import (
            LiteLLMInternalTools,
        )

        # Check if code execution is enabled for this request
        litellm_metadata = request_data.get("litellm_metadata") or {}
        metadata = request_data.get("metadata") or {}
        
        code_exec_enabled = (
            litellm_metadata.get("_litellm_code_execution_enabled") or
            metadata.get("_litellm_code_execution_enabled")
        )
        if not code_exec_enabled:
            return None
        
        # Get skill files
        skill_files_by_id = (
            litellm_metadata.get("_skill_files") or
            metadata.get("_skill_files", {})
        )
        all_skill_files: Dict[str, bytes] = {}
        for files_dict in skill_files_by_id.values():
            all_skill_files.update(files_dict)
        
        if not all_skill_files:
            verbose_proxy_logger.warning(
                "SkillsInjectionHook: No skill files found, cannot execute code"
            )
            return None
        
        # Check for tool calls - handle both Anthropic and OpenAI formats
        tool_calls = self._extract_tool_calls(response)
        if not tool_calls:
            return None
        
        # Check if any tool call needs execution (litellm_code_execution or skill tool)
        has_executable_tool = False
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            # Execute if it's litellm_code_execution OR a skill tool (skill_xxx)
            if tool_name == LiteLLMInternalTools.CODE_EXECUTION.value or tool_name.startswith("skill_"):
                has_executable_tool = True
                break
        
        if not has_executable_tool:
            return None
        
        verbose_proxy_logger.debug(
            "SkillsInjectionHook: Detected tool call, starting execution loop"
        )
        
        # Start the agentic loop
        return await self._execute_code_loop_messages_api(
            data=request_data,
            response=response,
            skill_files=all_skill_files,
        )
    
    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """Extract tool calls from response, handling both formats."""
        tool_calls = []
        
        # Get content - handle both dict and object responses
        content = None
        if isinstance(response, dict):
            content = response.get("content", [])
        elif hasattr(response, "content"):
            content = response.content
        
        # Anthropic/messages API format: response has "content" list with tool_use blocks
        if content:
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input", {}),
                    })
                elif hasattr(block, "type") and getattr(block, "type", None) == "tool_use":
                    tool_calls.append({
                        "id": getattr(block, "id", None),
                        "name": getattr(block, "name", None),
                        "input": getattr(block, "input", {}),
                    })
        
        # OpenAI format: response has choices[0].message.tool_calls
        if not tool_calls and hasattr(response, "choices") and response.choices:  # type: ignore[union-attr]
            msg = response.choices[0].message  # type: ignore[union-attr]
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments) if tc.function.arguments else {},
                    })
        
        return tool_calls

    async def _execute_code_loop_messages_api(
        self,
        data: dict,
        response: Any,
        skill_files: Dict[str, bytes],
    ) -> Any:
        """
        Execute the code execution loop for messages API (Anthropic format).
        
        Returns the final response with generated files inline.
        """
        import litellm
        from litellm.llms.litellm_proxy.skills.code_execution import (
            LiteLLMInternalTools,
        )
        from litellm.llms.litellm_proxy.skills.sandbox_executor import (
            SkillsSandboxExecutor,
        )

        # Ensure response is not None
        if response is None:
            verbose_proxy_logger.error(
                "SkillsInjectionHook: Response is None, cannot execute code loop"
            )
            return None
        
        model = data.get("model", "")
        messages = list(data.get("messages", []))
        tools = data.get("tools", [])
        max_tokens = data.get("max_tokens", 4096)
        
        executor = SkillsSandboxExecutor(timeout=self.sandbox_timeout)
        generated_files: List[Dict[str, Any]] = []
        current_response = response
        
        for iteration in range(self.max_iterations):
            # Extract tool calls from current response
            tool_calls = self._extract_tool_calls(current_response)
            stop_reason = current_response.get("stop_reason") if isinstance(current_response, dict) else getattr(current_response, "stop_reason", None)
            
            # Get content for assistant message - convert to plain dicts
            raw_content = current_response.get("content", []) if isinstance(current_response, dict) else getattr(current_response, "content", [])
            content_blocks = []
            for block in raw_content or []:
                if isinstance(block, dict):
                    content_blocks.append(block)
                elif hasattr(block, "model_dump"):
                    content_blocks.append(block.model_dump())
                elif hasattr(block, "__dict__"):
                    content_blocks.append(dict(block.__dict__))
                else:
                    content_blocks.append({"type": "text", "text": str(block)})
            
            # Build assistant message for conversation history (Anthropic format)
            assistant_msg = {"role": "assistant", "content": content_blocks}
            messages.append(assistant_msg)
            
            # Check if we're done (no tool calls)
            if stop_reason != "tool_use" or not tool_calls:
                verbose_proxy_logger.debug(
                    f"SkillsInjectionHook: Loop completed after {iteration + 1} iterations, "
                    f"{len(generated_files)} files generated"
                )
                return self._attach_files_to_response(current_response, generated_files)
            
            # Process tool calls
            tool_results = []
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_id = tc.get("id", "")
                tool_input = tc.get("input", {})
                
                # Execute if it's litellm_code_execution OR a skill tool
                if tool_name == LiteLLMInternalTools.CODE_EXECUTION.value:
                    code = tool_input.get("code", "")
                    result = await self._execute_code(code, skill_files, executor, generated_files)
                elif tool_name.startswith("skill_"):
                    # Skill tool - execute the skill's code
                    result = await self._execute_skill_tool(tool_name, tool_input, skill_files, executor, generated_files)
                else:
                    result = f"Tool '{tool_name}' not handled"
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result,
                })
            
            # Add tool results to messages (Anthropic format)
            messages.append({"role": "user", "content": tool_results})
            
            # Make next LLM call
            verbose_proxy_logger.debug(
                f"SkillsInjectionHook: Making LLM call iteration {iteration + 2}"
            )
            try:
                current_response = await litellm.anthropic.acreate(
                    model=model,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                )
                if current_response is None:
                    verbose_proxy_logger.error(
                        "SkillsInjectionHook: LLM call returned None"
                    )
                    return self._attach_files_to_response(response, generated_files)
            except Exception as e:
                verbose_proxy_logger.error(
                    f"SkillsInjectionHook: LLM call failed: {e}"
                )
                return self._attach_files_to_response(response, generated_files)
        
        verbose_proxy_logger.warning(
            f"SkillsInjectionHook: Max iterations ({self.max_iterations}) reached"
        )
        return self._attach_files_to_response(current_response, generated_files)

    async def _execute_code(
        self,
        code: str,
        skill_files: Dict[str, bytes],
        executor: Any,
        generated_files: List[Dict[str, Any]],
    ) -> str:
        """Execute code in sandbox and return result string."""
        try:
            verbose_proxy_logger.debug(f"SkillsInjectionHook: Executing code ({len(code)} chars)")
            
            exec_result = executor.execute(code=code, skill_files=skill_files)
            
            result = exec_result.get("output", "") or ""
            
            # Collect generated files
            if exec_result.get("files"):
                for f in exec_result["files"]:
                    generated_files.append({
                        "name": f["name"],
                        "mime_type": f["mime_type"],
                        "content_base64": f["content_base64"],
                        "size": len(base64.b64decode(f["content_base64"])),
                    })
                    result += f"\n\nGenerated file: {f['name']}"
            
            if exec_result.get("error"):
                result += f"\n\nError: {exec_result['error']}"
            
            return result or "Code executed successfully"
        except Exception as e:
            return f"Code execution failed: {str(e)}"

    async def _execute_skill_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        skill_files: Dict[str, bytes],
        executor: Any,
        generated_files: List[Dict[str, Any]],
    ) -> str:
        """Execute a skill tool by generating and running code based on skill content."""
        # Generate code based on available skill modules
        # Look for Python modules in the skill
        python_modules = [p for p in skill_files.keys() if p.endswith(".py") and not p.endswith("__init__.py")]
        
        # Try to find the main builder/creator module
        main_module = None
        for mod in python_modules:
            if "builder" in mod.lower() or "creator" in mod.lower() or "generator" in mod.lower():
                main_module = mod
                break
        
        if not main_module and python_modules:
            # Use first non-init module
            main_module = python_modules[0]
        
        if main_module:
            # Convert path to import: "core/gif_builder.py" -> "core.gif_builder"
            import_path = main_module.replace("/", ".").replace(".py", "")
            
            # Generate code that imports and uses the module
            code = f"""
# Auto-generated code to execute skill
import sys
sys.path.insert(0, '/sandbox')

from {import_path} import *

# Try to find and use a Builder/Creator class
import inspect
module = __import__('{import_path}', fromlist=[''])

for name, obj in inspect.getmembers(module):
    if inspect.isclass(obj) and name != 'object':
        try:
            instance = obj()
            # Try common methods
            if hasattr(instance, 'create'):
                result = instance.create()
            elif hasattr(instance, 'build'):
                result = instance.build()
            elif hasattr(instance, 'generate'):
                result = instance.generate()
            elif hasattr(instance, 'save'):
                instance.save('output.gif')
            print(f'Used {{name}} class')
            break
        except Exception as e:
            print(f'Error with {{name}}: {{e}}')
            continue

# List generated files
import os
for f in os.listdir('.'):
    if f.endswith(('.gif', '.png', '.jpg')):
        print(f'Generated: {{f}}')
"""
        else:
            # Fallback generic code
            code = """
print('No executable skill module found')
"""
        
        return await self._execute_code(code, skill_files, executor, generated_files)

    async def _execute_code_loop(
        self,
        data: dict,
        response: Any,
        skill_files: Dict[str, bytes],
    ) -> Any:
        """
        Execute the code execution loop until model gives final response.
        
        Returns the final response with generated files inline.
        """
        import litellm
        from litellm.llms.litellm_proxy.skills.code_execution import (
            LiteLLMInternalTools,
        )
        from litellm.llms.litellm_proxy.skills.sandbox_executor import (
            SkillsSandboxExecutor,
        )
        
        model = data.get("model", "")
        messages = list(data.get("messages", []))
        tools = data.get("tools", [])
        
        # Keys to exclude when passing through to acompletion
        # These are either handled explicitly or are internal LiteLLM fields
        _EXCLUDED_ACOMPLETION_KEYS = frozenset({
            "messages",
            "model", 
            "tools",
            "metadata",
            "litellm_metadata",
            "container",
        })
        
        kwargs = {
            k: v for k, v in data.items() 
            if k not in _EXCLUDED_ACOMPLETION_KEYS
        }
        
        executor = SkillsSandboxExecutor(timeout=self.sandbox_timeout)
        generated_files: List[Dict[str, Any]] = []
        current_response: Any = response
        
        for iteration in range(self.max_iterations):
            # OpenAI format response has choices[0].message
            assistant_message = current_response.choices[0].message  # type: ignore[union-attr]
            stop_reason = current_response.choices[0].finish_reason  # type: ignore[union-attr]
            
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
            messages.append(assistant_msg_dict)
            
            # Check if we're done (no tool calls)
            if stop_reason != "tool_calls" or not assistant_message.tool_calls:
                verbose_proxy_logger.debug(
                    f"SkillsInjectionHook: Code execution loop completed after "
                    f"{iteration + 1} iterations, {len(generated_files)} files generated"
                )
                # Attach generated files to response
                return self._attach_files_to_response(current_response, generated_files)
            
            # Process tool calls
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                
                if tool_name == LiteLLMInternalTools.CODE_EXECUTION.value:
                    tool_result = await self._execute_code_tool(
                        tool_call=tool_call,
                        skill_files=skill_files,
                        executor=executor,
                        generated_files=generated_files,
                    )
                else:
                    # Non-code-execution tool - cannot handle
                    tool_result = f"Tool '{tool_name}' not handled automatically"
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })
            
            # Make next LLM call using the messages API
            verbose_proxy_logger.debug(
                f"SkillsInjectionHook: Making LLM call iteration {iteration + 2}"
            )
            current_response = await litellm.anthropic.acreate(
                model=model,
                messages=messages,
                tools=tools,
                max_tokens=kwargs.get("max_tokens", 4096),
            )
        
        # Max iterations reached
        verbose_proxy_logger.warning(
            f"SkillsInjectionHook: Max iterations ({self.max_iterations}) reached"
        )
        return self._attach_files_to_response(current_response, generated_files)
    
    async def _execute_code_tool(
        self,
        tool_call: Any,
        skill_files: Dict[str, bytes],
        executor: Any,
        generated_files: List[Dict[str, Any]],
    ) -> str:
        """Execute a litellm_code_execution tool call and return result string."""
        try:
            args = json.loads(tool_call.function.arguments)
            code = args.get("code", "")
            
            verbose_proxy_logger.debug(
                f"SkillsInjectionHook: Executing code ({len(code)} chars)"
            )
            
            exec_result = executor.execute(
                code=code,
                skill_files=skill_files,
            )
            
            # Build tool result content
            tool_result = exec_result.get("output", "") or ""
            
            # Collect generated files
            if exec_result.get("files"):
                tool_result += "\n\nGenerated files:"
                for f in exec_result["files"]:
                    file_content = base64.b64decode(f["content_base64"])
                    generated_files.append({
                        "name": f["name"],
                        "mime_type": f["mime_type"],
                        "content_base64": f["content_base64"],
                        "size": len(file_content),
                    })
                    tool_result += f"\n- {f['name']} ({len(file_content)} bytes)"
                    
                    verbose_proxy_logger.debug(
                        f"SkillsInjectionHook: Generated file {f['name']} "
                        f"({len(file_content)} bytes)"
                    )
            
            if exec_result.get("error"):
                tool_result += f"\n\nError:\n{exec_result['error']}"
            
            return tool_result
            
        except Exception as e:
            verbose_proxy_logger.error(
                f"SkillsInjectionHook: Code execution failed: {e}"
            )
            return f"Code execution failed: {str(e)}"
    
    def _attach_files_to_response(
        self,
        response: Any,
        generated_files: List[Dict[str, Any]],
    ) -> Any:
        """
        Attach generated files to the response object.
        
        Files are added to response._litellm_generated_files for easy access.
        For dict responses, files are added as a key.
        """
        if not generated_files:
            return response
        
        # Handle dict response (Anthropic/messages API format)
        if isinstance(response, dict):
            response["_litellm_generated_files"] = generated_files
            verbose_proxy_logger.debug(
                f"SkillsInjectionHook: Attached {len(generated_files)} files to dict response"
            )
            return response
        
        # Handle object response (OpenAI format)
        try:
            response._litellm_generated_files = generated_files
        except AttributeError:
            pass
        
        # Also add to model_extra if available (for serialization)
        if hasattr(response, "model_extra"):
            if response.model_extra is None:
                response.model_extra = {}
            response.model_extra["_litellm_generated_files"] = generated_files
        
        verbose_proxy_logger.debug(
            f"SkillsInjectionHook: Attached {len(generated_files)} files to response"
        )
        
        return response


# Global instance for registration
skills_injection_hook = SkillsInjectionHook()

import litellm

litellm.logging_callback_manager.add_litellm_callback(skills_injection_hook)
