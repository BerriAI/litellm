"""
Prompt Injection Handler for LiteLLM Skills

Handles extraction of skill content (SKILL.md) from stored ZIP files
and injection into the system prompt for non-Anthropic models.
"""

import zipfile
from io import BytesIO
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.proxy._types import LiteLLM_SkillsTable


class SkillPromptInjectionHandler:
    """
    Handles skill content extraction and system prompt injection.
    
    Responsibilities:
    - Extract SKILL.md content from skill ZIP files
    - Extract ALL files from ZIP for code execution
    - Inject skill content into system message
    - Create execute_code tool definition
    """

    def extract_skill_content(self, skill: LiteLLM_SkillsTable) -> Optional[str]:
        """
        Extract skill content from the stored zip file.
        
        Looks for SKILL.md or README.md in the zip and returns its content.
        This content describes the skill's capabilities and instructions.
        
        Args:
            skill: The skill from LiteLLM database
            
        Returns:
            The skill content as a string, or None if not available
        """
        if not skill.file_content:
            return skill.instructions
        
        try:
            zip_buffer = BytesIO(skill.file_content)
            with zipfile.ZipFile(zip_buffer, "r") as zf:
                # Look for SKILL.md first
                for name in zf.namelist():
                    if name.endswith("SKILL.md"):
                        content = zf.read(name).decode("utf-8")
                        if content:
                            return f"## Skill: {skill.display_title or skill.skill_id}\n\n{content}"
                
                # Fall back to README.md
                for name in zf.namelist():
                    if name.endswith("README.md"):
                        content = zf.read(name).decode("utf-8")
                        if content:
                            return f"## Skill: {skill.display_title or skill.skill_id}\n\n{content}"
                
                # Fall back to any .md file
                for name in zf.namelist():
                    if name.endswith(".md"):
                        content = zf.read(name).decode("utf-8")
                        if content:
                            return f"## Skill: {skill.display_title or skill.skill_id}\n\n{content}"
        except Exception as e:
            verbose_logger.warning(
                f"SkillPromptInjectionHandler: Error extracting content from skill {skill.skill_id}: {e}"
            )
        
        return skill.instructions

    def extract_all_files(self, skill: LiteLLM_SkillsTable) -> Dict[str, bytes]:
        """
        Extract ALL files from skill ZIP for code execution.
        
        Returns a dict mapping file paths to their binary content.
        The paths have the skill folder prefix removed (e.g., "slack-gif-creator/core/..." -> "core/...").
        
        Args:
            skill: The skill from LiteLLM database
            
        Returns:
            Dict mapping file paths to binary content
        """
        files: Dict[str, bytes] = {}
        
        if not skill.file_content:
            return files
        
        try:
            zip_buffer = BytesIO(skill.file_content)
            with zipfile.ZipFile(zip_buffer, "r") as zf:
                for name in zf.namelist():
                    # Skip directories
                    if name.endswith("/"):
                        continue
                    
                    # Remove skill folder prefix (first path component)
                    parts = name.split("/")
                    if len(parts) > 1:
                        clean_path = "/".join(parts[1:])
                    else:
                        clean_path = name
                    
                    if clean_path:
                        files[clean_path] = zf.read(name)
        except Exception as e:
            verbose_logger.warning(
                f"SkillPromptInjectionHandler: Error extracting files from skill {skill.skill_id}: {e}"
            )
        
        return files

    def inject_skill_content_to_messages(
        self, data: dict, skill_contents: List[str], use_anthropic_format: bool = False
    ) -> dict:
        """
        Inject skill content into the system prompt.
        
        For Anthropic messages API (use_anthropic_format=True):
        - Injects into top-level 'system' parameter (not in messages array)
        
        For OpenAI-style APIs (use_anthropic_format=False):
        - Injects into messages array with role="system"
        
        Args:
            data: The request data dict
            skill_contents: List of skill content strings to inject
            use_anthropic_format: If True, use top-level 'system' param for Anthropic
            
        Returns:
            Modified data dict with skill content in system prompt
        """
        if not skill_contents:
            return data
        
        # Build the skill injection text
        skill_section = "\n\n---\n\n# Available Skills\n\n" + "\n\n---\n\n".join(skill_contents)
        
        if use_anthropic_format:
            # Anthropic messages API: use top-level 'system' parameter
            current_system = data.get("system", "")
            if current_system:
                data["system"] = current_system + skill_section
            else:
                data["system"] = skill_section.strip()
            return data
        
        # OpenAI-style: inject into messages array
        messages = data.get("messages", [])
        if not messages:
            return data
        
        # Find or create system message
        system_msg_idx = None
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "system":
                system_msg_idx = i
                break
        
        if system_msg_idx is not None:
            # Append to existing system message
            current_content = messages[system_msg_idx].get("content", "")
            messages[system_msg_idx]["content"] = current_content + skill_section
        else:
            # Create new system message at the beginning
            messages.insert(0, {"role": "system", "content": skill_section.strip()})
        
        data["messages"] = messages
        return data

    def create_execute_code_tool(self, skill_modules: List[str]) -> Dict[str, Any]:
        """
        Create the execute_code tool definition.
        
        This tool allows the model to execute Python code with access
        to the skill's modules (e.g., 'from core.gif_builder import GIFBuilder').
        
        Args:
            skill_modules: List of available module paths (e.g., ["core/gif_builder.py"])
            
        Returns:
            OpenAI-style tool definition
        """
        # Format module list for description
        module_examples = []
        for mod in skill_modules[:5]:  # Limit to 5 examples
            if mod.endswith(".py"):
                # Convert path to import: "core/gif_builder.py" -> "from core.gif_builder import ..."
                import_path = mod.replace("/", ".").replace(".py", "")
                module_examples.append(f"from {import_path} import ...")
        
        module_hint = ""
        if module_examples:
            module_hint = f" Available modules: {', '.join(module_examples)}"
        
        return {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": f"Execute Python code in a sandboxed environment. Generated files will be returned.{module_hint}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute. You can import skill modules and use standard libraries."
                        }
                    },
                    "required": ["code"]
                }
            }
        }

    def convert_skill_to_tool(self, skill: LiteLLM_SkillsTable) -> Dict[str, Any]:
        """
        Convert a LiteLLM skill to an OpenAI-style tool.

        The skill's instructions are used as the function description,
        allowing the model to understand when and how to use the skill.

        Args:
            skill: The skill from LiteLLM database

        Returns:
            OpenAI-style tool definition
        """
        # Create a function name from skill_id (sanitize for function naming)
        func_name = skill.skill_id.replace("-", "_").replace(" ", "_")

        # Use instructions as description, fall back to description or title
        description = (
            skill.instructions
            or skill.description
            or skill.display_title
            or f"Skill: {skill.skill_id}"
        )

        # Truncate description if too long (OpenAI has limits)
        max_desc_length = 1024
        if len(description) > max_desc_length:
            description = description[: max_desc_length - 3] + "..."

        tool: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

        # If skill has metadata with parameter definitions, use them
        if skill.metadata and isinstance(skill.metadata, dict):
            params = skill.metadata.get("parameters")
            if params and isinstance(params, dict):
                tool["function"]["parameters"] = params

        return tool

    def convert_skill_to_anthropic_tool(self, skill: LiteLLM_SkillsTable) -> Dict[str, Any]:
        """
        Convert a LiteLLM skill to an Anthropic-style tool (messages API format).

        Args:
            skill: The skill from LiteLLM database

        Returns:
            Anthropic-style tool definition with name, description, input_schema
        """
        func_name = skill.skill_id.replace("-", "_").replace(" ", "_")

        description = (
            skill.instructions
            or skill.description
            or skill.display_title
            or f"Skill: {skill.skill_id}"
        )

        max_desc_length = 1024
        if len(description) > max_desc_length:
            description = description[: max_desc_length - 3] + "..."

        input_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        if skill.metadata and isinstance(skill.metadata, dict):
            params = skill.metadata.get("parameters")
            if params and isinstance(params, dict):
                input_schema = params

        return {
            "name": func_name,
            "description": description,
            "input_schema": input_schema,
        }

