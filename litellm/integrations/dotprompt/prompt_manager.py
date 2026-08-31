"""
Based on Google's GenAI Kit dotprompt implementation: https://google.github.io/dotprompt/reference/frontmatter/
"""

import re
from pathlib import Path
from typing import Any, Final

import yaml
from jinja2 import DictLoader, select_autoescape
from jinja2.sandbox import ImmutableSandboxedEnvironment


def strip_version_suffix(prompt_id: str) -> str | None:
    base, separator, version = prompt_id.rpartition(".v")
    if separator and base and version.isdigit():
        return base
    return None


class PromptTemplate:
    """Represents a single prompt template with metadata and content."""

    def __init__(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        template_id: str | None = None,
    ):
        self.content = content
        self.metadata = metadata or {}
        self.template_id = template_id

        # Extract common metadata fields
        restricted_keys: Final = ["model", "input", "output"]
        self.model = self.metadata.get("model")
        self.input_schema = self.metadata.get("input", {}).get("schema", {})
        self.output_format = self.metadata.get("output", {}).get("format")
        self.output_schema = self.metadata.get("output", {}).get("schema", {})
        self.optional_params = {}
        for key in self.metadata:
            if key not in restricted_keys:
                self.optional_params[key] = self.metadata[key]

    def __repr__(self):
        return f"PromptTemplate(id='{self.template_id}', model='{self.model}')"


class PromptManager:
    """
    Manager for loading and rendering .prompt files following the Dotprompt specification.

    Supports:
    - YAML frontmatter for metadata
    - Handlebars-style templating (using Jinja2)
    - Input/output schema validation
    - Model configuration
    """

    def __init__(
        self,
        prompt_id: str | None = None,
        prompt_directory: str | None = None,
        prompt_data: dict[str, dict[str, Any]] | None = None,
        prompt_file: str | None = None,
    ):
        self.prompt_directory = Path(prompt_directory) if prompt_directory else None
        self.prompts: dict[str, PromptTemplate] = {}
        self.prompt_file = prompt_file
        # Sandboxed env: templates can come from user input via /prompts/test,
        # so we must block access to unsafe Python attributes and mutation of
        # caller-supplied mutables.
        self.jinja_env = ImmutableSandboxedEnvironment(
            loader=DictLoader({}),
            autoescape=select_autoescape(["html", "xml"]),
            # Use Handlebars-style delimiters to match Dotprompt spec
            variable_start_string="{{",
            variable_end_string="}}",
            block_start_string="{%",
            block_end_string="%}",
            comment_start_string="{#",
            comment_end_string="#}",
        )

        # Load prompts from directory if provided
        if self.prompt_directory:
            self._load_prompts()

        if self.prompt_file:
            if not prompt_id:
                raise ValueError("prompt_id is required when prompt_file is provided")

            template: Final = self._load_prompt_file(self.prompt_file, prompt_id)
            self.prompts[prompt_id] = template

        # Load prompts from JSON data if provided
        if prompt_data:
            self._load_prompts_from_json(prompt_data, prompt_id)

    def _load_prompts(self) -> None:
        """Load all .prompt files from the prompt directory."""
        if not self.prompt_directory or not self.prompt_directory.exists():
            raise ValueError(f"Prompt directory does not exist: {self.prompt_directory}")

        prompt_files: Final = list(self.prompt_directory.glob("*.prompt"))

        for prompt_file in prompt_files:
            try:
                prompt_id = prompt_file.stem  # filename without extension
                template = self._load_prompt_file(prompt_file, prompt_id)
                self.prompts[prompt_id] = template
                # Optional: print(f"Loaded prompt: {prompt_id}")
            except Exception:
                # Optional: print(f"Error loading prompt file {prompt_file}")
                pass

    def _load_prompts_from_json(self, prompt_data: dict[str, dict[str, Any]], prompt_id: str | None = None) -> None:
        """Load prompts from JSON data structure.

        Expected format:
        {
            "prompt_id": {
                "content": "template content",
                "metadata": {"model": "gpt-4", "temperature": 0.7, ...}
            }
        }

        or

        {
            "content": "template content",
            "metadata": {"model": "gpt-4", "temperature": 0.7, ...}
        } + prompt_id

        A dict carrying a "content" key is a single flat template registered under
        prompt_id; anything else is treated as already keyed by template ID.
        """
        keyed_prompts: Final = {prompt_id: prompt_data} if prompt_id and "content" in prompt_data else prompt_data

        for template_id, prompt_info in keyed_prompts.items():
            try:
                content = prompt_info.get("content", "")
                metadata = prompt_info.get("metadata", {})

                template = PromptTemplate(
                    content=content,
                    metadata=metadata,
                    template_id=template_id,
                )
                self.prompts[template_id] = template
            except Exception:
                pass

    def _load_prompt_file(self, file_path: str | Path, prompt_id: str) -> PromptTemplate:
        """Load and parse a single .prompt file."""
        if isinstance(file_path, str):
            file_path = Path(file_path)

        content: Final = file_path.read_text(encoding="utf-8")

        # Split frontmatter and content
        frontmatter, template_content = self._parse_frontmatter(content)

        return PromptTemplate(
            content=template_content.strip(),
            metadata=frontmatter,
            template_id=prompt_id,
        )

    def _parse_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """Parse YAML frontmatter from prompt content."""
        # Match YAML frontmatter between --- delimiters
        frontmatter_pattern: Final = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match: Final = re.match(frontmatter_pattern, content, re.DOTALL)

        if match:
            frontmatter_yaml: Final = match.group(1)
            template_content = match.group(2)

            try:
                frontmatter = yaml.safe_load(frontmatter_yaml) or {}
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML frontmatter: {e}")
        else:
            # No frontmatter found, treat entire content as template
            frontmatter = {}
            template_content = content

        return frontmatter, template_content

    def render(
        self,
        prompt_id: str,
        prompt_variables: dict[str, Any] | None = None,
        version: int | None = None,
    ) -> str:
        """
        Render a prompt template with the given variables.

        Args:
            prompt_id: The ID of the prompt template to render
            prompt_variables: Variables to substitute in the template
            version: Optional version number. If provided, looks for {prompt_id}.v{version}

        Returns:
            The rendered prompt string

        Raises:
            KeyError: If prompt_id is not found
            ValueError: If template rendering fails
        """
        # Get the template (versioned or base)
        template: Final = self.get_prompt(prompt_id=prompt_id, version=version)

        if template is None:
            available_prompts: Final = list(self.prompts.keys())
            version_str: Final = f" (version {version})" if version else ""
            raise KeyError(f"Prompt '{prompt_id}'{version_str} not found. Available prompts: {available_prompts}")

        variables: Final = prompt_variables or {}

        # Validate input variables against schema if defined
        if template.input_schema:
            self._validate_input(variables, template.input_schema)

        try:
            # Create Jinja2 template and render
            jinja_template: Final = self.jinja_env.from_string(template.content)
            rendered: Final = jinja_template.render(**variables)
            return rendered
        except Exception as e:
            raise ValueError(f"Error rendering template '{prompt_id}': {e}")

    def _validate_input(self, variables: dict[str, Any], schema: dict[str, Any]) -> None:
        """Basic validation of input variables against schema."""
        for field_name, field_type in schema.items():
            if field_name in variables:
                value = variables[field_name]
                expected_type = self._get_python_type(field_type)

                if not isinstance(value, expected_type):
                    raise ValueError(
                        f"Invalid type for field '{field_name}': "
                        f"expected {getattr(expected_type, '__name__', str(expected_type))}, got {type(value).__name__}"
                    )

    def _get_python_type(self, schema_type: str) -> type | tuple:
        """Convert schema type string to Python type."""
        type_mapping: Final[dict[str, type | tuple]] = {
            "string": str,
            "str": str,
            "number": (int, float),
            "integer": int,
            "int": int,
            "float": float,
            "boolean": bool,
            "bool": bool,
            "array": list,
            "list": list,
            "object": dict,
            "dict": dict,
        }

        return type_mapping.get(schema_type.lower(), str)

    def get_prompt(self, prompt_id: str, version: int | None = None) -> PromptTemplate | None:
        """
        Get a prompt template by ID and optional version.

        Args:
            prompt_id: The base prompt ID
            version: Optional version number. If provided, looks for {prompt_id}.v{version}

        Returns:
            The prompt template if found, None otherwise
        """
        if version is not None:
            # Try versioned prompt first: prompt_id.v{version}
            versioned_id: Final = f"{prompt_id}.v{version}"
            if versioned_id in self.prompts:
                return self.prompts[versioned_id]

        direct_match: Final = self.prompts.get(prompt_id)
        if direct_match is not None:
            return direct_match

        base_prompt_id: Final = strip_version_suffix(prompt_id)
        return self.prompts.get(base_prompt_id) if base_prompt_id else None

    def list_prompts(self) -> list[str]:
        """Get a list of all available prompt IDs."""
        return list(self.prompts.keys())

    def get_prompt_metadata(self, prompt_id: str) -> dict[str, Any] | None:
        """Get metadata for a specific prompt."""
        template: Final = self.prompts.get(prompt_id)
        return template.metadata if template else None

    def reload_prompts(self) -> None:
        """Reload all prompts from the directory (if directory was provided)."""
        self.prompts.clear()
        if self.prompt_directory:
            self._load_prompts()

    def add_prompt(self, prompt_id: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a prompt template programmatically."""
        template: Final = PromptTemplate(content=content, metadata=metadata or {}, template_id=prompt_id)
        self.prompts[prompt_id] = template

    def prompt_file_to_json(self, file_path: str | Path) -> dict[str, Any]:
        """Convert a .prompt file to JSON format.

        Args:
            file_path: Path to the .prompt file

        Returns:
            Dictionary with 'content' and 'metadata' keys
        """
        file_path = Path(file_path)
        content: Final = file_path.read_text(encoding="utf-8")

        # Parse frontmatter and content
        frontmatter, template_content = self._parse_frontmatter(content)

        return {"content": template_content.strip(), "metadata": frontmatter}

    def json_to_prompt_file(self, prompt_data: dict[str, Any]) -> str:
        """Convert JSON prompt data to .prompt file format.

        Args:
            prompt_data: Dictionary with 'content' and 'metadata' keys

        Returns:
            String content in .prompt file format
        """
        content: Final = prompt_data.get("content", "")
        metadata: Final = prompt_data.get("metadata", {})

        if not metadata:
            # No metadata, return just the content
            return content

        # Convert metadata to YAML frontmatter
        import yaml

        frontmatter_yaml: Final = yaml.dump(metadata, default_flow_style=False)

        return f"---\n{frontmatter_yaml}---\n{content}"

    def get_all_prompts_as_json(self) -> dict[str, dict[str, Any]]:
        """Get all loaded prompts in JSON format.

        Returns:
            Dictionary mapping prompt_id to prompt data
        """
        result: Final = {}
        for prompt_id, template in self.prompts.items():
            result[prompt_id] = {
                "content": template.content,
                "metadata": template.metadata,
            }
        return result

    def load_prompts_from_json_data(self, prompt_data: dict[str, dict[str, Any]]) -> None:
        """Load additional prompts from JSON data (merges with existing prompts)."""
        self._load_prompts_from_json(prompt_data)
