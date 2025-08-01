"""
Based on Google's GenAI Kit dotprompt implementation: https://google.github.io/dotprompt/reference/frontmatter/
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from jinja2 import DictLoader, Environment, select_autoescape


class PromptTemplate:
    """Represents a single prompt template with metadata and content."""

    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        template_id: Optional[str] = None,
    ):
        self.content = content
        self.metadata = metadata or {}
        self.template_id = template_id

        # Extract common metadata fields
        restricted_keys = ["model", "input", "output"]
        self.model = self.metadata.get("model")
        self.input_schema = self.metadata.get("input", {}).get("schema", {})
        self.output_format = self.metadata.get("output", {}).get("format")
        self.output_schema = self.metadata.get("output", {}).get("schema", {})
        self.optional_params = {}
        for key in self.metadata.keys():
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

    def __init__(self, prompt_directory: str):
        self.prompt_directory = Path(prompt_directory)
        self.prompts: Dict[str, PromptTemplate] = {}
        self.jinja_env = Environment(
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

        # Load all prompts in the directory
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load all .prompt files from the prompt directory."""
        if not self.prompt_directory.exists():
            raise ValueError(
                f"Prompt directory does not exist: {self.prompt_directory}"
            )

        prompt_files = list(self.prompt_directory.glob("*.prompt"))

        for prompt_file in prompt_files:
            try:
                prompt_id = prompt_file.stem  # filename without extension
                template = self._load_prompt_file(prompt_file, prompt_id)
                self.prompts[prompt_id] = template
                # Optional: print(f"Loaded prompt: {prompt_id}")
            except Exception:
                # Optional: print(f"Error loading prompt file {prompt_file}")
                pass

    def _load_prompt_file(self, file_path: Path, prompt_id: str) -> PromptTemplate:
        """Load and parse a single .prompt file."""
        content = file_path.read_text(encoding="utf-8")

        # Split frontmatter and content
        frontmatter, template_content = self._parse_frontmatter(content)

        return PromptTemplate(
            content=template_content.strip(),
            metadata=frontmatter,
            template_id=prompt_id,
        )

    def _parse_frontmatter(self, content: str) -> Tuple[Dict[str, Any], str]:
        """Parse YAML frontmatter from prompt content."""
        # Match YAML frontmatter between --- delimiters
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(frontmatter_pattern, content, re.DOTALL)

        if match:
            frontmatter_yaml = match.group(1)
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
        self, prompt_id: str, prompt_variables: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Render a prompt template with the given variables.

        Args:
            prompt_id: The ID of the prompt template to render
            prompt_variables: Variables to substitute in the template

        Returns:
            The rendered prompt string

        Raises:
            KeyError: If prompt_id is not found
            ValueError: If template rendering fails
        """
        if prompt_id not in self.prompts:
            available_prompts = list(self.prompts.keys())
            raise KeyError(
                f"Prompt '{prompt_id}' not found. Available prompts: {available_prompts}"
            )

        template = self.prompts[prompt_id]
        variables = prompt_variables or {}

        # Validate input variables against schema if defined
        if template.input_schema:
            self._validate_input(variables, template.input_schema)

        try:
            # Create Jinja2 template and render
            jinja_template = self.jinja_env.from_string(template.content)
            rendered = jinja_template.render(**variables)
            return rendered
        except Exception as e:
            raise ValueError(f"Error rendering template '{prompt_id}': {e}")

    def _validate_input(
        self, variables: Dict[str, Any], schema: Dict[str, Any]
    ) -> None:
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

    def _get_python_type(self, schema_type: str) -> Union[type, tuple]:
        """Convert schema type string to Python type."""
        type_mapping: Dict[str, Union[type, tuple]] = {
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

        return type_mapping.get(schema_type.lower(), str)  # type: ignore

    def get_prompt(self, prompt_id: str) -> Optional[PromptTemplate]:
        """Get a prompt template by ID."""
        return self.prompts.get(prompt_id)

    def list_prompts(self) -> List[str]:
        """Get a list of all available prompt IDs."""
        return list(self.prompts.keys())

    def get_prompt_metadata(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific prompt."""
        template = self.prompts.get(prompt_id)
        return template.metadata if template else None

    def reload_prompts(self) -> None:
        """Reload all prompts from the directory."""
        self.prompts.clear()
        self._load_prompts()

    def add_prompt(
        self, prompt_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a prompt template programmatically."""
        template = PromptTemplate(
            content=content, metadata=metadata or {}, template_id=prompt_id
        )
        self.prompts[prompt_id] = template
