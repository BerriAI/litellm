"""
Validation utilities for Gateway Skills API.

Handles YAML frontmatter parsing and validation from SKILL.md files.
"""

import io
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from litellm._logging import verbose_logger

# Maximum file size for skill uploads (8MB)
MAX_SKILL_FILE_SIZE = 8 * 1024 * 1024

# YAML frontmatter regex pattern
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


class SkillFrontmatter(BaseModel):
    """
    Pydantic model for SKILL.md YAML frontmatter.

    Validates the frontmatter according to the spec:
    - name: required, max 64 characters
    - description: optional, max 1024 characters
    """

    name: str = Field(..., max_length=64, description="Skill name (max 64 chars)")
    description: Optional[str] = Field(
        None, max_length=1024, description="Skill description (max 1024 chars)"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate skill name is not empty and within limits."""
        if not v or not v.strip():
            raise ValueError("Skill name cannot be empty")
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: Optional[str]) -> Optional[str]:
        """Validate description is within limits."""
        if v is not None:
            return v.strip() if v.strip() else None
        return v


def parse_yaml_frontmatter(content: str) -> Optional[Dict[str, Any]]:
    """
    Parse YAML frontmatter from markdown content.

    Frontmatter is expected in the format:
    ---
    name: My Skill
    description: Does something cool
    ---

    Args:
        content: The markdown content with potential frontmatter

    Returns:
        Dict of frontmatter values, or None if no frontmatter found
    """
    try:
        import yaml
    except ImportError:
        verbose_logger.warning(
            "PyYAML not installed, cannot parse SKILL.md frontmatter"
        )
        return None

    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return None

    yaml_content = match.group(1)
    try:
        return yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        verbose_logger.warning(f"Failed to parse YAML frontmatter: {e}")
        return None


def parse_skill_md(content: str) -> Tuple[Optional[SkillFrontmatter], str]:
    """
    Parse SKILL.md content to extract frontmatter and body.

    Args:
        content: The full SKILL.md content

    Returns:
        Tuple of (SkillFrontmatter or None, body content without frontmatter)
    """
    # Try to parse frontmatter
    frontmatter_data = parse_yaml_frontmatter(content)

    # Extract body (content after frontmatter)
    body = content
    match = FRONTMATTER_PATTERN.match(content)
    if match:
        body = content[match.end() :].strip()

    # Validate frontmatter if present
    if frontmatter_data:
        try:
            frontmatter = SkillFrontmatter(**frontmatter_data)
            return frontmatter, body
        except Exception as e:
            verbose_logger.warning(f"Invalid SKILL.md frontmatter: {e}")
            return None, body

    return None, body


def validate_skill_files(
    files: List[Tuple[str, bytes]],
) -> Tuple[Optional[bytes], Optional[SkillFrontmatter], str, List[str]]:
    """
    Validate uploaded skill files and create a ZIP archive.

    Validates:
    - SKILL.md is present
    - Total size is under 8MB limit
    - YAML frontmatter is valid (if present)

    Args:
        files: List of (filename, content) tuples

    Returns:
        Tuple of (zip_content, frontmatter, body_content, error_messages)
        If errors, zip_content and frontmatter will be None
    """
    errors: List[str] = []
    skill_md_content: Optional[str] = None
    total_size = 0

    # Check for SKILL.md and calculate total size
    for filename, content in files:
        total_size += len(content)
        if filename == "SKILL.md" or filename.endswith("/SKILL.md"):
            try:
                skill_md_content = content.decode("utf-8")
            except UnicodeDecodeError:
                errors.append("SKILL.md must be valid UTF-8 text")

    # Validate SKILL.md presence
    if skill_md_content is None:
        errors.append("SKILL.md is required at the root of the skill files")

    # Validate total size
    if total_size > MAX_SKILL_FILE_SIZE:
        errors.append(
            f"Total file size ({total_size / 1024 / 1024:.2f}MB) exceeds "
            f"limit ({MAX_SKILL_FILE_SIZE / 1024 / 1024}MB)"
        )

    if errors:
        return None, None, "", errors

    # Parse SKILL.md frontmatter
    assert skill_md_content is not None  # We checked above
    frontmatter, body = parse_skill_md(skill_md_content)

    # Validate frontmatter is present and valid
    if frontmatter is None:
        errors.append(
            "SKILL.md must have valid YAML frontmatter with at least a 'name' field"
        )
        return None, None, body, errors

    # Create ZIP archive
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content in files:
            zf.writestr(filename, content)

    return zip_buffer.getvalue(), frontmatter, body, []


def extract_skill_name_from_zip(zip_content: bytes) -> Optional[str]:
    """
    Extract skill name from a ZIP file containing SKILL.md.

    Args:
        zip_content: The ZIP file content as bytes

    Returns:
        The skill name from frontmatter, or None if not found
    """
    try:
        zip_buffer = io.BytesIO(zip_content)
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            for name in zf.namelist():
                if name.endswith("SKILL.md"):
                    content = zf.read(name).decode("utf-8")
                    frontmatter, _ = parse_skill_md(content)
                    if frontmatter:
                        return frontmatter.name
    except Exception as e:
        verbose_logger.warning(f"Failed to extract skill name from ZIP: {e}")

    return None
