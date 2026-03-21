"""
Tests for Skills validation utilities (YAML frontmatter parsing, file validation).
"""

import pytest

from litellm.proxy.skills_endpoints.validation import (
    SkillFrontmatter,
    parse_skill_md,
    validate_skill_files,
)


class TestParseSKillMd:
    """Tests for parse_skill_md function."""

    def test_parse_valid_frontmatter(self):
        """Test parsing valid YAML frontmatter."""
        content = """---
name: Test Skill
description: A test skill for testing
---

# Instructions

Use this skill to do testing.
"""
        frontmatter, body = parse_skill_md(content)

        assert frontmatter is not None
        assert frontmatter.name == "Test Skill"
        assert frontmatter.description == "A test skill for testing"
        assert "# Instructions" in body
        assert "Use this skill to do testing." in body

    def test_parse_frontmatter_name_only(self):
        """Test parsing frontmatter with only required name field."""
        content = """---
name: Minimal Skill
---

Instructions here.
"""
        frontmatter, body = parse_skill_md(content)

        assert frontmatter is not None
        assert frontmatter.name == "Minimal Skill"
        assert frontmatter.description is None
        assert "Instructions here." in body

    def test_parse_no_frontmatter(self):
        """Test parsing content without frontmatter."""
        content = """# Just Markdown

No YAML frontmatter here.
"""
        frontmatter, body = parse_skill_md(content)

        assert frontmatter is None
        assert "# Just Markdown" in body

    def test_parse_empty_frontmatter(self):
        """Test parsing empty frontmatter block."""
        content = """---
---

Content after empty frontmatter.
"""
        frontmatter, body = parse_skill_md(content)

        # Empty frontmatter should fail validation (no name)
        assert frontmatter is None

    def test_parse_frontmatter_missing_name(self):
        """Test parsing frontmatter without required name field."""
        content = """---
description: Description but no name
---

Body content.
"""
        frontmatter, body = parse_skill_md(content)

        # Should fail validation since name is required
        assert frontmatter is None

    def test_parse_frontmatter_name_too_long(self):
        """Test that name exceeding 64 characters fails validation."""
        long_name = "A" * 65  # 65 chars, exceeds limit
        content = f"""---
name: {long_name}
---

Body content.
"""
        frontmatter, body = parse_skill_md(content)

        # Should fail validation due to name length
        assert frontmatter is None

    def test_parse_frontmatter_name_at_limit(self):
        """Test that name exactly 64 characters is valid."""
        exact_name = "A" * 64  # exactly 64 chars
        content = f"""---
name: {exact_name}
---

Body content.
"""
        frontmatter, body = parse_skill_md(content)

        assert frontmatter is not None
        assert frontmatter.name == exact_name

    def test_parse_frontmatter_description_too_long(self):
        """Test that description exceeding 1024 characters fails validation."""
        long_desc = "B" * 1025  # 1025 chars, exceeds limit
        content = f"""---
name: Valid Name
description: {long_desc}
---

Body content.
"""
        frontmatter, body = parse_skill_md(content)

        # Should fail validation due to description length
        assert frontmatter is None


class TestSkillFrontmatter:
    """Tests for SkillFrontmatter Pydantic model."""

    def test_valid_frontmatter(self):
        """Test creating valid frontmatter."""
        fm = SkillFrontmatter(name="Test", description="A test skill")
        assert fm.name == "Test"
        assert fm.description == "A test skill"

    def test_frontmatter_name_only(self):
        """Test creating frontmatter with name only."""
        fm = SkillFrontmatter(name="NameOnly")
        assert fm.name == "NameOnly"
        assert fm.description is None

    def test_frontmatter_name_required(self):
        """Test that name is required."""
        with pytest.raises(Exception):
            SkillFrontmatter(description="No name")

    def test_frontmatter_name_max_length(self):
        """Test name max length constraint."""
        # Should work at 64
        fm = SkillFrontmatter(name="A" * 64)
        assert len(fm.name) == 64

        # Should fail at 65
        with pytest.raises(Exception):
            SkillFrontmatter(name="A" * 65)

    def test_frontmatter_description_max_length(self):
        """Test description max length constraint."""
        # Should work at 1024
        fm = SkillFrontmatter(name="Test", description="B" * 1024)
        assert len(fm.description) == 1024

        # Should fail at 1025
        with pytest.raises(Exception):
            SkillFrontmatter(name="Test", description="B" * 1025)


class TestValidateSkillFiles:
    """Tests for validate_skill_files function."""

    def test_valid_skill_files(self):
        """Test validating files with valid SKILL.md."""
        skill_md_content = b"""---
name: Test Skill
description: A test
---

Instructions here.
"""
        files = [("SKILL.md", skill_md_content)]

        zip_content, frontmatter, body, errors = validate_skill_files(files)

        assert errors == []
        assert zip_content is not None
        assert frontmatter is not None
        assert frontmatter.name == "Test Skill"
        assert "Instructions here." in body

    def test_missing_skill_md(self):
        """Test validation fails without SKILL.md."""
        files = [("README.md", b"# Readme\nSome content")]

        zip_content, frontmatter, body, errors = validate_skill_files(files)

        assert "SKILL.md is required" in errors[0]
        assert zip_content is None

    def test_file_size_limit(self):
        """Test validation fails for files exceeding 8MB."""
        # Create content > 8MB
        large_content = b"x" * (8 * 1024 * 1024 + 1)  # 8MB + 1 byte
        files = [
            ("SKILL.md", b"---\nname: Test\n---\nContent"),
            ("large_file.bin", large_content),
        ]

        zip_content, frontmatter, body, errors = validate_skill_files(files)

        assert any("8MB" in err or "size" in err.lower() for err in errors)

    def test_nested_skill_md(self):
        """Test that SKILL.md in nested folder is found."""
        skill_md_content = b"""---
name: Nested Skill
---

Nested instructions.
"""
        files = [("subfolder/SKILL.md", skill_md_content)]

        zip_content, frontmatter, body, errors = validate_skill_files(files)

        assert errors == []
        assert frontmatter is not None
        assert frontmatter.name == "Nested Skill"

    def test_multiple_files_creates_zip(self):
        """Test that multiple files are packed into a valid ZIP."""
        import zipfile
        from io import BytesIO

        skill_md = b"""---
name: Multi File Skill
---

Use the helper module.
"""
        helper_py = b"def helper(): return 42"

        files = [
            ("SKILL.md", skill_md),
            ("helper.py", helper_py),
        ]

        zip_content, frontmatter, body, errors = validate_skill_files(files)

        assert errors == []
        assert zip_content is not None

        # Verify it's a valid ZIP
        zip_buffer = BytesIO(zip_content)
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            names = zf.namelist()
            assert any("SKILL.md" in n for n in names)
            assert any("helper.py" in n for n in names)

    def test_invalid_frontmatter_in_skill_md(self):
        """Test validation fails for SKILL.md with invalid frontmatter."""
        # Missing required name
        skill_md = b"""---
description: No name field
---

Body content.
"""
        files = [("SKILL.md", skill_md)]

        zip_content, frontmatter, body, errors = validate_skill_files(files)

        assert any("frontmatter" in err.lower() or "name" in err.lower() for err in errors)
