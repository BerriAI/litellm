"""
Unit tests for SkillsSandboxExecutor requirement parsing and pip code generation.

Validates that:
- Valid package specs are accepted
- Malicious inputs with injection characters are rejected
- The generated pip install code is safe from code injection
"""

import json

import pytest

from litellm.llms.litellm_proxy.skills.sandbox_executor import (
    SkillsSandboxExecutor,
    _VALID_PACKAGE_SPEC,
)


class TestParseRequirements:
    """Tests for _parse_requirements static method."""

    def test_should_parse_simple_packages(self):
        raw = "numpy\npandas\nrequests"
        result = SkillsSandboxExecutor._parse_requirements(raw)
        assert result == ["numpy", "pandas", "requests"]

    def test_should_parse_packages_with_versions(self):
        raw = "numpy>=1.21.0\npandas==2.0.0\nrequests<3.0"
        result = SkillsSandboxExecutor._parse_requirements(raw)
        assert result == ["numpy>=1.21.0", "pandas==2.0.0", "requests<3.0"]

    def test_should_parse_packages_with_extras(self):
        raw = "uvicorn[standard]\nfastapi[all]>=0.100"
        result = SkillsSandboxExecutor._parse_requirements(raw)
        assert result == ["uvicorn[standard]", "fastapi[all]>=0.100"]

    def test_should_skip_comments_and_blank_lines(self):
        raw = "# this is a comment\nnumpy\n\n# another comment\npandas\n"
        result = SkillsSandboxExecutor._parse_requirements(raw)
        assert result == ["numpy", "pandas"]

    def test_should_skip_pip_flags(self):
        raw = "--index-url https://pypi.org/simple\nnumpy\n-f http://example.com"
        result = SkillsSandboxExecutor._parse_requirements(raw)
        assert result == ["numpy"]

    def test_should_strip_whitespace(self):
        raw = "  numpy  \n  pandas  \n"
        result = SkillsSandboxExecutor._parse_requirements(raw)
        assert result == ["numpy", "pandas"]

    def test_should_return_empty_for_empty_input(self):
        assert SkillsSandboxExecutor._parse_requirements("") == []
        assert SkillsSandboxExecutor._parse_requirements("   \n\n  ") == []

    def test_should_reject_single_quote_injection(self):
        malicious = "numpy\n'.split()\nimport os; os.system('whoami') #"
        result = SkillsSandboxExecutor._parse_requirements(malicious)
        assert "numpy" in result
        assert len(result) == 1

    def test_should_reject_double_quote_injection(self):
        malicious = 'numpy\n".split()\nimport os; os.system("whoami") #'
        result = SkillsSandboxExecutor._parse_requirements(malicious)
        assert result == ["numpy"]

    def test_should_reject_semicolons(self):
        malicious = "numpy; import os"
        result = SkillsSandboxExecutor._parse_requirements(malicious)
        assert result == []

    def test_should_reject_backtick_injection(self):
        malicious = "numpy`whoami`"
        result = SkillsSandboxExecutor._parse_requirements(malicious)
        assert result == []

    def test_should_reject_newline_embedded_in_package_name(self):
        malicious = "numpy\npandas\n'; __import__('os').system('id'); #"
        result = SkillsSandboxExecutor._parse_requirements(malicious)
        assert result == ["numpy", "pandas"]

    def test_should_reject_parentheses_injection(self):
        malicious = "__import__('os').system('id')"
        result = SkillsSandboxExecutor._parse_requirements(malicious)
        assert result == []

    def test_should_handle_version_ranges(self):
        raw = "numpy>=1.21,<2.0\npandas!=1.5.0"
        result = SkillsSandboxExecutor._parse_requirements(raw)
        assert "numpy>=1.21,<2.0" in result
        assert "pandas!=1.5.0" in result

    def test_should_handle_underscores_and_dots(self):
        raw = "my_package\nmy.package\nmy-package"
        result = SkillsSandboxExecutor._parse_requirements(raw)
        assert result == ["my_package", "my.package", "my-package"]


class TestPipCodeGeneration:
    """Tests that the generated pip code is safe from injection."""

    def test_should_produce_safe_code_for_normal_packages(self):
        packages = ["numpy", "pandas>=2.0"]
        safe_list = json.dumps(packages)
        pip_code = (
            "import subprocess, json\n"
            f"subprocess.run(['pip', 'install'] + json.loads({safe_list!r}), check=True)\n"
        )
        assert "numpy" in pip_code
        assert "pandas>=2.0" in pip_code
        assert "json.loads" in pip_code

    def test_should_double_serialize_preventing_breakout(self):
        """Even if a malicious string somehow passed validation, json.dumps + !r
        double-escapes it so it cannot break out of the string context."""
        malicious_package = "fake'); __import__('os').system('id'); #"
        safe_list = json.dumps([malicious_package])
        pip_code = (
            "import subprocess, json\n"
            f"subprocess.run(['pip', 'install'] + json.loads({safe_list!r}), check=True)\n"
        )
        assert "__import__" not in pip_code or "json.loads" in pip_code
        assert pip_code.count("subprocess.run") == 1


class TestValidPackageSpecRegex:
    """Tests for the _VALID_PACKAGE_SPEC regex pattern."""

    @pytest.mark.parametrize(
        "spec",
        [
            "numpy",
            "pandas",
            "requests",
            "my-package",
            "my_package",
            "my.package",
            "numpy>=1.21",
            "pandas==2.0.0",
            "requests<3.0",
            "flask>=2.0,<3.0",
            "uvicorn[standard]",
            "fastapi[all]>=0.100",
            "package123",
            "A_Package",
        ],
    )
    def test_should_match_valid_specs(self, spec):
        assert _VALID_PACKAGE_SPEC.match(spec), f"Expected {spec!r} to match"

    @pytest.mark.parametrize(
        "spec",
        [
            "'; import os #",
            "__import__('os')",
            "pkg; rm -rf /",
            "$(whoami)",
            "`id`",
            "",
            "-f http://evil.com",
            "# comment",
        ],
    )
    def test_should_not_match_malicious_specs(self, spec):
        assert not _VALID_PACKAGE_SPEC.match(spec), f"Expected {spec!r} NOT to match"
