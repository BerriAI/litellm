"""
Tests for Skills API operations across providers
"""

import os
import sys
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.types.llms.anthropic_skills import (
    DeleteSkillResponse,
    ListSkillsResponse,
    Skill,
)


@contextmanager
def create_skill_zip(skill_name: str):
    """
    Helper context manager to create a zip file for a skill.
    
    Args:
        skill_name: Name of the skill directory in test_skills_data/
        
    Yields:
        File handle to the zip file
        
    The zip file is automatically cleaned up after use.
    """
    test_dir = Path(__file__).parent / "test_skills_data"
    skill_dir = test_dir / skill_name
    
    # Create a zip file containing the skill directory
    zip_path = test_dir / f"{skill_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(skill_dir, arcname=skill_name)
        zip_file.write(skill_dir / "SKILL.md", arcname=f"{skill_name}/SKILL.md")
    
    try:
        with open(zip_path, "rb") as f:
            yield f
    finally:
        # Clean up zip file
        if zip_path.exists():
            zip_path.unlink()


class BaseSkillsAPITest(ABC):
    """
    Base test class for Skills API operations.
    Tests create, list, get, and delete operations.
    """

    @abstractmethod
    def get_custom_llm_provider(self) -> str:
        """Return the provider name (e.g., 'anthropic')"""
        pass

    @abstractmethod
    def get_api_key(self) -> Optional[str]:
        """Return the API key for the provider"""
        pass

    @abstractmethod
    def get_api_base(self) -> Optional[str]:
        """Return the API base URL for the provider"""
        pass

    def test_create_skill(self):
        """
        Test creating a skill.
        
        Note: This test creates a skill but does not clean it up,
        as we want to verify it was created successfully.
        The test_delete_skill test will handle cleanup.
        """
        import time
        
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        litellm.set_verbose = True
        litellm._turn_on_debug()

        # Use helper to create skill zip
        skill_name = "test-skill-litellm"
        
        # Use unique title to avoid conflicts with previous test runs
        unique_title = f"Test Skill {int(time.time())}"
        
        # Upload the skill with the zip file
        with create_skill_zip(skill_name) as zip_file:
            response = litellm.create_skill(
                display_title=unique_title,
                files=[zip_file],
                custom_llm_provider=custom_llm_provider,
                api_key=api_key,
                api_base=api_base,
            )

        assert response is not None
        assert isinstance(response, Skill)
        assert response.id is not None
        print(f"Created skill: {response}")
        print(f"Skill ID: {response.id}")

    def test_list_skills(self):
        """
        Test listing skills.
        """
        import os
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        # Enable debug logging
        os.environ["LITELLM_LOG"] = "DEBUG"
        litellm.set_verbose = True

        print(f"\n=== Testing list_skills ===")
        print("API Key: [REDACTED]")
        print(f"API Base: {api_base}")
        
        response = litellm.list_skills(
            limit=10,
            custom_llm_provider=custom_llm_provider,
            api_key=api_key,
            api_base=api_base,
        )

        assert response is not None
        assert isinstance(response, ListSkillsResponse)
        assert hasattr(response, "data")
        print(f"Listed skills: {response}")

    def test_get_skill(self):
        """
        Test getting a specific skill by ID.
        """
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        litellm.set_verbose = True

        # First list existing skills to see if any exist
        list_response = litellm.list_skills(
            limit=1,
            custom_llm_provider=custom_llm_provider,
            api_key=api_key,
            api_base=api_base,
        )
        
        # Type assertion for linter
        assert isinstance(list_response, ListSkillsResponse)
        print(f"List response: {list_response}")
        
        # If there are existing skills, use the first one
        if list_response.data and len(list_response.data) > 0:
            skill_id = list_response.data[0].id
            should_cleanup = False
            print(f"Using existing skill: {skill_id}")
        

            # Now get the skill
            response = litellm.get_skill(
                skill_id=skill_id,
                custom_llm_provider=custom_llm_provider,
                api_key=api_key,
                api_base=api_base,
            )

            assert response is not None
            assert isinstance(response, Skill)
            assert response.id == skill_id
            print(f"GET - Retrieved skill: {response}")



    def test_delete_skill(self):
        """
        Test deleting a skill.
        
        Note: Anthropic requires deleting all skill versions before deleting the skill itself.
        This test is currently skipped as it would require additional API calls to delete versions.
        """
        import time
        
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        pytest.skip("Anthropic requires deleting all skill versions first - skipping for now")

        litellm.set_verbose = True

        # Use helper to create skill zip
        skill_name = "test-delete-skill"
        
        # Use unique title to avoid conflicts
        unique_title = f"Test Delete Skill {int(time.time())}"
        
        # Create a skill specifically to delete
        with create_skill_zip(skill_name) as zip_file:
            created_skill = litellm.create_skill(
                display_title=unique_title,
                files=[zip_file],
                custom_llm_provider=custom_llm_provider,
                api_key=api_key,
                api_base=api_base,
            )
        
        # Type assertion for linter
        assert isinstance(created_skill, Skill)
        skill_id = created_skill.id
        print(f"Created skill to delete: {skill_id}")

        # Now delete the skill
        response = litellm.delete_skill(
            skill_id=skill_id,
            custom_llm_provider=custom_llm_provider,
            api_key=api_key,
            api_base=api_base,
        )

        assert response is not None
        assert isinstance(response, DeleteSkillResponse)
        assert response.type == "skill_deleted"
        print(f"Deleted skill response: {response}")


class TestAnthropicSkillsAPI(BaseSkillsAPITest):
    """
    Test Anthropic Skills API implementation.
    """

    def get_custom_llm_provider(self) -> str:
        return "anthropic"

    def get_api_key(self) -> Optional[str]:
        return os.environ.get("ANTHROPIC_API_KEY")

    def get_api_base(self) -> Optional[str]:
        return os.environ.get("ANTHROPIC_API_BASE")

