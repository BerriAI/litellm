"""
Tests for Skills API operations across providers
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Optional

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.types.llms.anthropic_skills import (
    DeleteSkillResponse,
    ListSkillsResponse,
    Skill,
)


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
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        litellm.set_verbose = True

        # Create skill with minimal parameters
        response = litellm.create_skill(
            display_title="Test Skill - Do Not Delete",
            files=[],
            custom_llm_provider=custom_llm_provider,
            api_key=api_key,
            api_base=api_base,
        )

        assert response is not None
        assert isinstance(response, Skill)
        assert response.id is not None
        print(f"Created skill: {response}")
        print(f"Skill ID: {response.id}")

        # Store skill_id for cleanup in other tests
        return response.id

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
        print(f"API Key: {api_key[:10]}...")
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
        """
        custom_llm_provider = self.get_custom_llm_provider()
        api_key = self.get_api_key()
        api_base = self.get_api_base()

        if not api_key:
            pytest.skip(f"No API key provided for {custom_llm_provider}")

        litellm.set_verbose = True

        # Create a skill specifically to delete
        created_skill = litellm.create_skill(
            display_title="Test Delete Skill - To Be Deleted",
            files=[],
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

