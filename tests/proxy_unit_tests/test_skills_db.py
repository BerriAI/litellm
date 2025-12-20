"""
Test LiteLLM Skills SDK with custom_llm_provider=litellm_proxy

Tests the SDK-level skills methods when using the LiteLLM database backend:
1. Create a skill using SDK and verify it was stored correctly
2. List skills using SDK
3. Get a skill by ID using SDK
4. Delete a skill using SDK
5. Skills injection hook correctly resolves skills from database
"""

import os
import sys
import zipfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.types.utils import LlmProviders

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())


@contextmanager
def create_skill_zip(skill_name: str):
    """
    Helper context manager to create a zip file for a skill.
    
    Args:
        skill_name: Name of the skill directory in test_skills_data/
        
    Yields:
        Tuple of (file handle, file content bytes)
        
    The zip file is automatically cleaned up after use.
    """
    test_dir = Path(__file__).parent.parent / "llm_translation" / "test_skills_data"
    skill_dir = test_dir / skill_name
    
    # Create a zip file containing the skill directory
    zip_path = test_dir / f"{skill_name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(skill_dir, arcname=skill_name)
        zip_file.write(skill_dir / "SKILL.md", arcname=f"{skill_name}/SKILL.md")
    
    try:
        with open(zip_path, "rb") as f:
            content = f.read()
            f.seek(0)
            yield f, content
    finally:
        # Clean up zip file
        if zip_path.exists():
            zip_path.unlink()


@pytest.fixture
def prisma_client():
    """Set up prisma client for tests."""
    from litellm.proxy.proxy_cli import append_query_params

    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    return prisma_client


@pytest.mark.asyncio
async def test_create_skill_sdk(prisma_client):
    """
    Test creating a skill using SDK with custom_llm_provider=litellm_proxy.
    
    Verifies that:
    - Skill is created with correct display_title
    - Skill ID is generated and returned
    - Skill response has correct type
    """
    setattr(proxy_server, "prisma_client", prisma_client)
    await proxy_server.prisma_client.connect()

    from litellm.skills.main import acreate_skill, adelete_skill

    # Create a skill using SDK
    skill = await acreate_skill(
        display_title="SDK Test Skill",
        extra_body={
            "description": "A test skill created via SDK",
            "instructions": "Use this skill for SDK testing",
        },
        custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
    )

    # Verify skill was created correctly
    assert skill is not None
    assert skill.id is not None
    assert skill.id.startswith("litellm_skill")
    assert skill.display_title == "SDK Test Skill"
    assert skill.type == "skill"
    assert skill.source == "custom"

    # Clean up
    await adelete_skill(
        skill_id=skill.id,
        custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
    )


@pytest.mark.asyncio
async def test_list_skills_sdk(prisma_client):
    """
    Test listing skills using SDK with custom_llm_provider=litellm_proxy.
    
    Verifies that:
    - Multiple skills can be created
    - List returns the created skills
    """
    setattr(proxy_server, "prisma_client", prisma_client)
    await proxy_server.prisma_client.connect()

    from litellm.skills.main import acreate_skill, adelete_skill, alist_skills

    # Create multiple skills
    created_skill_ids = []
    for i in range(3):
        skill = await acreate_skill(
            display_title=f"List Test Skill {i}",
            extra_body={
                "description": f"Test skill {i} for list test",
            },
            custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
        )
        created_skill_ids.append(skill.id)

    # List skills using SDK
    response = await alist_skills(
        limit=10,
        custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
    )

    # Verify we got skills back
    assert response is not None
    assert response.data is not None
    assert len(response.data) >= 3

    # Verify our created skills are in the list
    skill_ids_in_list = [s.id for s in response.data]
    for created_id in created_skill_ids:
        assert created_id in skill_ids_in_list

    # Clean up
    for skill_id in created_skill_ids:
        await adelete_skill(
            skill_id=skill_id,
            custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
        )


@pytest.mark.asyncio
async def test_get_skill_sdk(prisma_client):
    """
    Test getting a skill by ID using SDK with custom_llm_provider=litellm_proxy.
    
    Verifies that:
    - Skill can be retrieved by ID
    - Retrieved skill has correct data
    """
    setattr(proxy_server, "prisma_client", prisma_client)
    await proxy_server.prisma_client.connect()

    from litellm.skills.main import acreate_skill, adelete_skill, aget_skill

    # Create a skill
    created_skill = await acreate_skill(
        display_title="Get Test Skill",
        extra_body={
            "description": "A skill for get test",
        },
        custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
    )

    # Get the skill by ID using SDK
    retrieved_skill = await aget_skill(
        skill_id=created_skill.id,
        custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
    )

    # Verify retrieved skill matches created skill
    assert retrieved_skill is not None
    assert retrieved_skill.id == created_skill.id
    assert retrieved_skill.display_title == "Get Test Skill"

    # Clean up
    await adelete_skill(
        skill_id=created_skill.id,
        custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
    )


@pytest.mark.asyncio
async def test_delete_skill_sdk(prisma_client):
    """
    Test deleting a skill using SDK with custom_llm_provider=litellm_proxy.
    
    Verifies that:
    - Skill can be deleted by ID
    - Deleted skill cannot be retrieved
    """
    setattr(proxy_server, "prisma_client", prisma_client)
    await proxy_server.prisma_client.connect()

    from litellm.skills.main import acreate_skill, adelete_skill, aget_skill

    # Create a skill
    created_skill = await acreate_skill(
        display_title="Delete Test Skill",
        extra_body={
            "description": "A skill to be deleted",
        },
        custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
    )

    # Verify skill exists
    retrieved = await aget_skill(
        skill_id=created_skill.id,
        custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
    )
    assert retrieved is not None

    # Delete the skill using SDK
    result = await adelete_skill(
        skill_id=created_skill.id,
        custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
    )
    assert result.id == created_skill.id
    assert result.type == "skill_deleted"

    # Verify skill no longer exists
    with pytest.raises(Exception):
        await aget_skill(
            skill_id=created_skill.id,
            custom_llm_provider=LlmProviders.LITELLM_PROXY.value,
        )
