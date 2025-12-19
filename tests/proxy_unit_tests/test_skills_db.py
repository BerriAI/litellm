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
    assert skill.id.startswith("skill_")
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


@pytest.mark.asyncio
async def test_skills_injection_hook_resolves_db_skill(prisma_client):
    """
    Test that the skills injection hook correctly resolves skills from the database.
    
    Creates a skill from a zip file (test_skills_data/test-skill-litellm) and verifies:
    - Hook fetches skill with 'litellm:' prefix from DB
    - Hook converts skill to tool for non-Anthropic models
    - Tool has correct function name and description
    - File content is stored and can be retrieved
    """
    setattr(proxy_server, "prisma_client", prisma_client)
    await proxy_server.prisma_client.connect()

    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler
    from litellm.proxy._types import NewSkillRequest
    from litellm.proxy.hooks.skills_injection_hook import SkillsInjectionHook

    # Create a skill using the test skill zip file
    skill_name = "test-skill-litellm"
    with create_skill_zip(skill_name) as (zip_file, zip_content):
        skill_request = NewSkillRequest(
            display_title="Hook Test Skill from Zip",
            description="A skill for hook testing loaded from zip",
            instructions="When the user asks about testing, use this skill to help them.",
            file_content=zip_content,
            file_name=f"{skill_name}.zip",
            file_type="application/zip",
        )
        created_skill = await LiteLLMSkillsHandler.create_skill(
            data=skill_request,
            user_id="test_user",
        )

    # Verify file content was stored
    assert created_skill.file_content is not None
    assert created_skill.file_name == f"{skill_name}.zip"
    assert created_skill.file_type == "application/zip"

    # Verify the zip content can be read back
    stored_zip = BytesIO(created_skill.file_content)
    with zipfile.ZipFile(stored_zip, "r") as zf:
        # Check that the zip contains the expected files
        names = zf.namelist()
        assert any(skill_name in name for name in names)
        assert any("SKILL.md" in name for name in names)
        
        # Read the SKILL.md content to verify it was stored correctly
        skill_md_path = f"{skill_name}/SKILL.md"
        if skill_md_path in names:
            skill_md_content = zf.read(skill_md_path).decode("utf-8")
            assert len(skill_md_content) > 0

    # Initialize the hook
    hook = SkillsInjectionHook()

    # Create a request with the skill using 'litellm:' prefix
    request_data = {
        "model": "gpt-4",  # Non-Anthropic model
        "messages": [{"role": "user", "content": "Help me with testing"}],
        "container": {
            "skills": [
                {"type": "custom", "skill_id": f"litellm:{created_skill.skill_id}"}
            ]
        },
    }

    # Process the request through the hook
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
    cache = DualCache()
    
    result = await hook.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=request_data,
        call_type="completion",
    )

    # Verify the hook processed the request
    assert result is not None
    assert isinstance(result, dict)
    
    # For non-Anthropic models, container should be removed and tools should be added
    assert "container" not in result
    assert "tools" in result
    assert len(result["tools"]) == 1
    
    # Verify the tool was created correctly
    tool = result["tools"][0]
    assert tool["type"] == "function"
    assert "function" in tool
    assert tool["function"]["name"] == created_skill.skill_id.replace("-", "_")
    assert "testing" in tool["function"]["description"].lower()
    
    # Verify skill content was injected into system prompt
    messages = result.get("messages", [])
    assert len(messages) >= 2  # Original user message + injected system message
    
    # Find system message with skill content
    system_msg = None
    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg
            break
    
    assert system_msg is not None, "System message with skill content should be present"
    assert "Available Skills" in system_msg["content"]
    assert "Hook Test Skill from Zip" in system_msg["content"]

    # Clean up
    await LiteLLMSkillsHandler.delete_skill(skill_id=created_skill.skill_id)

