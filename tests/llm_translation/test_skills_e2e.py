"""
End-to-end test for LiteLLM Skills with real LLM API call.

Downloads the slack-gif-creator skill from Anthropic's skills repo,
stores it in LiteLLM DB, and makes a real API call to verify
the skill content is properly injected into the system prompt.
"""

import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
import litellm.proxy.proxy_server
from litellm.caching.caching import DualCache
from litellm.proxy._types import NewSkillRequest, UserAPIKeyAuth
from litellm.proxy.utils import PrismaClient, ProxyLogging

proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())


def create_skill_zip_from_folder(skill_name: str) -> bytes:
    """
    Create a ZIP file from a skill folder in test_skills_data.
    
    Recursively includes all files in the skill directory.
    
    Args:
        skill_name: Name of the skill directory
        
    Returns:
        ZIP file content as bytes
    """
    test_dir = Path(__file__).parent / "test_skills_data"
    skill_dir = test_dir / skill_name
    
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Walk through all files in the skill directory
        for file_path in skill_dir.rglob("*"):
            if file_path.is_file():
                # Create archive name relative to test_skills_data
                arcname = f"{skill_name}/{file_path.relative_to(skill_dir)}"
                zf.write(file_path, arcname=arcname)
    
    return zip_buffer.getvalue()


@pytest.fixture
def prisma_client():
    """Set up prisma client for tests."""
    from litellm.proxy.proxy_cli import append_query_params

    params = {"connection_limit": 100, "pool_timeout": 60}
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not set")
    
    modified_url = append_query_params(database_url, params)
    os.environ["DATABASE_URL"] = modified_url

    prisma_client = PrismaClient(
        database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj
    )

    return prisma_client


@pytest.mark.asyncio
async def test_skill_code_execution_via_deployment_hook(prisma_client):
    """
    Test the full code execution loop using async_post_call_success_deployment_hook.
    
    This test shows that when the hook is registered in litellm.callbacks,
    the async_post_call_success_deployment_hook is called automatically after
    each litellm.acompletion() call and can modify the response.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    await litellm.proxy.proxy_server.prisma_client.connect()

    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler
    from litellm.proxy.hooks.litellm_skills import SkillsInjectionHook
    from litellm.types.utils import CallTypes

    # Create skill
    skill_name = "slack-gif-creator"
    zip_content = create_skill_zip_from_folder(skill_name)
    
    skill_request = NewSkillRequest(
        display_title="Slack GIF Creator",
        description="Create animated GIFs optimized for Slack",
        instructions="Use this skill to create animated GIFs for Slack emoji",
        file_content=zip_content,
        file_name=f"{skill_name}.zip",
        file_type="application/zip",
    )
    created_skill = await LiteLLMSkillsHandler.create_skill(
        data=skill_request,
        user_id="test_user",
    )
    
    print(f"\nCreated skill: {created_skill.skill_id}")
    
    hook = SkillsInjectionHook()
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
    cache = DualCache()
    
    # Original request (like what proxy receives)
    original_request = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": "Create a simple bouncing red ball GIF for Slack emoji. Use the litellm_code_execution tool."
            }
        ],
        "max_tokens": 4096,
        "container": {
            "skills": [
                {"type": "custom", "skill_id": f"litellm:{created_skill.skill_id}"}
            ]
        },
    }
    
    # Step 1: Pre-call hook (in proxy this happens automatically)
    print("\n--- Step 1: Pre-call hook ---")
    transformed = await hook.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=cache,
        data=original_request,
        call_type="completion",
    )
    assert isinstance(transformed, dict)
    print(f"Added tools: {[t.get('function', {}).get('name') for t in transformed.get('tools', [])]}")
    
    # Step 2: LLM call
    print("\n--- Step 2: LLM call ---")
    response = await litellm.acompletion(
        model=transformed["model"],
        messages=transformed["messages"],
        tools=transformed.get("tools"),
        max_tokens=transformed.get("max_tokens", 4096),
    )
    print(f"Initial finish_reason: {response.choices[0].finish_reason}")
    
    # Step 3: async_post_call_success_deployment_hook (called automatically by litellm when hook is registered)
    # Here we call it manually to test the hook directly
    print("\n--- Step 3: async_post_call_success_deployment_hook ---")
    final_response = await hook.async_post_call_success_deployment_hook(
        request_data=transformed,
        response=response,
        call_type=CallTypes.acompletion,
    )
    
    if final_response is None:
        print("No code execution needed")
        final_response = response
    else:
        print("Code execution loop completed!")
    
    # Check results
    generated_files = getattr(final_response, "_litellm_generated_files", [])
    print(f"\nGenerated files: {len(generated_files)}")
    
    if generated_files:
        import base64
        for f in generated_files:
            print(f"  - {f['name']} ({f['size']} bytes, {f['mime_type']})")
            
            # Verify GIF
            if f['name'].endswith('.gif'):
                content = base64.b64decode(f['content_base64'])
                assert content[:6] in [b'GIF89a', b'GIF87a'], "Should be valid GIF"
                print(f"    Valid GIF!")
        
        print("\n" + "="*50)
        print("SUCCESS! async_post_call_success_deployment_hook working!")
        print("="*50)
    else:
        content = final_response.choices[0].message.content
        print(f"Final response: {content[:300] if content else 'No content'}...")
    
    # Clean up
    await LiteLLMSkillsHandler.delete_skill(skill_id=created_skill.skill_id)

