"""
End-to-end test for LiteLLM Skills with Messages API.

Tests the slack-gif-creator skill with GPT-4o via messages API
to verify skills work correctly and can generate a GIF.
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
    """Create a ZIP file from a skill folder in test_skills_data."""
    test_dir = Path(__file__).parent / "test_skills_data"
    skill_dir = test_dir / skill_name
    
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in skill_dir.rglob("*"):
            if file_path.is_file():
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
@pytest.mark.skip(reason="local testing only")
async def test_slack_gif_skill_creates_gif(prisma_client):
    """
    Test slack-gif-creator skill generates a GIF using GPT-4o via messages API.
    
    Flow:
    1. Store skill in LiteLLM DB
    2. Hook resolves skill, adds litellm_code_execution tool, injects SKILL.md
    3. Make GPT-4o call via messages API
    4. Hook handles code execution loop
    5. Verify GIF is generated
    """
    litellm._turn_on_debug()
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    
    setattr(litellm.proxy.proxy_server, "prisma_client", prisma_client)
    await litellm.proxy.proxy_server.prisma_client.connect()

    from litellm.llms.litellm_proxy.skills.handler import LiteLLMSkillsHandler
    from litellm.proxy.hooks.litellm_skills import SkillsInjectionHook
    from litellm.types.utils import CallTypes

    # 1. Store skill in DB
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
    
    try:
        # 2. Build request with container.skills (messages API spec)
        request_data = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": "Create a simple bouncing red ball GIF for Slack emoji."
                }
            ],
            "container": {
                "skills": [
                    {"type": "custom", "skill_id": f"litellm:{created_skill.skill_id}"}
                ]
            },
        }
        
        # 3. Pre-call hook resolves skill
        user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
        cache = DualCache()
        
        transformed = await hook.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=cache,
            data=request_data,
            call_type="anthropic_messages",
        )
        assert isinstance(transformed, dict)
        
        # Hook returns Anthropic-format tools for messages API
        tool_names = [t.get('name') for t in transformed.get('tools', [])]
        print(f"\nTools after hook: {tool_names}")
        assert "litellm_code_execution" in tool_names, "Should have litellm_code_execution tool"
        
        # 4. Make GPT-4o call via messages API (tools already in Anthropic format)
        print("\n--- Making GPT-4o call via messages API ---")
        response = await litellm.anthropic.acreate(
            model=transformed["model"],
            max_tokens=transformed.get("max_tokens", 4096),
            messages=transformed["messages"],
            tools=transformed.get("tools"),
        )
        
        print(f"Initial response: {response}")
        
        # 5. Post-call hook handles code execution loop
        final_response = await hook.async_post_call_success_deployment_hook(
            request_data=transformed,
            response=response,
            call_type=CallTypes.anthropic_messages,
        )
        
        if final_response:
            response = final_response
            print("Code execution completed!")
        
        # 6. Check for generated files (handle both dict and object response)
        if isinstance(response, dict):
            generated_files = response.get("_litellm_generated_files", [])
        else:
            generated_files = getattr(response, "_litellm_generated_files", [])
        print(f"\nGenerated files: {len(generated_files)}")
        
        if generated_files:
            import base64
            for f in generated_files:
                print(f"  - {f['name']} ({f['size']} bytes)")
                if f['name'].endswith('.gif'):
                    content = base64.b64decode(f['content_base64'])
                    assert content[:6] in [b'GIF89a', b'GIF87a'], "Should be valid GIF"
                    print("    Valid GIF!")
            print("\nSUCCESS - GIF generated!")
        else:
            # Print response for debugging
            if hasattr(response, "choices"):
                print(f"\nResponse: {response.choices[0].message}")
            else:
                print(f"\nResponse: {response}")
    
    finally:
        await LiteLLMSkillsHandler.delete_skill(skill_id=created_skill.skill_id)
