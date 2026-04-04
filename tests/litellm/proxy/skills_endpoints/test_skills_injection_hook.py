"""
Tests for SkillsInjectionHook - pre-call skill processing and system prompt injection.

Covers:
- Request opt-in via container.skills
- OpenAI system prompt injection (with and without existing system message)
- Anthropic tool conversion
- Provider fallback behavior
- Skills skipped for non-completion call types
"""

import io
import zipfile
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy._types import LiteLLM_SkillsTable, UserAPIKeyAuth
from litellm.proxy.hooks.litellm_skills.main import SkillsInjectionHook


def _make_skill(
    skill_id: str = "litellm_skill_test1",
    display_title: str = "Test Skill",
    description: str = "A test skill",
    instructions: str = "Follow these instructions for testing.",
    file_content: bytes | None = None,
) -> LiteLLM_SkillsTable:
    """Create a LiteLLM_SkillsTable for testing."""
    if file_content is None:
        # Create a minimal ZIP with SKILL.md
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "test_skill/SKILL.md",
                "---\nname: test-skill\ndescription: A test\n---\n\n"
                "Follow these instructions for testing.",
            )
        file_content = buf.getvalue()

    return LiteLLM_SkillsTable(
        skill_id=skill_id,
        display_title=display_title,
        description=description,
        instructions=instructions,
        source="custom",
        file_content=file_content,
        file_name="skill.zip",
        file_type="application/zip",
        created_at=datetime(2026, 3, 21),
        updated_at=datetime(2026, 3, 21),
    )


def _make_skill_with_code(
    skill_id: str = "litellm_skill_code1",
    display_title: str = "Code Skill",
) -> LiteLLM_SkillsTable:
    """Create a skill with Python files in the ZIP."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "code_skill/SKILL.md",
            "---\nname: code-skill\n---\n\nA skill with code.",
        )
        zf.writestr("code_skill/main.py", "def run(): return 42")

    return LiteLLM_SkillsTable(
        skill_id=skill_id,
        display_title=display_title,
        instructions="A skill with code.",
        source="custom",
        file_content=buf.getvalue(),
        file_name="skill.zip",
        file_type="application/zip",
        created_at=datetime(2026, 3, 21),
        updated_at=datetime(2026, 3, 21),
    )


class TestPreCallHookOptIn:
    """Tests for the opt-in activation model via container.skills."""

    @pytest.mark.asyncio
    async def test_no_container_passes_through(self):
        """Test that requests without container are unchanged."""
        hook = SkillsInjectionHook()
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = await hook.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test"),
            cache=None,
            data=data,
            call_type="completion",
        )

        assert result == data
        assert "tools" not in result

    @pytest.mark.asyncio
    async def test_empty_container_passes_through(self):
        """Test that requests with empty container are unchanged."""
        hook = SkillsInjectionHook()
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "container": {},
        }

        result = await hook.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test"),
            cache=None,
            data=data,
            call_type="completion",
        )

        assert result == data

    @pytest.mark.asyncio
    async def test_non_completion_call_type_skipped(self):
        """Test that non-completion call types are not processed."""
        hook = SkillsInjectionHook()
        data = {
            "model": "gpt-4o",
            "container": {"skills": [{"skill_id": "litellm_skill_x"}]},
        }

        result = await hook.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test"),
            cache=None,
            data=data,
            call_type="embedding",
        )

        assert result == data

    @pytest.mark.asyncio
    async def test_litellm_skill_fetched_from_db(self):
        """Test that litellm_* skills are fetched from DB."""
        hook = SkillsInjectionHook()
        skill = _make_skill()

        with patch.object(
            hook, "_fetch_skill_from_db", new_callable=AsyncMock, return_value=skill
        ):
            data = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "container": {
                    "skills": [{"skill_id": "litellm_skill_test1", "type": "anthropic"}]
                },
            }

            result = await hook.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=None,
                data=data,
                call_type="completion",
            )

            # container should be removed after processing
            assert "container" not in result

    @pytest.mark.asyncio
    async def test_missing_skill_raises_404(self):
        """Test that referencing a nonexistent skill raises 404."""
        from fastapi import HTTPException

        hook = SkillsInjectionHook()

        with patch.object(
            hook, "_fetch_skill_from_db", new_callable=AsyncMock, return_value=None
        ):
            data = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "container": {
                    "skills": [{"skill_id": "litellm_skill_nonexistent"}]
                },
            }

            with pytest.raises(HTTPException) as exc_info:
                await hook.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=None,
                    data=data,
                    call_type="completion",
                )

            assert exc_info.value.status_code == 404
            assert "litellm_skill_nonexistent" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_anthropic_native_skill_on_native_provider(self):
        """Test that Anthropic skill_ IDs pass through on native providers."""
        hook = SkillsInjectionHook()

        with patch(
            "litellm.llms.litellm_proxy.skills.skill_applicator.get_provider_from_model",
            return_value="anthropic",
        ):
            data = {
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "Hello"}],
                "container": {
                    "skills": [{"skill_id": "skill_01abc123", "type": "anthropic"}]
                },
            }

            result = await hook.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                cache=None,
                data=data,
                call_type="completion",
            )

            # Anthropic native skill should pass through
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_anthropic_native_skill_on_non_native_provider_fails(self):
        """Test that Anthropic skill_ IDs fail on non-native providers."""
        from fastapi import HTTPException

        hook = SkillsInjectionHook()

        with patch(
            "litellm.llms.litellm_proxy.skills.skill_applicator.get_provider_from_model",
            return_value="openai",
        ):
            data = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "container": {
                    "skills": [{"skill_id": "skill_01abc123", "type": "anthropic"}]
                },
            }

            with pytest.raises(HTTPException) as exc_info:
                await hook.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=None,
                    data=data,
                    call_type="completion",
                )

            assert exc_info.value.status_code == 400
            assert "does not support native skills" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_invalid_skill_id_prefix_fails(self):
        """Test that skill IDs with unrecognized prefixes fail."""
        from fastapi import HTTPException

        hook = SkillsInjectionHook()

        with patch(
            "litellm.llms.litellm_proxy.skills.skill_applicator.get_provider_from_model",
            return_value="openai",
        ):
            data = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
                "container": {
                    "skills": [{"skill_id": "random_garbage_id", "type": "anthropic"}]
                },
            }

            with pytest.raises(HTTPException) as exc_info:
                await hook.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=None,
                    data=data,
                    call_type="completion",
                )

            assert exc_info.value.status_code == 400
            assert "Invalid skill_id" in str(exc_info.value.detail)


class TestSystemPromptInjection:
    """Tests for OpenAI-style system prompt injection."""

    def test_inject_into_new_system_message(self):
        """Test injecting skill content when no system message exists."""
        from litellm.llms.litellm_proxy.skills.prompt_injection import (
            SkillPromptInjectionHandler,
        )

        handler = SkillPromptInjectionHandler()

        data = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = handler.inject_skill_content_to_messages(
            data,
            ["## Skill: Test\n\nDo testing."],
            use_anthropic_format=False,
        )

        messages = result["messages"]
        assert messages[0]["role"] == "system"
        assert "Available Skills" in messages[0]["content"]
        assert "Do testing." in messages[0]["content"]
        assert messages[1]["role"] == "user"

    def test_inject_into_existing_system_message(self):
        """Test injecting skill content appends to existing system message."""
        from litellm.llms.litellm_proxy.skills.prompt_injection import (
            SkillPromptInjectionHandler,
        )

        handler = SkillPromptInjectionHandler()

        data = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        }

        result = handler.inject_skill_content_to_messages(
            data,
            ["## Skill: Test\n\nDo testing."],
            use_anthropic_format=False,
        )

        messages = result["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"].startswith("You are a helpful assistant.")
        assert "Available Skills" in messages[0]["content"]
        assert "Do testing." in messages[0]["content"]

    def test_inject_anthropic_format_uses_system_param(self):
        """Test Anthropic format injects into top-level 'system' parameter."""
        from litellm.llms.litellm_proxy.skills.prompt_injection import (
            SkillPromptInjectionHandler,
        )

        handler = SkillPromptInjectionHandler()

        data = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = handler.inject_skill_content_to_messages(
            data,
            ["## Skill: Test\n\nDo testing."],
            use_anthropic_format=True,
        )

        assert "system" in result
        assert "Available Skills" in result["system"]
        assert "Do testing." in result["system"]

    def test_inject_anthropic_format_appends_to_existing_system(self):
        """Test Anthropic format appends to existing system parameter."""
        from litellm.llms.litellm_proxy.skills.prompt_injection import (
            SkillPromptInjectionHandler,
        )

        handler = SkillPromptInjectionHandler()

        data = {
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = handler.inject_skill_content_to_messages(
            data,
            ["## Skill: Test\n\nDo testing."],
            use_anthropic_format=True,
        )

        assert result["system"].startswith("You are a helpful assistant.")
        assert "Do testing." in result["system"]

    def test_inject_multiple_skills(self):
        """Test injecting multiple skills creates separate sections."""
        from litellm.llms.litellm_proxy.skills.prompt_injection import (
            SkillPromptInjectionHandler,
        )

        handler = SkillPromptInjectionHandler()

        data = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = handler.inject_skill_content_to_messages(
            data,
            [
                "## Skill: Alpha\n\nAlpha instructions.",
                "## Skill: Beta\n\nBeta instructions.",
            ],
            use_anthropic_format=False,
        )

        system_content = result["messages"][0]["content"]
        assert "Alpha instructions." in system_content
        assert "Beta instructions." in system_content

    def test_inject_empty_list_no_change(self):
        """Test that empty skill list doesn't modify data."""
        from litellm.llms.litellm_proxy.skills.prompt_injection import (
            SkillPromptInjectionHandler,
        )

        handler = SkillPromptInjectionHandler()

        data = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = handler.inject_skill_content_to_messages(
            data, [], use_anthropic_format=False
        )

        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"


class TestSkillApplicator:
    """Tests for SkillApplicator provider-specific strategies."""

    @pytest.mark.asyncio
    async def test_openai_uses_system_prompt_strategy(self):
        """Test OpenAI provider uses system prompt injection."""
        from litellm.llms.litellm_proxy.skills.skill_applicator import SkillApplicator

        applicator = SkillApplicator()
        skill = _make_skill()

        data = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = await applicator.apply_skills(data, [skill], provider="openai")

        # Should have injected into system message
        assert result["messages"][0]["role"] == "system"
        assert "Test Skill" in result["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_anthropic_uses_tool_conversion_strategy(self):
        """Test Anthropic provider uses tool conversion strategy."""
        from litellm.llms.litellm_proxy.skills.skill_applicator import SkillApplicator

        applicator = SkillApplicator()
        skill = _make_skill()

        data = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = await applicator.apply_skills(data, [skill], provider="anthropic")

        # Should have added tools with Anthropic format (name, description, input_schema)
        assert "tools" in result
        assert len(result["tools"]) >= 1
        tool = result["tools"][0]
        assert "name" in tool
        assert "input_schema" in tool

    @pytest.mark.asyncio
    async def test_azure_uses_system_prompt_strategy(self):
        """Test Azure provider uses system prompt injection (same as OpenAI)."""
        from litellm.llms.litellm_proxy.skills.skill_applicator import SkillApplicator

        applicator = SkillApplicator()
        skill = _make_skill()

        data = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = await applicator.apply_skills(data, [skill], provider="azure")

        assert result["messages"][0]["role"] == "system"
        assert "Test Skill" in result["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_unknown_provider_defaults_to_system_prompt(self):
        """Test unknown providers default to system prompt injection."""
        from litellm.llms.litellm_proxy.skills.skill_applicator import SkillApplicator

        applicator = SkillApplicator()
        skill = _make_skill()

        data = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = await applicator.apply_skills(
            data, [skill], provider="unknown_provider"
        )

        assert result["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_empty_skills_list_no_change(self):
        """Test that empty skills list doesn't modify data."""
        from litellm.llms.litellm_proxy.skills.skill_applicator import SkillApplicator

        applicator = SkillApplicator()

        data = {
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = await applicator.apply_skills(data, [], provider="openai")

        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_supports_native_skills(self):
        """Test native skills support detection."""
        from litellm.llms.litellm_proxy.skills.skill_applicator import SkillApplicator

        applicator = SkillApplicator()

        assert applicator.supports_native_skills("anthropic") is True
        assert applicator.supports_native_skills("openai") is False
        assert applicator.supports_native_skills("azure") is False
        assert applicator.supports_native_skills("bedrock") is False
        assert applicator.supports_native_skills("gemini") is False
        assert applicator.supports_native_skills("some_new_provider") is False


class TestSkillContentExtraction:
    """Tests for skill content extraction from ZIP files."""

    def test_extract_skill_md_from_zip(self):
        """Test extracting SKILL.md content from ZIP."""
        from litellm.llms.litellm_proxy.skills.prompt_injection import (
            SkillPromptInjectionHandler,
        )

        handler = SkillPromptInjectionHandler()
        skill = _make_skill()

        content = handler.extract_skill_content(skill)

        assert content is not None
        assert "Follow these instructions" in content

    def test_extract_fallback_to_instructions(self):
        """Test fallback to instructions field when no file_content."""
        from litellm.llms.litellm_proxy.skills.prompt_injection import (
            SkillPromptInjectionHandler,
        )

        handler = SkillPromptInjectionHandler()
        skill = LiteLLM_SkillsTable(
            skill_id="litellm_skill_nf",
            instructions="Fallback instructions",
            source="custom",
        )

        content = handler.extract_skill_content(skill)

        assert content == "Fallback instructions"

    def test_extract_all_files_from_zip(self):
        """Test extracting all files from skill ZIP."""
        from litellm.llms.litellm_proxy.skills.prompt_injection import (
            SkillPromptInjectionHandler,
        )

        handler = SkillPromptInjectionHandler()
        skill = _make_skill_with_code()

        files = handler.extract_all_files(skill)

        assert "SKILL.md" in files
        assert "main.py" in files
        assert files["main.py"] == b"def run(): return 42"


class TestMessagesAPIProcessing:
    """Tests for _process_for_messages_api in the hook."""

    def test_process_removes_container(self):
        """Test that processing removes the container field."""
        hook = SkillsInjectionHook()
        skill = _make_skill()

        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
            "container": {"skills": [{"skill_id": "litellm_skill_test1"}]},
        }

        result = hook._process_for_messages_api(
            data=data, litellm_skills=[skill], use_anthropic_format=False
        )

        assert "container" not in result

    def test_process_adds_tools(self):
        """Test that processing adds skill as Anthropic-style tool."""
        hook = SkillsInjectionHook()
        skill = _make_skill()

        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = hook._process_for_messages_api(
            data=data, litellm_skills=[skill], use_anthropic_format=False
        )

        assert "tools" in result
        # Should have skill tool + code_execution tool
        assert len(result["tools"]) >= 1

    def test_process_injects_system_prompt(self):
        """Test that processing injects skill content into system prompt."""
        hook = SkillsInjectionHook()
        skill = _make_skill()

        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = hook._process_for_messages_api(
            data=data, litellm_skills=[skill], use_anthropic_format=False
        )

        # Check system message was injected
        messages = result["messages"]
        system_msgs = [m for m in messages if m.get("role") == "system"]
        assert len(system_msgs) > 0

    def test_process_with_code_files_enables_execution(self):
        """Test that skills with code files enable code execution."""
        hook = SkillsInjectionHook()
        skill = _make_skill_with_code()

        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Generate something"}],
        }

        result = hook._process_for_messages_api(
            data=data, litellm_skills=[skill], use_anthropic_format=False
        )

        # Should have code execution enabled in metadata
        assert result.get("litellm_metadata", {}).get(
            "_litellm_code_execution_enabled"
        )
        assert "_skill_files" in result.get("litellm_metadata", {})


class TestPreCallProviderRouting:
    """
    Tests that the pre-call hook routes to the correct strategy based on provider.

    Non-native providers (OpenAI, Azure, Bedrock, etc.):
    - Skill content injected into system prompt
    - No skill tools added
    - container removed

    Native providers (Anthropic, azure_ai, databricks):
    - Skill converted to Anthropic-style tool
    - Skill content injected into system prompt
    - Code execution tool added (if skill has files)
    - container removed
    """

    @pytest.mark.asyncio
    async def test_openai_gets_system_prompt_only(self):
        """OpenAI should get system prompt injection, no tools from skill."""
        hook = SkillsInjectionHook()
        skill = _make_skill()

        with patch.object(
            hook, "_fetch_skill_from_db", new_callable=AsyncMock, return_value=skill
        ):
            with patch(
                "litellm.llms.litellm_proxy.skills.skill_applicator.get_provider_from_model",
                return_value="openai",
            ):
                data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "container": {
                        "skills": [{"skill_id": "litellm_skill_test1"}]
                    },
                }

                result = await hook.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=None,
                    data=data,
                    call_type="completion",
                )

                # System prompt should be injected
                system_msgs = [
                    m for m in result["messages"] if m.get("role") == "system"
                ]
                assert len(system_msgs) == 1
                assert "Test Skill" in system_msgs[0]["content"]

                # No tools should be added for OpenAI
                assert "tools" not in result

                # container should be removed
                assert "container" not in result

    @pytest.mark.asyncio
    async def test_azure_gets_system_prompt_only(self):
        """Azure should get system prompt injection, no tools from skill."""
        hook = SkillsInjectionHook()
        skill = _make_skill()

        with patch.object(
            hook, "_fetch_skill_from_db", new_callable=AsyncMock, return_value=skill
        ):
            with patch(
                "litellm.llms.litellm_proxy.skills.skill_applicator.get_provider_from_model",
                return_value="azure",
            ):
                data = {
                    "model": "azure/gpt-4o",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "container": {
                        "skills": [{"skill_id": "litellm_skill_test1"}]
                    },
                }

                result = await hook.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=None,
                    data=data,
                    call_type="completion",
                )

                system_msgs = [
                    m for m in result["messages"] if m.get("role") == "system"
                ]
                assert len(system_msgs) == 1
                assert "tools" not in result
                assert "container" not in result

    @pytest.mark.asyncio
    async def test_bedrock_gets_system_prompt_only(self):
        """Bedrock should get system prompt injection, no tools from skill."""
        hook = SkillsInjectionHook()
        skill = _make_skill()

        with patch.object(
            hook, "_fetch_skill_from_db", new_callable=AsyncMock, return_value=skill
        ):
            with patch(
                "litellm.llms.litellm_proxy.skills.skill_applicator.get_provider_from_model",
                return_value="bedrock",
            ):
                data = {
                    "model": "bedrock/anthropic.claude-v2",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "container": {
                        "skills": [{"skill_id": "litellm_skill_test1"}]
                    },
                }

                result = await hook.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=None,
                    data=data,
                    call_type="completion",
                )

                system_msgs = [
                    m for m in result["messages"] if m.get("role") == "system"
                ]
                assert len(system_msgs) == 1
                assert "tools" not in result
                assert "container" not in result

    @pytest.mark.asyncio
    async def test_anthropic_gets_tools_and_system_prompt(self):
        """Anthropic should get tool conversion + system prompt injection."""
        hook = SkillsInjectionHook()
        skill = _make_skill()

        with patch.object(
            hook, "_fetch_skill_from_db", new_callable=AsyncMock, return_value=skill
        ):
            with patch(
                "litellm.llms.litellm_proxy.skills.skill_applicator.get_provider_from_model",
                return_value="anthropic",
            ):
                data = {
                    "model": "claude-sonnet-4-20250514",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "container": {
                        "skills": [{"skill_id": "litellm_skill_test1"}]
                    },
                }

                result = await hook.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=None,
                    data=data,
                    call_type="completion",
                )

                # Tools should be added (skill tool)
                assert "tools" in result
                assert len(result["tools"]) >= 1

                # container should be removed
                assert "container" not in result

    @pytest.mark.asyncio
    async def test_openai_preserves_existing_system_message(self):
        """OpenAI skill injection should append to existing system message."""
        hook = SkillsInjectionHook()
        skill = _make_skill()

        with patch.object(
            hook, "_fetch_skill_from_db", new_callable=AsyncMock, return_value=skill
        ):
            with patch(
                "litellm.llms.litellm_proxy.skills.skill_applicator.get_provider_from_model",
                return_value="openai",
            ):
                data = {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello"},
                    ],
                    "container": {
                        "skills": [{"skill_id": "litellm_skill_test1"}]
                    },
                }

                result = await hook.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=None,
                    data=data,
                    call_type="completion",
                )

                system_msg = result["messages"][0]
                assert system_msg["role"] == "system"
                # Original content preserved
                assert system_msg["content"].startswith("You are a helpful assistant.")
                # Skill content appended
                assert "Test Skill" in system_msg["content"]

    @pytest.mark.asyncio
    async def test_openai_no_tools_even_with_code_files(self):
        """OpenAI should NOT get tools even if skill has Python files."""
        hook = SkillsInjectionHook()
        skill = _make_skill_with_code()

        with patch.object(
            hook, "_fetch_skill_from_db", new_callable=AsyncMock, return_value=skill
        ):
            with patch(
                "litellm.llms.litellm_proxy.skills.skill_applicator.get_provider_from_model",
                return_value="openai",
            ):
                data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "container": {
                        "skills": [{"skill_id": "litellm_skill_code1"}]
                    },
                }

                result = await hook.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=None,
                    data=data,
                    call_type="completion",
                )

                # System prompt should have skill content
                system_msgs = [
                    m for m in result["messages"] if m.get("role") == "system"
                ]
                assert len(system_msgs) == 1

                # No tools — OpenAI just gets the prompt
                assert "tools" not in result

    @pytest.mark.asyncio
    async def test_multiple_skills_openai(self):
        """Multiple skills should all be injected into system prompt for OpenAI."""
        hook = SkillsInjectionHook()
        skill1 = _make_skill(
            skill_id="litellm_skill_a",
            display_title="Skill Alpha",
            instructions="Alpha instructions.",
        )
        skill2 = _make_skill(
            skill_id="litellm_skill_b",
            display_title="Skill Beta",
            instructions="Beta instructions.",
        )

        async def _fetch(skill_id):
            return {"litellm_skill_a": skill1, "litellm_skill_b": skill2}.get(skill_id)

        with patch.object(hook, "_fetch_skill_from_db", side_effect=_fetch):
            with patch(
                "litellm.llms.litellm_proxy.skills.skill_applicator.get_provider_from_model",
                return_value="openai",
            ):
                data = {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "container": {
                        "skills": [
                            {"skill_id": "litellm_skill_a"},
                            {"skill_id": "litellm_skill_b"},
                        ]
                    },
                }

                result = await hook.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(api_key="test"),
                    cache=None,
                    data=data,
                    call_type="completion",
                )

                system_content = result["messages"][0]["content"]
                assert "Skill Alpha" in system_content
                assert "Skill Beta" in system_content
                assert "tools" not in result
