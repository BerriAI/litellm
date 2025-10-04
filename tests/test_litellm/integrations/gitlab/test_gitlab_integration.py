import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.gitlab.gitlab_prompt_manager import GitLabPromptManager


# -----------------------------
# Basic init & template loading
# -----------------------------
@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_manager_initialization_with_root_folder(mock_client_class):
    """Loads a prompt from the repo root when no prompts_path is specified."""
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = """---
model: gpt-4
temperature: 0.7
max_tokens: 150
---
System: You are a helpful assistant.

User: {{user_message}}"""
    mock_client_class.return_value = mock_client

    config = {
        "project": "group/sub/repo",
        "access_token": "glpat_xxx",
        # no prompts_path -> root
    }

    manager = GitLabPromptManager(config, prompt_id="test_prompt")
    # Should have loaded the prompt
    assert "test_prompt" in manager.prompt_manager.prompts
    template = manager.prompt_manager.prompts["test_prompt"]
    assert template.model == "gpt-4"
    assert template.temperature == 0.7
    assert template.max_tokens == 150

    # Ensures correct file path was requested at repo root (test_prompt.prompt)
    mock_client.get_file_content.assert_called_with("test_prompt.prompt", ref=None)

    # Rendering
    rendered = manager.prompt_manager.render_template(
        "test_prompt", {"user_message": "What is AI?"}
    )
    assert "You are a helpful assistant." in rendered
    assert "What is AI?" in rendered


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_manager_with_prompts_path(mock_client_class):
    """Loads a prompt from a configured prompts folder; ID maps to folder + .prompt."""
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = "Hello {{name}}!"
    mock_client_class.return_value = mock_client

    config = {
        "project": "group/repo",
        "access_token": "token",
        "prompts_path": "prompts/chat",  # folder setting
    }

    manager = GitLabPromptManager(config, prompt_id="greet/hi")
    # Expected path: prompts/chat/greet/hi.prompt
    mock_client.get_file_content.assert_called_with("prompts/chat/greet/hi.prompt", ref=None)

    rendered = manager.prompt_manager.render_template("greet/hi", {"name": "World"})
    assert rendered == "Hello World!"


# -----------------------------
# Error handling / validation
# -----------------------------
@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_manager_error_handling_load(mock_client_class):
    """Errors from GitLabClient surface with helpful context."""
    mock_client = MagicMock()
    mock_client.get_file_content.side_effect = Exception("GitLab API error")
    mock_client_class.return_value = mock_client

    config = {"project": "g/s/r", "access_token": "tkn"}

    with pytest.raises(Exception, match="Failed to load prompt 'oops' from GitLab"):
        GitLabPromptManager(config, prompt_id="oops").prompt_manager  # triggers load


def test_gitlab_prompt_manager_config_validation_via_client_ctor():
    """
    If GitLabClient validates config in __init__, simulate that with a side_effect.
    Ensures manager surfaces the ValueError while building prompt_manager.
    """
    with patch(
            "litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient",
            side_effect=ValueError("project and access_token are required"),
    ):
        with pytest.raises(ValueError, match="project and access_token are required"):
            GitLabPromptManager({}).prompt_manager


# -----------------------------
# Message parsing
# -----------------------------
@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_manager_message_parsing(mock_client_class):
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = """---
model: gpt-4
---
System: You are a helpful assistant.

User: {{user_message}}

Assistant: I'll help you with that."""
    mock_client_class.return_value = mock_client

    config = {"project": "g/s/r", "access_token": "t"}

    manager = GitLabPromptManager(config, prompt_id="conversation_prompt")

    messages = manager._parse_prompt_to_messages(
        "System: You are a helpful assistant.\n\nUser: Hello!\n\nAssistant: Hi there!"
    )
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello!"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Hi there!"


# -----------------------------
# pre_call_hook behavior & ref precedence
# -----------------------------
@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_manager_pre_call_hook_updates_params(mock_client_class):
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = """---
model: gpt-4o
temperature: 0.8
max_tokens: 256
---
System: You are a helpful assistant.

User: {{user_message}}"""
    mock_client_class.return_value = mock_client

    config = {"project": "g/s/r", "access_token": "tkn"}

    manager = GitLabPromptManager(config, prompt_id="test_prompt")

    original_messages = [{"role": "user", "content": "This will be ignored"}]
    litellm_params = {"api_key": "keep-me"}

    result_messages, result_params = manager.pre_call_hook(
        user_id="u",
        messages=original_messages,
        litellm_params=litellm_params,
        prompt_id="test_prompt",
        prompt_variables={"user_message": "What is AI?"},
    )

    # Prompt parsed into messages
    assert len(result_messages) == 2
    assert result_messages[0]["role"] == "system"
    assert result_messages[1]["role"] == "user"
    assert result_messages[1]["content"] == "What is AI?"

    # Params merged + preserved
    assert result_params["model"] == "gpt-4o"
    assert result_params["temperature"] == 0.8
    assert result_params["max_tokens"] == 256
    assert result_params["api_key"] == "keep-me"


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_manager_pre_call_hook_ref_precedence(mock_client_class):
    """
    Precedence for selecting git ref:
      prompt_version (arg) > git_ref kwarg > manager's _ref_override > client's default
    Validate that the chosen ref gets passed down to client.get_file_content.
    """
    mock_client = MagicMock()

    # Return any minimal valid prompt; we just need the call path to succeed.
    mock_client.get_file_content.return_value = """---
model: gpt-4
---
User: {{q}}"""
    mock_client_class.return_value = mock_client

    config = {"project": "g/s/r", "access_token": "tkn"}

    # Set a manager-level default ref override
    manager = GitLabPromptManager(config, prompt_id=None, ref="manager-default")

    # 1) No prior load; call with prompt_version -> should win
    _msgs, _params = manager.pre_call_hook(
        user_id="u",
        messages=[],
        litellm_params={},
        prompt_id="p1",
        prompt_variables={"q": "hello"},
        prompt_version="explicit-sha",
    )
    # get_file_content called with ref="explicit-sha"
    mock_client.get_file_content.assert_any_call("p1.prompt", ref="explicit-sha")

    # 2) Use git_ref kwarg (when no prompt_version)
    _msgs, _params = manager.pre_call_hook(
        user_id="u",
        messages=[],
        litellm_params={},
        prompt_id="p2",
        prompt_variables={"q": "hello"},
        git_ref="per-call-branch",
    )
    mock_client.get_file_content.assert_any_call("p2.prompt", ref="per-call-branch")

    # 3) Neither prompt_version nor git_ref -> falls back to manager _ref_override
    _msgs, _params = manager.pre_call_hook(
        user_id="u",
        messages=[],
        litellm_params={},
        prompt_id="p3",
        prompt_variables={"q": "hello"},
    )
    mock_client.get_file_content.assert_any_call("p3.prompt", ref="manager-default")


# -----------------------------
# Listing & availability
# -----------------------------
@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_manager_list_templates_with_prompts_path(mock_client_class):
    mock_client = MagicMock()
    mock_client.list_files.return_value = [
        "prompts/chat/a.prompt",
        "prompts/chat/sub/b.prompt",
        "prompts/chat/ignore.txt",
    ]
    mock_client.get_file_content.return_value = "Hello"
    mock_client_class.return_value = mock_client

    config = {
        "project": "g/s/r",
        "access_token": "tkn",
        "prompts_path": "prompts/chat",
    }

    manager = GitLabPromptManager(config, prompt_id="a")

    # list_templates strips folder prefix + extension
    ids = manager.get_available_prompts()
    assert "a" in ids
    assert "sub/b" in ids
    assert all(not x.endswith(".prompt") for x in ids)
    assert all("/prompts/chat/" not in x for x in ids)


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_template_manager_load_all_prompts(mock_client_class):
    """load_all_prompts should fetch all .prompt files and populate the internal cache."""
    mock_client = MagicMock()
    mock_client.list_files.return_value = [
        "prompts/a.prompt",
        "prompts/sub/b.prompt",
    ]
    mock_client.get_file_content.side_effect = [
        "Hello {{x}}",        # for a.prompt
        "---\nmodel: gpt-4\n---\nUser: {{y}}",  # for b.prompt with frontmatter
    ]
    mock_client_class.return_value = mock_client

    config = {
        "project": "g/s/r",
        "access_token": "tkn",
        "prompts_path": "prompts",
    }

    pm = GitLabPromptManager(config).prompt_manager
    loaded = pm.load_all_prompts()
    assert set(loaded) == {"a", "sub/b"}
    assert "a" in pm.prompts and "sub/b" in pm.prompts


# -----------------------------
# post_call & integration name
# -----------------------------
def test_gitlab_prompt_manager_integration_name():
    config = {"project": "g/s/r", "access_token": "tkn"}
    manager = GitLabPromptManager(config)
    assert manager.integration_name == "gitlab"


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_manager_post_call_hook_passthrough(mock_client_class):
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = "User: {{m}}"
    mock_client_class.return_value = mock_client

    config = {"project": "g/s/r", "access_token": "tkn"}

    manager = GitLabPromptManager(config, prompt_id="p")

    dummy_response = MagicMock()
    out = manager.post_call_hook(
        user_id="u",
        response=dummy_response,
        input_messages=[{"role": "user", "content": "x"}],
        litellm_params={},
        prompt_id="p",
    )
    assert out is dummy_response

@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_version_precedence_prompt_version_wins(mock_client_class):
    """
    prompt_version > git_ref kwarg > manager _ref_override.
    Ensure prompt_version wins and is passed down to GitLabClient.get_file_content.
    """
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = """---
model: gpt-4
---
User: {{q}}"""
    mock_client_class.return_value = mock_client

    cfg = {"project": "g/s/r", "access_token": "tkn"}

    # Manager with a default override ref
    mgr = GitLabPromptManager(cfg, ref="manager-default")

    # Provide both git_ref kwarg and prompt_version, the latter should win
    msgs, params = mgr.pre_call_hook(
        user_id="u",
        messages=[],
        litellm_params={},
        prompt_id="promptA",
        prompt_variables={"q": "hello"},
        prompt_version="sha-111",         # highest precedence
        git_ref="feature/branch-xyz",     # should be ignored because prompt_version provided
    )

    mock_client.get_file_content.assert_any_call("promptA.prompt", ref="sha-111")
    # sanity — prompt parsed and params returned
    assert any(m["role"] == "user" for m in msgs)
    assert params.get("model") == "gpt-4"


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_version_ref_kwarg_used_when_no_prompt_version(mock_client_class):
    """
    If prompt_version is omitted, git_ref kwarg should be used.
    """
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = "User: {{q}}"
    mock_client_class.return_value = mock_client

    cfg = {"project": "g/s/r", "access_token": "tkn"}
    mgr = GitLabPromptManager(cfg, ref="fallback-manager-ref")

    _msgs, _params = mgr.pre_call_hook(
        user_id="u",
        messages=[],
        litellm_params={},
        prompt_id="promptB",
        prompt_variables={"q": "hi"},
        git_ref="hotfix/ref-2",   # used since prompt_version not provided
    )

    mock_client.get_file_content.assert_any_call("promptB.prompt", ref="hotfix/ref-2")


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_version_manager_override_used_when_no_prompt_version_or_kwarg(mock_client_class):
    """
    If neither prompt_version nor git_ref is supplied, fall back to manager-level ref override.
    """
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = "User: {{q}}"
    mock_client_class.return_value = mock_client

    cfg = {"project": "g/s/r", "access_token": "tkn"}
    mgr = GitLabPromptManager(cfg, ref="manager-override-ref")

    _msgs, _params = mgr.pre_call_hook(
        user_id="u",
        messages=[],
        litellm_params={},
        prompt_id="promptC",
        prompt_variables={"q": "hey"},
    )

    mock_client.get_file_content.assert_any_call("promptC.prompt", ref="manager-override-ref")


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_get_prompt_template_explicit_ref_param(mock_client_class):
    """
    Directly calling get_prompt_template(ref=...) should pass that ref to GitLabClient.
    """
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = """---
model: gpt-4o
---
User: {{x}}"""
    mock_client_class.return_value = mock_client

    cfg = {"project": "g/s/r", "access_token": "tkn"}
    mgr = GitLabPromptManager(cfg)

    rendered, metadata = mgr.get_prompt_template(
        prompt_id="promptD",
        prompt_variables={"x": "value"},
        ref="v1.2.3",  # explicit tag
    )
    mock_client.get_file_content.assert_any_call("promptD.prompt", ref="v1.2.3")
    assert "value" in rendered
    assert metadata.get("model") == "gpt-4o"


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_version_with_prompts_path(mock_client_class):
    """
    Ensure prompts_path + prompt_version work together (path resolution + ref).
    """
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = "User: {{q}}"
    mock_client_class.return_value = mock_client

    cfg = {
        "project": "g/s/r",
        "access_token": "tkn",
        "prompts_path": "prompts/chat",
    }
    mgr = GitLabPromptManager(cfg)

    _msgs, _params = mgr.pre_call_hook(
        user_id="u",
        messages=[],
        litellm_params={},
        prompt_id="folder/sub/my_prompt",
        prompt_variables={"q": "ok"},
        prompt_version="commit-sha-999",
    )

    # Path should include prompts_path and end with .prompt
    mock_client.get_file_content.assert_any_call(
        "prompts/chat/folder/sub/my_prompt.prompt", ref="commit-sha-999"
    )
