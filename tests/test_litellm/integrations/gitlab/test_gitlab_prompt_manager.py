import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

from litellm.integrations.gitlab.gitlab_client import GitLabClient
from litellm.integrations.gitlab.gitlab_prompt_manager import (
    GitLabPromptManager,
    GitLabPromptTemplate,
)

# -----------------------
# GitLabPromptTemplate
# -----------------------

def test_gitlab_prompt_template_creation():
    """Test GitLabPromptTemplate creation and metadata extraction."""
    metadata = {
        "model": "gpt-4",
        "temperature": 0.7,
        "input": {"schema": {"text": "string"}},
        "output": {"format": "json"},
    }

    template = GitLabPromptTemplate(
        template_id="test_template",
        content="Hello {{name}}!",
        metadata=metadata,
    )

    assert template.template_id == "test_template"
    assert template.content == "Hello {{name}}!"
    assert template.model == "gpt-4"
    assert template.optional_params["temperature"] == 0.7
    assert template.input_schema == {"text": "string"}


# -----------------------
# GitLabClient init & validation
# -----------------------

def test_gitlab_client_initialization_token_vs_oauth():
    """Test GitLabClient initialization with token and oauth auth methods."""
    # token (default)
    config_token = {
        "project": "group/sub/repo",
        "access_token": "glpat-XYZ",
        "branch": "main",
    }
    client = GitLabClient(config_token)
    assert client.project == "group/sub/repo"
    assert client.access_token == "glpat-XYZ"
    assert client.branch == "main"
    assert client.auth_method == "token"
    # token header is used
    assert client.headers.get("Private-Token") == "glpat-XYZ"
    assert "Authorization" not in client.headers

    # oauth
    config_oauth = {
        "project": 123456,  # numeric project id supported
        "access_token": "oauth-bearer",
        "auth_method": "oauth",
    }
    client_oauth = GitLabClient(config_oauth)
    assert client_oauth.auth_method == "oauth"
    assert client_oauth.headers.get("Authorization") == "Bearer oauth-bearer"
    assert "Private-Token" not in client_oauth.headers


def test_gitlab_client_missing_required_fields():
    """Test GitLabClient initialization with missing required fields."""
    with pytest.raises(ValueError, match="project and access_token are required"):
        GitLabClient({"project": "group/x/repo"})
    with pytest.raises(ValueError, match="project and access_token are required"):
        GitLabClient({"access_token": "tok"})


# -----------------------
# GitLabClient: get_file_content
# -----------------------

@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get")
def test_gitlab_client_get_file_content_raw_success(mock_get):
    """Successful file content retrieval via RAW endpoint."""
    mock_response = MagicMock()
    mock_response.text = "file content"
    mock_response.content = b"file content"
    mock_response.headers = {"content-type": "text/plain"}
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    client = GitLabClient({"project": "g/s/r", "access_token": "tok"})
    content = client.get_file_content("prompts/test.prompt")
    assert content == "file content"
    mock_get.assert_called_once()


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get")
def test_gitlab_client_get_file_content_raw_404_fallback_json_base64(mock_get):
    """When RAW returns 404, fallback to JSON endpoint and decode base64 content."""
    import base64

    # First RAW 404
    resp_raw = MagicMock()
    resp_raw.status_code = 404
    resp_raw.raise_for_status.side_effect = Exception()
    mock_get.side_effect = [resp_raw]

    # Then JSON OK
    resp_json = MagicMock()
    encoded = base64.b64encode(b"json-content").decode("utf-8")
    resp_json.json.return_value = {"content": encoded, "encoding": "base64"}
    resp_json.status_code = 200
    resp_json.raise_for_status.return_value = None

    # We need mock_get to return JSON response second time; easiest: reset side_effect to list of returns
    def side_effect(url, headers):
        if "/raw?" in url:
            return resp_raw
        else:
            return resp_json

    mock_get.side_effect = side_effect

    client = GitLabClient({"project": "g/s/r", "access_token": "tok"})
    content = client.get_file_content("prompts/test.prompt")
    assert content == "json-content"


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get")
def test_gitlab_client_get_file_content_not_found(mock_get):
    """File not found returns None."""
    # Simulate RAW 404 and JSON 404
    resp_404 = MagicMock()
    resp_404.status_code = 404
    resp_404.raise_for_status.side_effect = Exception()
    def side_effect(url, headers):
        return resp_404
    mock_get.side_effect = side_effect

    client = GitLabClient({"project": "g/s/r", "access_token": "tok"})
    content = client.get_file_content("missing.prompt")
    assert content is None


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get")
def test_gitlab_client_get_file_content_access_denied(mock_get):
    """403 raises a helpful message."""
    import httpx
    resp = MagicMock()
    resp.status_code = 403
    # raise_for_status inside client only called on non-404 success path;
    # simulate exception path by making the request itself raise an httpx error wrapper
    err = httpx.HTTPStatusError("403", request=MagicMock(), response=resp)
    mock_get.side_effect = err

    client = GitLabClient({"project": "g/s/r", "access_token": "tok"})
    with pytest.raises(Exception, match="Access denied to file 'test.prompt'"):
        client.get_file_content("test.prompt")


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get")
def test_gitlab_client_get_file_content_auth_failed(mock_get):
    """401 raises auth error."""
    import httpx
    resp = MagicMock()
    resp.status_code = 401
    err = httpx.HTTPStatusError("401", request=MagicMock(), response=resp)
    mock_get.side_effect = err

    client = GitLabClient({"project": "g/s/r", "access_token": "tok"})
    with pytest.raises(Exception, match="Authentication failed"):
        client.get_file_content("test.prompt")


# -----------------------
# GitLabClient: list_files
# -----------------------

@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get")
def test_gitlab_client_list_files_success(mock_get):
    """List .prompt files via repository tree API."""
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"type": "blob", "path": "prompts/test1.prompt"},
        {"type": "blob", "path": "prompts/test2.prompt"},
        {"type": "blob", "path": "prompts/other.txt"},
        {"type": "tree", "path": "prompts/subdir"},
    ]
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    client = GitLabClient({"project": "g/s/r", "access_token": "tok"})
    files = client.list_files("prompts", ".prompt", recursive=True)

    assert files == ["prompts/test1.prompt", "prompts/test2.prompt"]


# -----------------------
# GitLabTemplateManager: parsing & rendering
# -----------------------

def test_gitlab_prompt_manager_parse_prompt_file():
    """Parse .prompt with YAML frontmatter."""
    prompt_content = """---
model: gpt-4
temperature: 0.7
max_tokens: 150
input:
  schema:
    user_message: string
    system_context?: string
---

{% if system_context %}System: {{system_context}}

{% endif %}User: {{user_message}}"""

    manager = GitLabPromptManager({"project": "g/s/r", "access_token": "tok"})
    template = manager.prompt_manager._parse_prompt_file(prompt_content, "test_prompt")

    assert template.template_id == "test_prompt"
    assert template.model == "gpt-4"
    assert template.temperature == 0.7
    assert template.max_tokens == 150
    assert template.input_schema == {"user_message": "string", "system_context?": "string"}
    assert "{% if system_context %}" in template.content


def test_gitlab_prompt_manager_parse_prompt_file_no_frontmatter():
    """Parse .prompt without YAML frontmatter."""
    prompt_content = "Simple prompt: {{message}}"
    manager = GitLabPromptManager({"project": "g/s/r", "access_token": "tok"})
    template = manager.prompt_manager._parse_prompt_file(prompt_content, "simple_prompt")
    assert template.template_id == "simple_prompt"
    assert template.content == "Simple prompt: {{message}}"
    assert template.metadata == {}


def test_gitlab_prompt_manager_render_template_and_errors():
    """Render a stored template; error if missing."""
    manager = GitLabPromptManager({"project": "g/s/r", "access_token": "tok"})

    tpl = GitLabPromptTemplate(
        template_id="t1",
        content="Hello {{name}}! Welcome to {{place}}.",
        metadata={"model": "gpt-4"},
    )
    manager.prompt_manager.prompts["t1"] = tpl

    rendered = manager.prompt_manager.render_template("t1", {"name": "World", "place": "Earth"})
    assert rendered == "Hello World! Welcome to Earth."

    with pytest.raises(ValueError, match="Template 'nope' not found"):
        manager.prompt_manager.render_template("nope", {})


# -----------------------
# GitLabPromptManager: integration & behavior
# -----------------------

@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_manager_integration(mock_client_class):
    """Load prompt on init and render."""
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = """---
model: gpt-4
temperature: 0.7
---
Hello {{name}}!"""
    mock_client_class.return_value = mock_client

    mgr = GitLabPromptManager({"project": "g/s/r", "access_token": "tok"}, prompt_id="test_prompt")
    assert "test_prompt" in mgr.prompt_manager.prompts

    template = mgr.prompt_manager.prompts["test_prompt"]
    assert template.model == "gpt-4"
    assert template.temperature == 0.7

    rendered = mgr.prompt_manager.render_template("test_prompt", {"name": "World"})
    assert rendered == "Hello World!"


def test_gitlab_prompt_manager_parse_prompt_to_messages():
    """Parse prompt content into chat messages."""
    mgr = GitLabPromptManager({"project": "g/s/r", "access_token": "tok"})

    # single user msg
    simple = "Hello there!"
    msgs = mgr._parse_prompt_to_messages(simple)
    assert msgs == [{"role": "user", "content": "Hello there!"}]

    # multi-role
    multi = """System: You are helpful.

User: Hi?

Assistant: Hello!"""
    msgs = mgr._parse_prompt_to_messages(multi)
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "You are helpful."
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == "Hi?"
    assert msgs[2]["role"] == "assistant" and msgs[2]["content"] == "Hello!"


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_manager_pre_call_hook_basic(mock_client_class):
    """Pre-call hook parses messages and injects params."""
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = """---
model: gpt-4
temperature: 0.7
---
System: You are helpful.

User: {{q}}"""
    mock_client_class.return_value = mock_client

    mgr = GitLabPromptManager({"project": "g/s/r", "access_token": "tok"}, prompt_id="p1")

    original = [{"role": "user", "content": "ignored"}]
    msgs, params = mgr.pre_call_hook(
        user_id="u",
        messages=original,
        litellm_params={},
        prompt_id="p1",
        prompt_variables={"q": "What is AI?"},
    )

    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == "What is AI?"
    assert params["model"] == "gpt-4" and params["temperature"] == 0.7


def test_gitlab_prompt_manager_pre_call_hook_no_prompt_id():
    """If no prompt_id provided, messages/params unchanged."""
    mgr = GitLabPromptManager({"project": "g/s/r", "access_token": "tok"})
    original = [{"role": "user", "content": "Hello"}]
    msgs, params = mgr.pre_call_hook(user_id="u", messages=original, litellm_params={}, prompt_id=None)
    assert msgs == original and params == {}


def test_gitlab_prompt_manager_get_available_prompts():
    """Return keys of stored templates."""
    mgr = GitLabPromptManager({"project": "g/s/r", "access_token": "tok"})
    mgr.prompt_manager.prompts.update({
        "p1": GitLabPromptTemplate("p1", "c1", {}),
        "p2": GitLabPromptTemplate("p2", "c2", {}),
    })
    assert set(mgr.get_available_prompts()) == {"p1", "p2"}


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_manager_reload_prompts(mock_client_class):
    """Ensure reload resets and re-inits manager."""
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = """---
model: gpt-4
---
Hello {{x}}"""
    mock_client_class.return_value = mock_client

    mgr = GitLabPromptManager({"project": "g/s/r", "access_token": "tok"}, prompt_id="t0")
    assert "t0" in mgr.prompt_manager.prompts

    # force reset
    with patch.object(mgr, "_prompt_manager", None):
        mgr.reload_prompts()
        _ = mgr.prompt_manager
        # No assertion beyond not raising and property access works


# -----------------------
# YAML fallback parsing
# -----------------------

def test_gitlab_prompt_manager_yaml_parsing_fallback_and_types():
    mgr = GitLabPromptManager({"project": "g/s/r", "access_token": "tok"})
    yaml_content = """model: gpt-4
temperature: 0.7
max_tokens: 150
enabled: true
disabled: false
count: 42
rate: 0.5"""
    parsed = mgr.prompt_manager._parse_yaml_basic(yaml_content)
    assert parsed["model"] == "gpt-4"
    assert parsed["temperature"] == 0.7
    assert parsed["max_tokens"] == 150
    assert parsed["enabled"] is True
    assert parsed["disabled"] is False
    assert parsed["count"] == 42
    assert parsed["rate"] == 0.5


# -----------------------
# prompts_path handling + prompt_version (ref) precedence
# -----------------------

@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_manager_prompts_path_resolution_and_version(mock_client_class):
    """prompts_path + explicit prompt_version should produce correct repo path and ref."""
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = "User: {{q}}"
    mock_client_class.return_value = mock_client

    cfg = {
        "project": "g/s/r",
        "access_token": "tok",
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

    mock_client.get_file_content.assert_any_call(
        "prompts/chat/folder/sub/my_prompt.prompt", ref="commit-sha-999"
    )


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient")
def test_gitlab_prompt_manager_version_precedence(mock_client_class):
    """
    prompt_version > git_ref kwarg > manager _ref_override.
    """
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = "User: {{q}}"
    mock_client_class.return_value = mock_client

    mgr = GitLabPromptManager({"project": "g/s/r", "access_token": "tok"}, ref="manager-default")

    # prompt_version wins over git_ref kwarg
    _msgs, _params = mgr.pre_call_hook(
        user_id="u",
        messages=[],
        litellm_params={},
        prompt_id="pA",
        prompt_variables={"q": "hello"},
        prompt_version="sha-111",
        git_ref="feature/branch-xyz",
    )
    mock_client.get_file_content.assert_any_call("pA.prompt", ref="sha-111")

    # If no prompt_version, use git_ref kwarg
    _msgs, _params = mgr.pre_call_hook(
        user_id="u",
        messages=[],
        litellm_params={},
        prompt_id="pB",
        prompt_variables={"q": "hello"},
        git_ref="hotfix/ref-2",
    )
    mock_client.get_file_content.assert_any_call("pB.prompt", ref="hotfix/ref-2")

    # If neither provided, fall back to manager override
    _msgs, _params = mgr.pre_call_hook(
        user_id="u",
        messages=[],
        litellm_params={},
        prompt_id="pC",
        prompt_variables={"q": "hello"},
    )
    mock_client.get_file_content.assert_any_call("pC.prompt", ref="manager-default")
