import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

from litellm.integrations.gitlab.gitlab_client import GitLabClient
from litellm.integrations.gitlab.gitlab_prompt_manager import (
    GitLabPromptManager,
    GitLabPromptTemplate,
    GitLabTemplateManager,
    GitLabPromptCache,
    encode_prompt_id,
    decode_prompt_id,
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




# ---------------------------------------------------------------------
# ID Encoding/Decoding helpers
# ---------------------------------------------------------------------

def test_encode_decode_prompt_id_roundtrip():
    raw = "invoice/extract"
    encoded = encode_prompt_id(raw)
    assert encoded == "gitlab::invoice::extract"
    assert decode_prompt_id(encoded) == raw

def test_encode_prompt_id_already_encoded():
    encoded = "gitlab::test::path"
    assert encode_prompt_id(encoded) == encoded


# ---------------------------------------------------------------------
# GitLabTemplateManager behavior
# ---------------------------------------------------------------------

@pytest.fixture
def mock_gitlab_client():
    client = MagicMock()
    client.get_file_content.return_value = """---
model: bedrock/anthropic.claude-3-sonnet
temperature: 0.3
max_tokens: 100
---
system: You are a helpful bot.
user: Hello {{ name }}
"""
    client.list_files.return_value = [
        "prompts/chat/hello.prompt",
        "prompts/chat/nested/sub.prompt",
    ]
    return client


@pytest.fixture
def manager(mock_gitlab_client):
    cfg = {
        "project": "group/repo",
        "access_token": "token",
        "prompts_path": "prompts/chat",
    }
    return GitLabTemplateManager(gitlab_config=cfg, gitlab_client=mock_gitlab_client)


def test_list_templates_returns_encoded_ids(manager):
    ids = manager.list_templates()
    assert all(id.startswith("gitlab::") for id in ids)
    assert "gitlab::hello" in ids
    assert "gitlab::nested::sub" in ids


def test_load_prompt_from_gitlab_parses_metadata(manager, mock_gitlab_client):
    manager._load_prompt_from_gitlab("gitlab::hello")
    assert "gitlab::hello" in manager.prompts

    tmpl = manager.prompts["gitlab::hello"]
    assert isinstance(tmpl, GitLabPromptTemplate)
    assert tmpl.metadata["model"].startswith("bedrock/")
    assert "You are a helpful bot." in tmpl.content


def test_render_template_renders_jinja(manager, mock_gitlab_client):
    manager._load_prompt_from_gitlab("gitlab::hello")
    output = manager.render_template("gitlab::hello", {"name": "Prishu"})
    assert "Hello Prishu" in output


def test_get_template_returns_none_if_not_loaded(manager):
    assert manager.get_template("gitlab::missing") is None


def test_repo_path_conversion(manager):
    raw = "gitlab::nested::sub"
    repo_path = manager._id_to_repo_path(raw)
    assert repo_path.endswith("nested/sub.prompt")
    # Ensure decode/encode reversibility
    decoded = manager._repo_path_to_id(repo_path)
    assert decoded == raw


# ---------------------------------------------------------------------
# GitLabPromptManager high-level integration
# ---------------------------------------------------------------------

@pytest.fixture
def prompt_manager(mock_gitlab_client):
    cfg = {"project": "group/repo", "access_token": "tkn", "prompts_path": "prompts/chat"}
    return GitLabPromptManager(gitlab_config=cfg, gitlab_client=mock_gitlab_client)


def test_get_prompt_template_renders_content(prompt_manager):
    encoded_id = "gitlab::hello"
    content, meta = prompt_manager.get_prompt_template(encoded_id, {"name": "World"})
    assert "Hello World" in content
    assert "model" in meta


def test_pre_call_hook_parses_roles(prompt_manager):
    prompt_id = "gitlab::hello"
    messages, params = prompt_manager.pre_call_hook(
        user_id="user123",
        messages=[],
        prompt_id=prompt_id,
        prompt_variables={"name": "Tester"},
    )
    assert isinstance(messages, list)
    roles = [m["role"] for m in messages]
    assert "system" in roles and "user" in roles
    assert "model" in params


def test_get_available_prompts_returns_sorted(prompt_manager):
    ids = prompt_manager.get_available_prompts()
    assert any(id.startswith("gitlab::") for id in ids)
    assert ids == sorted(ids)


# ---------------------------------------------------------------------
# GitLabPromptCache behavior
# ---------------------------------------------------------------------

@pytest.fixture
def prompt_cache(mock_gitlab_client):
    cfg = {"project": "group/repo", "access_token": "tkn", "prompts_path": "prompts/chat"}
    return GitLabPromptCache(cfg, gitlab_client=mock_gitlab_client)


def test_cache_load_all_builds_internal_maps(prompt_cache):
    result = prompt_cache.load_all()
    assert isinstance(result, dict)
    # check encoded key presence
    assert any(k.startswith("gitlab::") for k in result)
    assert prompt_cache.list_files()
    assert prompt_cache.list_ids()


def test_cache_get_by_id_handles_encoded_and_decoded(prompt_cache):
    prompt_cache.load_all()
    encoded = "gitlab::hello"
    decoded = decode_prompt_id(encoded)
    assert prompt_cache.get_by_id(encoded)
    assert prompt_cache.get_by_id(decoded)


def test_cache_reload_resets_and_reloads(prompt_cache):
    prompt_cache.load_all()
    before = set(prompt_cache.list_ids())
    prompt_cache.reload()
    after = set(prompt_cache.list_ids())
    assert before == after


# -----------------------
# Test fakes / fixtures
# -----------------------

class FakeTemplateManager:
    """
    Minimal stand-in for GitLabTemplateManager that GitLabPromptCache expects.
    """
    def __init__(self, prompts_path="prompts"):
        # simulate a configured prompts folder (affects _id_to_repo_path)
        self.prompts_path = prompts_path.strip("/")
        self.prompts = {}  # id -> GitLabPromptTemplate

        # Seeds used by list_templates()
        self._discoverable_ids = []

    # Methods used by GitLabPromptCache.load_all
    def list_templates(self, *, recursive: bool = True):
        return list(self._discoverable_ids)

    def _load_prompt_from_gitlab(self, pid, ref=None):
        # Pretend we fetched and parsed a file; add a basic template if not present
        if pid not in self.prompts:
            self.prompts[pid] = GitLabPromptTemplate(
                template_id=pid,
                content=f"User: Hello from {pid}",
                metadata={"model": "gpt-4", "temperature": 0.1},
            )

    def get_template(self, pid):
        return self.prompts.get(pid)

    def _id_to_repo_path(self, pid):
        base = f"{self.prompts_path}/" if self.prompts_path else ""
        return f"{base}{pid}.prompt"


class FakePromptManagerWrapper:
    """
    Minimal wrapper to mimic GitLabPromptManager(prompt_manager=<GitLabTemplateManager>).
    GitLabPromptCache.__init__ expects GitLabPromptManager(...).prompt_manager.
    """
    def __init__(self, fake_tm):
        self.prompt_manager = fake_tm


@pytest.fixture()
def fake_managers():
    """
    Provide a fresh FakeTemplateManager plus a wrapper for each test.
    """
    tm = FakeTemplateManager(prompts_path="prompts/chat")
    wrapper = FakePromptManagerWrapper(tm)
    return tm, wrapper


# -----------------------
# Tests
# -----------------------

@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabPromptManager")
def test_cache_load_all_encodes_ids_and_populates_maps(mock_pm_cls, fake_managers):
    tm, wrapper = fake_managers
    # Simulate two files discovered under prompts_path
    tm._discoverable_ids = ["a", "sub/b"]

    # When GitLabPromptCache constructs GitLabPromptManager(...), return our wrapper
    mock_pm_cls.return_value = wrapper

    cache = GitLabPromptCache({"project": "g/s/r", "access_token": "tkn"})
    result = cache.load_all()

    # Encoded keys are present
    assert set(result.keys()) == {encode_prompt_id("a"), encode_prompt_id("sub/b")}

    # Files map built with full repo paths
    expect_a_path = tm._id_to_repo_path("a")
    expect_b_path = tm._id_to_repo_path("sub/b")
    assert cache.list_files() == [expect_a_path, expect_b_path]

    # IDs list is the encoded IDs
    assert set(cache.list_ids()) == {encode_prompt_id("a"), encode_prompt_id("sub/b")}

    # Stored entries have normalized json shape
    a_entry = cache.get_by_id("gitlab::a")
    assert a_entry["id"] == "a"  # id is the raw (decoded) id in the entry body
    assert a_entry["path"] == expect_a_path
    assert a_entry["metadata"]["model"] == "gpt-4"


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabPromptManager")
def test_cache_get_by_id_accepts_encoded_and_decoded(mock_pm_cls, fake_managers):
    tm, wrapper = fake_managers
    tm._discoverable_ids = ["x/y"]
    mock_pm_cls.return_value = wrapper

    cache = GitLabPromptCache({"project": "g/s/r", "access_token": "tkn"})
    cache.load_all()

    # Encoded lookup
    encoded = encode_prompt_id("x/y")
    decoded = "x/y"

    by_encoded = cache.get_by_id(encoded)
    by_decoded = cache.get_by_id(decoded)

    assert by_encoded is not None
    assert by_decoded is not None
    assert by_encoded == by_decoded  # normalization works
    # sanity on shape
    assert by_encoded["id"] == "x/y"
    assert by_encoded["path"].endswith("prompts/chat/x/y.prompt")


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabPromptManager")
def test_cache_reload_clears_then_reloads(mock_pm_cls, fake_managers):
    tm, wrapper = fake_managers
    tm._discoverable_ids = ["p1"]
    mock_pm_cls.return_value = wrapper

    cache = GitLabPromptCache({"project": "g/s/r", "access_token": "tkn"})
    first = cache.load_all()
    assert encode_prompt_id("p1") in first

    # Change discovered ids and ensure reload reflects the change
    tm._discoverable_ids = ["p2"]
    reloaded = cache.reload()

    assert encode_prompt_id("p1") not in reloaded
    assert encode_prompt_id("p2") in reloaded
    # internal maps should reflect only new state
    assert cache.list_ids() == [encode_prompt_id("p2")]


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabPromptManager")
def test_cache_skips_when_template_missing_even_after_reload_attempt(mock_pm_cls, fake_managers):
    """
    If get_template(pid) returns None even after a retry load, the entry is skipped.
    """
    class MissingTemplateManager(FakeTemplateManager):
        def get_template(self, pid):
            # Always return None to trigger the continue path
            return None

        def _load_prompt_from_gitlab(self, pid, ref=None):
            # Pretend to load, but still don't populate prompts so get_template stays None
            pass

    tm = MissingTemplateManager(prompts_path="prompts")
    wrapper = FakePromptManagerWrapper(tm)
    mock_pm_cls.return_value = wrapper

    cache = GitLabPromptCache({"project": "g/s/r", "access_token": "tkn"})
    tm._discoverable_ids = ["will/vanish"]
    out = cache.load_all()

    assert out == {}
    assert cache.list_files() == []
    assert cache.list_ids() == []


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabPromptManager")
def test_cache_get_by_file_returns_exact_entry(mock_pm_cls, fake_managers):
    tm, wrapper = fake_managers
    tm._discoverable_ids = ["alpha", "nested/beta"]
    mock_pm_cls.return_value = wrapper

    cache = GitLabPromptCache({"project": "g/s/r", "access_token": "tkn"})
    cache.load_all()

    alpha_path = tm._id_to_repo_path("alpha")
    beta_path = tm._id_to_repo_path("nested/beta")

    alpha = cache.get_by_file(alpha_path)
    beta = cache.get_by_file(beta_path)

    assert alpha and alpha["id"] == "alpha"
    assert beta and beta["id"] == "nested/beta"


@patch("litellm.integrations.gitlab.gitlab_prompt_manager.GitLabPromptManager")
def test_encode_decode_helpers_roundtrip_in_cache_context(mock_pm_cls, fake_managers):
    tm, wrapper = fake_managers
    tm._discoverable_ids = ["dir1/dir2/item"]
    mock_pm_cls.return_value = wrapper

    cache = GitLabPromptCache({"project": "g/s/r", "access_token": "tkn"})
    cache.load_all()

    encoded = encode_prompt_id("dir1/dir2/item")
    assert encoded in cache.list_ids()

    # decode → encode → lookup should still work
    decoded = decode_prompt_id(encoded)
    assert decoded == "dir1/dir2/item"

    got = cache.get_by_id(decoded)
    assert got is not None
    assert got["id"] == "dir1/dir2/item"