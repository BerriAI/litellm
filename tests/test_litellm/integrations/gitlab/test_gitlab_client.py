import base64
import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.integrations.gitlab.gitlab_client import GitLabClient


# -----------------------------
# Test doubles for HTTP layer
# -----------------------------
class HTTPError(Exception):
    def __init__(self, msg, response=None):
        super().__init__(msg)
        self.response = response


class FakeResponse:
    def __init__(self, *, status_code=200, headers=None, text="", content=b"", json_data=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self.content = content if content else text.encode("utf-8")
        self._json_data = json_data

    def json(self):
        if self._json_data is not None:
            return self._json_data
        try:
            return json.loads(self.text)
        except Exception:
            raise ValueError("Invalid JSON")

    def raise_for_status(self):
        if 400 <= self.status_code:
            raise HTTPError(f"HTTP {self.status_code}", response=self)


class StubHTTPHandler:
    """
    Minimal stub that returns a FakeResponse based on url.
    Configure behavior by customizing self.routes in each test.
    """
    def __init__(self):
        self.routes = {}  # url -> FakeResponse or Exception
        self.calls = []   # [(method, url, headers)]

    def get(self, url, headers=None):
        self.calls.append(("GET", url, headers or {}))
        resp_or_exc = self.routes.get(url)
        if isinstance(resp_or_exc, Exception):
            raise resp_or_exc
        if resp_or_exc is None:
            # default: 404 not found
            return FakeResponse(status_code=404, headers={"content-type": "application/json"}, text="{}")
        return resp_or_exc

    def close(self):
        pass


# -----------------------------
# Fixtures / helpers
# -----------------------------
def make_client(**overrides):
    cfg = {
        "project": "group/sub/repo",
        "access_token": "glpat_xxx",
        "branch": "develop",
        "base_url": "https://gitlab.example.com/api/v4",
    }
    cfg.update(overrides)
    client = GitLabClient(cfg)
    # swap in stub http handler
    client.http_handler = StubHTTPHandler()
    return client


def enc_project(p):  # how client encodes project in urls
    return p.replace("/", "%2F")


# -----------------------------
# Constructor / config tests
# -----------------------------
def test_init_requires_project_and_token():
    with pytest.raises(ValueError):
        GitLabClient({"project": "p"})
    with pytest.raises(ValueError):
        GitLabClient({"access_token": "t"})


def test_ref_prefers_tag_over_branch():
    c = make_client(tag="v1.2.3", branch="main")
    assert c.ref == "v1.2.3"


def test_default_branch_is_main_when_absent():
    c = make_client(branch=None)  # explicit None
    assert c.ref == 'main'


def test_auth_header_token_default():
    c = make_client()
    assert c.headers.get("Private-Token") == "glpat_xxx"
    assert "Authorization" not in c.headers


def test_auth_header_oauth():
    c = make_client(auth_method="oauth")
    assert c.headers.get("Authorization") == "Bearer glpat_xxx"
    assert "Private-Token" not in c.headers


def test_set_ref_updates_effective_ref():
    c = make_client(branch="main")
    c.set_ref("feature/x")
    assert c.ref == "feature/x"
    with pytest.raises(ValueError):
        c.set_ref("")


# -----------------------------
# get_file_content
# -----------------------------
def test_get_file_content_raw_text_success():
    c = make_client(tag="release-1")
    raw_url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}/repository/files/path%2Fto%2Ffile.prompt/raw?ref=release-1"
    c.http_handler.routes[raw_url] = FakeResponse(
        status_code=200,
        headers={"content-type": "text/plain; charset=utf-8"},
        text="Hello world"
    )
    out = c.get_file_content("path/to/file.prompt")
    assert out == "Hello world"
    # ensure it used the expected URL
    assert c.http_handler.calls[-1][1] == raw_url


def test_get_file_content_raw_binary_utf8_decodes():
    c = make_client(branch="main")
    raw_url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}/repository/files/bin%2Ffile.raw/raw?ref=main"
    c.http_handler.routes[raw_url] = FakeResponse(
        status_code=200,
        headers={"content-type": "application/octet-stream"},
        content="προμ pt".encode("utf-8")
    )
    out = c.get_file_content("bin/file.raw")
    assert out == "προμ pt"


def test_get_file_content_fallbacks_to_json_when_raw_404_and_decodes_base64():
    c = make_client(branch="main")
    raw_url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}/repository/files/prompts%2Ffoo.prompt/raw?ref=main"
    json_url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}/repository/files/prompts%2Ffoo.prompt?ref=main"

    c.http_handler.routes[raw_url] = FakeResponse(status_code=404, headers={"content-type": "application/json"}, text="{}")
    encoded = base64.b64encode("FROM JSON".encode("utf-8")).decode("ascii")
    c.http_handler.routes[json_url] = FakeResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        json_data={"content": encoded, "encoding": "base64"}
    )

    out = c.get_file_content("prompts/foo.prompt")
    assert out == "FROM JSON"


def test_get_file_content_returns_none_on_404_everywhere():
    c = make_client(branch="main")
    raw_url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}/repository/files/ghost%2Fmissing.prompt/raw?ref=main"
    json_url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}/repository/files/ghost%2Fmissing.prompt?ref=main"
    c.http_handler.routes[raw_url] = FakeResponse(status_code=404)
    c.http_handler.routes[json_url] = FakeResponse(status_code=404)
    assert c.get_file_content("ghost/missing.prompt") is None


def test_get_file_content_permission_errors_are_mapped():
    c = make_client(branch="main")
    raw_url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}/repository/files/secure%2Ffile.prompt/raw?ref=main"
    # raise_for_status will be called, so return 403 response (not an exception from transport)
    c.http_handler.routes[raw_url] = FakeResponse(status_code=403)
    with pytest.raises(Exception) as ei:
        c.get_file_content("secure/file.prompt")
    assert "Access denied" in str(ei.value)

    c.http_handler.routes[raw_url] = FakeResponse(status_code=401)
    with pytest.raises(Exception) as ei2:
        c.get_file_content("secure/file.prompt")
    assert "Authentication failed" in str(ei2.value)


# -----------------------------
# list_files
# -----------------------------
def test_list_files_filters_by_extension_and_handles_recursive_flag():
    c = make_client(branch="dev")
    tree_url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}/repository/tree?ref=dev&path=prompts&recursive=true"
    c.http_handler.routes[tree_url] = FakeResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        json_data=[
            {"type": "blob", "path": "prompts/a.prompt"},
            {"type": "blob", "path": "prompts/b.txt"},
            {"type": "blob", "path": "prompts/sub/c.prompt"},
            {"type": "tree", "path": "prompts/sub"},
        ],
    )
    files = c.list_files("prompts", ".prompt", recursive=True)
    assert files == ["prompts/a.prompt", "prompts/sub/c.prompt"]


def test_list_files_404_returns_empty_list():
    c = make_client()
    tree_url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}/repository/tree?ref=develop&path=does%20not%20exist"
    c.http_handler.routes[tree_url] = FakeResponse(status_code=404)
    out = c.list_files("does not exist", ".prompt", recursive=False)
    assert out == []


def test_list_files_allows_ref_override():
    c = make_client(branch="main")
    url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}/repository/tree?ref=v2&path=prompts"
    c.http_handler.routes[url] = FakeResponse(status_code=200, json_data=[])
    out = c.list_files("prompts", ".prompt", ref="v2")
    assert out == []
    # verify correct URL used
    assert c.http_handler.calls[-1][1] == url


# -----------------------------
# repo info / branches / metadata / connection
# -----------------------------
def test_get_repository_info_success():
    c = make_client()
    url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}"
    c.http_handler.routes[url] = FakeResponse(status_code=200, json_data={"id": 123})
    info = c.get_repository_info()
    assert info["id"] == 123


def test_test_connection_true_and_false():
    c = make_client()
    ok_url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}"
    c.http_handler.routes[ok_url] = FakeResponse(status_code=200, json_data={"id": 1})
    assert c.test_connection() is True

    # make it fail next time
    c.http_handler.routes[ok_url] = FakeResponse(status_code=500)
    assert c.test_connection() is False


def test_get_branches_returns_list():
    c = make_client()
    url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}/repository/branches"
    c.http_handler.routes[url] = FakeResponse(status_code=200, json_data=[{"name": "main"}])
    branches = c.get_branches()
    assert isinstance(branches, list)
    assert branches[0]["name"] == "main"


def test_get_file_metadata_parses_headers_and_handles_404():
    c = make_client(branch="x")
    raw_url = f"https://gitlab.example.com/api/v4/projects/{enc_project('group/sub/repo')}/repository/files/foo%2Fbar.raw/raw?ref=x"
    c.http_handler.routes[raw_url] = FakeResponse(
        status_code=200,
        headers={"content-type": "application/octet-stream", "content-length": "1234", "last-modified": "Thu, 01 Jan 1970 00:00:00 GMT"},
        content=b"\x00"
    )
    meta = c.get_file_metadata("foo/bar.raw")
    assert meta["content_type"] == "application/octet-stream"
    assert meta["content_length"] == "1234"

    c.http_handler.routes[raw_url] = FakeResponse(status_code=404)
    assert c.get_file_metadata("foo/bar.raw") is None
