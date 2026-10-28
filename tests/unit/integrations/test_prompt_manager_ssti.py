"""SSTI regression coverage for non-dotprompt prompt managers.

DotpromptManager was hardened to render through
``ImmutableSandboxedEnvironment``. The sibling managers (gitlab, arize,
bitbucket) ship the exact same attacker-controlled-template surface —
repository write access or workspace edit access turns into RCE on the
proxy host if the renderer is unsandboxed. This suite locks in the sandbox
so the regression can't recur.
"""

from unittest.mock import MagicMock

import pytest
from jinja2.exceptions import SecurityError
from jinja2.sandbox import ImmutableSandboxedEnvironment

from litellm.integrations.arize.arize_phoenix_prompt_manager import (
    ArizePhoenixTemplateManager,
)
from litellm.integrations.bitbucket.bitbucket_prompt_manager import (
    BitBucketTemplateManager,
)
from litellm.integrations.gitlab.gitlab_prompt_manager import GitLabTemplateManager

# Classic Jinja2 SSTI payloads. Any one of these rendering as anything other
# than the literal string (or raising) means the sandbox isn't engaged.
_SSTI_PAYLOADS = [
    "{{ ''.__class__.__mro__[1].__subclasses__() }}",
    "{{ config.__class__.__init__.__globals__['os'].popen('id').read() }}",
    "{{ cycler.__init__.__globals__.os.popen('id').read() }}",
    "{{ ().__class__.__bases__[0].__subclasses__() }}",
]


def _build_gitlab_manager() -> GitLabTemplateManager:
    # The constructor calls into a GitLab client when prompt_id is set; pass
    # None so __init__ stops at jinja_env construction and we can assert on it.
    return GitLabTemplateManager(
        gitlab_config={"project": "p", "access_token": "t", "branch": "main"},
        prompt_id=None,
        gitlab_client=MagicMock(),
    )


def _build_bitbucket_manager(monkeypatch) -> BitBucketTemplateManager:
    # Stub the BitBucket client so we don't need network or real config.
    from litellm.integrations.bitbucket import bitbucket_prompt_manager

    monkeypatch.setattr(
        bitbucket_prompt_manager, "BitBucketClient", lambda *a, **kw: MagicMock()
    )
    return BitBucketTemplateManager(
        bitbucket_config={"workspace": "w", "repository": "r", "access_token": "t"},
        prompt_id=None,
    )


def _build_arize_manager(monkeypatch) -> ArizePhoenixTemplateManager:
    from litellm.integrations.arize import arize_phoenix_prompt_manager

    monkeypatch.setattr(
        arize_phoenix_prompt_manager, "ArizePhoenixClient", lambda *a, **kw: MagicMock()
    )
    return ArizePhoenixTemplateManager(
        api_key="k",
        api_base="https://example.test",
        prompt_id=None,
    )


@pytest.mark.parametrize(
    "manager_factory",
    [
        ("gitlab", lambda mp: _build_gitlab_manager()),
        ("bitbucket", _build_bitbucket_manager),
        ("arize", _build_arize_manager),
    ],
    ids=lambda v: v[0] if isinstance(v, tuple) else v,
)
def test_jinja_env_is_sandboxed(manager_factory, monkeypatch):
    """Each prompt manager must render via ``ImmutableSandboxedEnvironment``."""
    _, factory = manager_factory
    manager = factory(monkeypatch)
    assert isinstance(manager.jinja_env, ImmutableSandboxedEnvironment)


@pytest.mark.parametrize(
    "manager_factory",
    [
        ("gitlab", lambda mp: _build_gitlab_manager()),
        ("bitbucket", _build_bitbucket_manager),
        ("arize", _build_arize_manager),
    ],
    ids=lambda v: v[0] if isinstance(v, tuple) else v,
)
@pytest.mark.parametrize("payload", _SSTI_PAYLOADS)
def test_jinja_env_blocks_ssti_payloads(manager_factory, payload, monkeypatch):
    """Attribute-traversal payloads must raise ``SecurityError`` at render time.

    A plain ``Environment()`` would happily evaluate these and execute
    arbitrary Python on the proxy host.
    """
    _, factory = manager_factory
    manager = factory(monkeypatch)
    template = manager.jinja_env.from_string(payload)
    with pytest.raises(SecurityError):
        template.render()


@pytest.mark.parametrize(
    "manager_factory",
    [
        ("gitlab", lambda mp: _build_gitlab_manager()),
        ("bitbucket", _build_bitbucket_manager),
        ("arize", _build_arize_manager),
    ],
    ids=lambda v: v[0] if isinstance(v, tuple) else v,
)
def test_jinja_env_still_renders_normal_variables(manager_factory, monkeypatch):
    """The sandbox is a strict superset for the legitimate use case — plain
    ``{{ var }}`` substitution must keep working unchanged."""
    _, factory = manager_factory
    manager = factory(monkeypatch)
    template = manager.jinja_env.from_string("Hello {{ name }}!")
    assert template.render(name="world") == "Hello world!"
