import os
import sys
import pytest


sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

# Module under test
from litellm.proxy.prompts import prompt_registry as pr_mod

from litellm.types.prompts.init_prompts import (
    PromptInfo,
    PromptLiteLLMParams,
    PromptSpec,
)
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from contextlib import ExitStack

# ------------------------------
# Helpers
# ------------------------------
def make_prompt_spec(
        prompt_id="p1",
        integration="in_memory",
        content="hi",
        metadata=None,
):
    if metadata is None:
        metadata = {"model": "gpt-4"}
    return PromptSpec(
        prompt_id=prompt_id,
        litellm_params=PromptLiteLLMParams(
            prompt_id=prompt_id,                 # <-- add this line
            prompt_integration=integration,
            model_config={"content": content, "metadata": metadata},
        ),
        prompt_info=PromptInfo(prompt_type="config"),
    )



# ------------------------------
# get_prompt_initializer_from_integrations()
# ------------------------------
@patch.object(pr_mod, "importlib")
@patch.object(pr_mod, "os")
@patch.object(pr_mod, "Path")
def test_discovery_finds_initializer(PathMock, osMock, importlibMock, monkeypatch):
    # Pretend integrations dir exists with two packages: foo, bar (and __pycache__)
    PathMock.return_value.parent.parent.parent = "/x/y/z"
    osMock.path.exists.return_value = True
    osMock.listdir.return_value = ["foo", "bar", "__pycache__"]
    osMock.path.isdir.return_value = True
    osMock.path.exists.side_effect = lambda p: p.endswith("__init__.py") or True

    # import litellm.integrations.foo -> has registry
    foo_mod = SimpleNamespace(prompt_initializer_registry={"foo": lambda *a, **k: "FOO"})
    bar_mod = SimpleNamespace(prompt_initializer_registry={"bar": lambda *a, **k: "BAR"})
    def fake_import(name):
        if name.endswith(".foo"):
            return foo_mod
        if name.endswith(".bar"):
            return bar_mod
        raise ImportError(name)
    importlibMock.import_module.side_effect = fake_import

    discovered = pr_mod.get_prompt_initializer_from_integrations()
    assert set(discovered.keys()) == {"foo", "bar"}
    # Ensure no exception on module-level assignment
    assert isinstance(discovered["foo"], type(lambda: None))


# ------------------------------
# InMemoryPromptRegistry
# ------------------------------
@patch.object(pr_mod, "prompt_initializer_registry", new_callable=dict)
def test_in_memory_initialize_and_gets_callback(fake_registry, monkeypatch):
    fake_cb = MagicMock(spec=CustomPromptManagement)
    fake_registry["in_memory"] = lambda lp, p: fake_cb

    fake_llm = MagicMock()
    fake_llm.logging_callback_manager = MagicMock()

    real_litellm = sys.modules.get("litellm")

    with ExitStack() as stack:
        sys.modules["litellm"] = fake_llm
        # run the code inside a controlled scope
        reg = pr_mod.InMemoryPromptRegistry()
        spec = make_prompt_spec(prompt_id="pA", integration="in_memory")

        out = reg.initialize_prompt(spec)
        assert out is not None
        fake_llm.logging_callback_manager.add_litellm_callback.assert_called_with(fake_cb)

    # teardown — restore real module after context exit
    sys.modules["litellm"] = real_litellm




# ------------------------------
# GitlabPromptRegistry.load_all
# ------------------------------
@patch.object(pr_mod, "GitLabPromptCache")
def test_gitlab_load_all_populates_prompts(CacheMock, monkeypatch):
    # Fake cache instance with load_all + prompt_manager that acts like a CustomPromptManagement
    cache_instance = MagicMock()
    cache_instance.load_all.return_value = {
        "gitlab::a": {
            "content": "User: {{x}}",
            "metadata": {"model": "gpt-4o"},
        },
        "gitlab::b::c": {
            "content": "System: hi\nUser: {{q}}",
            "metadata": {"model": "gpt-4"},
        },
    }
    # prompt_manager used as callback; must be instance of CustomPromptManagement
    fake_pm_cb = MagicMock(spec=CustomPromptManagement)
    cache_instance.prompt_manager = fake_pm_cb
    CacheMock.return_value = cache_instance

    # Provide global_gitlab_config
    fake_llm = MagicMock()
    fake_llm.global_gitlab_config = {"project": "g/s/r", "access_token": "tkn"}
    monkeypatch.setattr(pr_mod, "litellm", fake_llm, raising=False)

    reg = pr_mod.GitlabPromptRegistry()
    out = reg.load_all()

    assert "gitlab::a" in out and "gitlab::b::c" in out
    # callback should be available
    assert reg.get_prompt_callback_by_id("gitlab::a") is fake_pm_cb
    assert reg.get_prompt_callback_by_id("gitlab::b::c") is fake_pm_cb
    # initialize_prompt should not duplicate
    again = reg.initialize_prompt(out["gitlab::a"])
    assert again is out["gitlab::a"]


# ------------------------------
# UnifiedPromptRegistry: register + precedence + listing
# ------------------------------
def test_unified_register_and_precedence(monkeypatch):
    # in-memory has p1, gitlab also has p1 — in_memory should win (registered first)
    p_in = make_prompt_spec(prompt_id="same", integration="in_memory")
    p_gl = make_prompt_spec(prompt_id="same", integration="gitlab")

    mem = MagicMock()
    mem.IN_MEMORY_PROMPTS = {"same": p_in}
    gl = MagicMock()
    gl.IN_MEMORY_PROMPTS = {"same": p_gl}

    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", mem)
    hub.register_registry("gitlab", gl)

    ids = hub.list_prompt_ids()
    assert ids == ["same"]
    # get returns in-memory version
    assert hub.get_prompt_by_id("same") is p_in


# ------------------------------
# UnifiedPromptRegistry: get_prompt_callback_by_id (fallback scan)
# ------------------------------
def test_unified_get_callback_fallback():
    mem = MagicMock()
    gl = MagicMock()

    # No mapping set; ensure fallback loop works
    cb = MagicMock(spec=CustomPromptManagement)
    gl.get_prompt_callback_by_id.return_value = cb
    mem.get_prompt_callback_by_id.return_value = None

    # Provide prompt data so it appears in cache
    p_gl = make_prompt_spec(prompt_id="g1", integration="gitlab")
    gl.IN_MEMORY_PROMPTS = {"g1": p_gl}
    mem.IN_MEMORY_PROMPTS = {}

    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", mem)
    hub.register_registry("gitlab", gl)

    # Rebuild aggregate cache
    hub.refresh()

    # Should find via gitlab registry fallback
    out_cb = hub.get_prompt_callback_by_id("g1")
    assert out_cb is cb


# ------------------------------
# UnifiedPromptRegistry: initialize_prompt routes by integration
# ------------------------------
def test_unified_initialize_routes_to_target_registry(monkeypatch):
    mem = MagicMock()
    gl = MagicMock()

    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", mem)
    hub.register_registry("gitlab", gl)

    # Explicitly route the integration
    hub.set_integration_route("gitlab", "gitlab")
    hub.set_integration_route("in_memory", "in_memory")

    p = make_prompt_spec(prompt_id="route1", integration="gitlab")
    gl.initialize_prompt.return_value = p

    out = hub.initialize_prompt(p)
    assert out is p
    gl.initialize_prompt.assert_called_once_with(p, None)


# ------------------------------
# UnifiedPromptRegistry: load_all calls underlying registries and aggregates
# ------------------------------
def test_unified_load_all_triggers_backends_and_aggregates():
    mem = MagicMock()
    gl = MagicMock()
    mem.load_all.return_value = {"m1": "..."}
    gl.load_all.return_value = {"g1": "..."}

    # Provide IN_MEMORY_PROMPTS for aggregation
    p1 = make_prompt_spec(prompt_id="m1", integration="in_memory")
    p2 = make_prompt_spec(prompt_id="g1", integration="gitlab")

    mem.IN_MEMORY_PROMPTS = {"m1": p1}
    gl.IN_MEMORY_PROMPTS = {"g1": p2}

    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", mem)
    hub.register_registry("gitlab", gl)

    out = hub.load_all()
    assert set(out.keys()) == {"m1", "g1"}
    mem.load_all.assert_called_once()
    gl.load_all.assert_called_once()


# ------------------------------
# UnifiedPromptRegistry: find_with_origin
# ------------------------------
def test_unified_find_with_origin():
    mem = MagicMock()
    gl = MagicMock()
    p1 = make_prompt_spec(prompt_id="m1", integration="in_memory")
    mem.IN_MEMORY_PROMPTS = {"m1": p1}
    gl.IN_MEMORY_PROMPTS = {}

    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", mem)
    hub.register_registry("gitlab", gl)

    name, spec = hub.find_with_origin("m1")
    assert name == "in_memory"
    assert spec is p1


# ------------------------------
# InMemoryPromptRegistry: duplicate initialize short-circuits
# ------------------------------
@patch.object(pr_mod, "prompt_initializer_registry", new_callable=dict)
def test_in_memory_initialize_twice_is_idempotent(fake_registry, monkeypatch):
    fake_cb = MagicMock(spec=CustomPromptManagement)
    fake_registry["in_memory"] = lambda lp, p: fake_cb

    fake_llm = MagicMock()
    fake_llm.logging_callback_manager = MagicMock()

    import sys
    real_litellm = sys.modules.get("litellm")
    sys.modules["litellm"] = fake_llm

    try:
        reg = pr_mod.InMemoryPromptRegistry()
        spec = make_prompt_spec(prompt_id="dup", integration="in_memory")

        out1 = reg.initialize_prompt(spec)
        out2 = reg.initialize_prompt(spec)
        assert out1 is out2
        fake_llm.logging_callback_manager.add_litellm_callback.assert_called_once_with(fake_cb)
    finally:
        sys.modules["litellm"] = real_litellm

@pytest.fixture
def fake_prompt_json():
    return {
        "id": "p1",
        "content": "Hello from GitLab",
        "metadata": {"model": "gpt-4", "temperature": 0.2},
        "model": "gpt-4",
    }


@pytest.fixture
def fake_prompt_spec():
    return PromptSpec(
        prompt_id="p1",
        litellm_params=PromptLiteLLMParams(
            prompt_id="p1",
            prompt_integration="gitlab",
            model_config={"model": "gpt-4"}
        ),
        prompt_info=PromptInfo(prompt_type="config"),
    )


def test_initialize_prompt_adds_callback_and_caches(monkeypatch, fake_prompt_spec):
    """Ensure initialize_prompt sets both caches and calls add_litellm_callback."""
    reg = pr_mod.GitlabPromptRegistry()
    fake_cache = MagicMock()
    fake_prompt_mgr = MagicMock(spec=CustomPromptManagement)
    fake_cache.prompt_manager = fake_prompt_mgr
    reg.gitlab_prompt_cache = fake_cache

    fake_llm = MagicMock()
    fake_llm.logging_callback_manager = MagicMock()
    real_litellm = sys.modules.get("litellm")
    sys.modules["litellm"] = fake_llm

    try:
        out = reg.initialize_prompt(fake_prompt_spec)
        # ✅ Compare content, not identity
        assert out.prompt_id == fake_prompt_spec.prompt_id
        assert out.litellm_params.prompt_integration == fake_prompt_spec.litellm_params.prompt_integration
        assert out.litellm_params.model_config == fake_prompt_spec.litellm_params.model_config
        assert out.prompt_info.prompt_type == fake_prompt_spec.prompt_info.prompt_type

        # ✅ Verify internal registry and callback setup
        assert reg.get_prompt_by_id("p1") == out
        assert reg.get_prompt_callback_by_id("p1") == fake_prompt_mgr
        fake_llm.logging_callback_manager.add_litellm_callback.assert_called_once_with(fake_prompt_mgr)


    finally:
        sys.modules["litellm"] = real_litellm


def test_initialize_prompt_raises_if_no_integration(fake_prompt_spec):
    """If prompt_integration missing in dict, expect ValidationError."""
    reg = pr_mod.GitlabPromptRegistry()
    bad_spec = fake_prompt_spec.model_copy(
        update={"litellm_params": {"prompt_id": "p1"}}
    )
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        reg.initialize_prompt(bad_spec)



def test_initialize_prompt_invalid_callback_type(monkeypatch, fake_prompt_spec):
    """Non-CustomPromptManagement prompt_manager should raise ValueError."""
    reg = pr_mod.GitlabPromptRegistry()
    fake_cache = MagicMock()
    fake_cache.prompt_manager = object()  # not subclass of CustomPromptManagement
    reg.gitlab_prompt_cache = fake_cache

    fake_llm = MagicMock()
    fake_llm.logging_callback_manager = MagicMock()
    real_litellm = sys.modules.get("litellm")
    sys.modules["litellm"] = fake_llm
    try:
        with pytest.raises(ValueError, match="CustomPromptManagement is required"):
            reg.initialize_prompt(fake_prompt_spec)
    finally:
        sys.modules["litellm"] = real_litellm


def test_load_all_populates_in_memory(monkeypatch, fake_prompt_json):
    """load_all should create PromptSpecs and cache them."""
    reg = pr_mod.GitlabPromptRegistry()
    fake_cache = MagicMock()
    fake_cache.load_all.return_value = {"p1": fake_prompt_json}
    fake_prompt_mgr = MagicMock(spec=CustomPromptManagement)
    fake_cache.prompt_manager = fake_prompt_mgr

    with patch.object(pr_mod, "GitLabPromptCache", return_value=fake_cache):
        fake_llm = MagicMock()
        fake_llm.logging_callback_manager = MagicMock()
        real_litellm = sys.modules.get("litellm")
        sys.modules["litellm"] = fake_llm

        try:
            result = reg.load_all()
            assert "p1" in result
            spec = result["p1"]
            assert isinstance(spec, PromptSpec)
            assert spec.litellm_params.prompt_integration == "gitlab"
            # ensure initialize_prompt is called and callback stored
            assert reg.get_prompt_callback_by_id("p1") is fake_prompt_mgr
            fake_llm.logging_callback_manager.add_litellm_callback.assert_called_once_with(fake_prompt_mgr)
        finally:
            sys.modules["litellm"] = real_litellm


def test_get_prompt_callback_by_id_triggers_lazy_init(monkeypatch):
    reg = pr_mod.GitlabPromptRegistry()
    fake_mgr = MagicMock(spec=CustomPromptManagement)
    fake_cache = MagicMock()
    fake_cache.prompt_manager = fake_mgr
    reg.gitlab_prompt_cache = fake_cache

    fake_prompt_spec = PromptSpec(
        prompt_id="lazy1",
        litellm_params=PromptLiteLLMParams(
            prompt_id="lazy1", prompt_integration="gitlab"
        ),
        prompt_info=PromptInfo(prompt_type="config"),
    )
    reg.IN_MEMORY_PROMPTS["lazy1"] = fake_prompt_spec

    fake_llm = MagicMock()
    fake_llm.logging_callback_manager = MagicMock()
    real_litellm = sys.modules.get("litellm")
    sys.modules["litellm"] = fake_llm
    try:
        cb = reg.get_prompt_callback_by_id("lazy1")
        assert cb is fake_mgr
        fake_llm.logging_callback_manager.add_litellm_callback.assert_called_once_with(fake_mgr)
    finally:
        sys.modules["litellm"] = real_litellm



def test_get_prompt_by_id_returns_none_for_missing():
    reg = pr_mod.GitlabPromptRegistry()
    assert reg.get_prompt_by_id("nope") is None


def test_get_prompt_callback_by_id_returns_none_if_not_found():
    reg = pr_mod.GitlabPromptRegistry()
    reg.gitlab_prompt_cache = None
    assert reg.get_prompt_callback_by_id("absent") is None


import pytest
from unittest.mock import MagicMock, create_autospec
from collections import OrderedDict

import litellm.proxy.prompts.prompt_registry as pr_mod


@pytest.fixture
def fake_prompt_spec():
    from litellm.proxy.prompts.prompt_registry import PromptSpec, PromptLiteLLMParams, PromptInfo
    return PromptSpec(
        prompt_id="p1",
        litellm_params=PromptLiteLLMParams(prompt_id="p1", prompt_integration="gitlab", model_config={"model": "gpt-4"}),
        prompt_info=PromptInfo(prompt_type="config"),
    )


@pytest.fixture
def fake_registries(fake_prompt_spec):
    """Create two fake registries with in-memory behavior."""
    reg_a = MagicMock()
    reg_b = MagicMock()

    reg_a.IN_MEMORY_PROMPTS = {"p1": fake_prompt_spec}
    reg_a.get_prompt_by_id.return_value = fake_prompt_spec
    reg_a.get_prompt_callback_by_id.return_value = "cb_a"
    reg_a.initialize_prompt.return_value = fake_prompt_spec
    reg_a.load_all.return_value = reg_a.IN_MEMORY_PROMPTS

    reg_b.IN_MEMORY_PROMPTS = {"p2": fake_prompt_spec.model_copy(update={"prompt_id": "p2"})}
    reg_b.get_prompt_by_id.return_value = reg_b.IN_MEMORY_PROMPTS["p2"]
    reg_b.get_prompt_callback_by_id.return_value = "cb_b"
    reg_b.initialize_prompt.return_value = reg_b.IN_MEMORY_PROMPTS["p2"]
    reg_b.load_all.return_value = reg_b.IN_MEMORY_PROMPTS

    return reg_a, reg_b


# ----------------------------------------------------------------------
# Registration and routing
# ----------------------------------------------------------------------

def test_register_registry_and_integration_routing(fake_registries):
    reg_a, reg_b = fake_registries
    hub = pr_mod.UnifiedPromptRegistry()

    hub.register_registry("in_memory", reg_a)
    hub.register_registry("gitlab", reg_b)

    assert list(hub._registries.keys()) == ["in_memory", "gitlab"]
    assert hub._integration_to_registry["gitlab"] == "gitlab"
    assert hub._integration_to_registry["in_memory"] == "in_memory"
    assert isinstance(hub.IN_MEMORY_PROMPTS, dict)
    assert "p1" in hub.IN_MEMORY_PROMPTS
    assert "p2" in hub.IN_MEMORY_PROMPTS


def test_set_integration_route_and_error(fake_registries):
    reg_a, reg_b = fake_registries
    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", reg_a)
    hub.register_registry("gitlab", reg_b)

    hub.set_integration_route("custom", "gitlab")
    assert hub._integration_to_registry["custom"] == "gitlab"

    with pytest.raises(ValueError, match="Unknown registry"):
        hub.set_integration_route("foo", "unknown")


# ----------------------------------------------------------------------
# Lookup behavior
# ----------------------------------------------------------------------

def test_get_prompt_by_id_hits_cache(fake_registries):
    reg_a, _ = fake_registries
    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", reg_a)

    # should hit IN_MEMORY_PROMPTS
    result = hub.get_prompt_by_id("p1")
    assert result is reg_a.IN_MEMORY_PROMPTS["p1"]
    reg_a.get_prompt_by_id.assert_not_called()


def test_get_prompt_by_id_fallback_and_cache(fake_registries):
    reg_a, _ = fake_registries
    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", reg_a)

    # simulate empty cache
    hub.IN_MEMORY_PROMPTS.clear()
    result = hub.get_prompt_by_id("p1")
    assert result == reg_a.get_prompt_by_id.return_value
    assert "p1" in hub.IN_MEMORY_PROMPTS


def test_get_prompt_callback_by_id_prefers_origin(fake_registries):
    reg_a, reg_b = fake_registries
    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", reg_a)
    hub.register_registry("gitlab", reg_b)

    # manually mark p1 origin
    hub.prompt_id_to_registry["p1"] = reg_a
    hub.prompt_id_to_registry_name["p1"] = "in_memory"

    cb = hub.get_prompt_callback_by_id("p1")
    assert cb == "cb_a"
    reg_a.get_prompt_callback_by_id.assert_called_once_with("p1")


def test_get_prompt_callback_by_id_fallback(fake_registries):
    reg_a, reg_b = fake_registries
    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", reg_a)
    hub.register_registry("gitlab", reg_b)

    hub.prompt_id_to_registry.clear()
    cb = hub.get_prompt_callback_by_id("p2")
    # ✅ first registered registry wins in fallback
    assert cb == "cb_a"



def test_find_with_origin_from_cache(fake_registries):
    reg_a, _ = fake_registries
    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", reg_a)

    name, spec = hub.find_with_origin("p1")
    assert name == "in_memory"
    assert spec.prompt_id == "p1"


def test_find_with_origin_not_in_cache(fake_registries):
    reg_a, _ = fake_registries
    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", reg_a)
    hub.IN_MEMORY_PROMPTS.clear()

    result = hub.find_with_origin("p1")
    assert result[0] == "in_memory"
    assert result[1].prompt_id == "p1"


# ----------------------------------------------------------------------
# Initialization routing
# ----------------------------------------------------------------------

def test_initialize_prompt_routes_to_correct_registry(fake_registries, fake_prompt_spec):
    reg_a, reg_b = fake_registries
    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("gitlab", reg_b)
    hub.set_integration_route("gitlab", "gitlab")

    out = hub.initialize_prompt(fake_prompt_spec)
    # ✅ matches mock’s return (reg_b returns p2)
    assert out.prompt_id == "p2"
    reg_b.initialize_prompt.assert_called_once_with(fake_prompt_spec, None)
    assert "p2" in hub.IN_MEMORY_PROMPTS



def test_initialize_prompt_raises_if_no_registry(fake_prompt_spec):
    hub = pr_mod.UnifiedPromptRegistry()
    with pytest.raises(RuntimeError, match="no registered registries"):
        hub.initialize_prompt(fake_prompt_spec)


def test_initialize_prompt_raises_if_registry_missing_method(fake_prompt_spec):
    hub = pr_mod.UnifiedPromptRegistry()
    bad_registry = object()
    hub.register_registry("dummy", bad_registry)
    hub._integration_to_registry["gitlab"] = "dummy"

    with pytest.raises(RuntimeError, match="does not support initialize_prompt"):
        hub.initialize_prompt(fake_prompt_spec)


# ----------------------------------------------------------------------
# Cache management
# ----------------------------------------------------------------------

def test_rebuild_cache_aggregates(fake_registries):
    reg_a, reg_b = fake_registries
    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", reg_a)
    hub.register_registry("gitlab", reg_b)

    hub.IN_MEMORY_PROMPTS.clear()
    hub._rebuild_cache()
    ids = hub.list_prompt_ids()
    assert "p1" in ids and "p2" in ids
    assert isinstance(hub.list_prompts(), list)
    assert len(hub.list_prompts()) >= 2


def test_cache_replace_and_if_absent(fake_prompt_spec):
    hub = pr_mod.UnifiedPromptRegistry()
    hub._cache_if_absent("p1", fake_prompt_spec)
    assert "p1" in hub.IN_MEMORY_PROMPTS

    new_spec = fake_prompt_spec.model_copy(update={"prompt_id": "p1"})
    hub._cache_replace("p1", new_spec)
    assert hub.IN_MEMORY_PROMPTS["p1"] is new_spec


def test_load_all_calls_each_registry(fake_registries):
    reg_a, reg_b = fake_registries
    hub = pr_mod.UnifiedPromptRegistry()
    hub.register_registry("in_memory", reg_a)
    hub.register_registry("gitlab", reg_b)

    result = hub.load_all()
    reg_a.load_all.assert_called_once()
    reg_b.load_all.assert_called_once()
    assert "p1" in result
    assert "p2" in result
