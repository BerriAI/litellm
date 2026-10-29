"""
The router surface fence (tests/code_coverage_tests/router_surface_check.py) must see every way
outside code reaches a Router member, or new dependencies slip past the shrink-only allowlist.
"""

import ast
import importlib.util
import pathlib

import pytest

_CHECK = pathlib.Path(__file__).parents[2] / "code_coverage_tests" / "router_surface_check.py"


@pytest.fixture(scope="module")
def surface_check():
    spec = importlib.util.spec_from_file_location("router_surface_check", _CHECK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _uses(surface_check, src: str) -> set:
    return surface_check.member_uses(ast.parse(src), surface_check.router_members())


def test_instance_and_dynamic_access_are_uses(surface_check):
    assert _uses(surface_check, "llm_router.get_model_list()\ngetattr(router, 'model_list')") == {
        "get_model_list",
        "model_list",
    }


def test_class_level_calls_are_uses(surface_check):
    assert _uses(surface_check, "litellm.Router.get_valid_args()") == {"get_valid_args"}
    assert _uses(surface_check, "from litellm import Router\nRouter.get_valid_args()") == {"get_valid_args"}


def test_container_file_endpoints_registered_by_name_are_members(surface_check):
    members = surface_check.router_members()
    assert {"aupload_container_file", "upload_container_file", "alist_container_files"} <= members
    assert _uses(surface_check, "await llm_router.aupload_container_file(file=f)") == {"aupload_container_file"}


def test_unlisted_class_level_use_fails_the_fence(surface_check):
    found = {
        "members": {"outside": {"get_valid_args": {"litellm/new_module.py"}}, "router_family": {}},
        "router_utils_imports": {},
    }
    allow = {"members_outside": [], "members_router_family": [], "router_utils_imports_outside": []}
    errors = surface_check.compare(allow, found)
    assert len(errors) == 1
    assert "'get_valid_args' in litellm/new_module.py" in errors[0]


def test_unrelated_receivers_are_not_uses(surface_check):
    assert _uses(surface_check, "client.get_model_list()\nRouterConfig.model_list") == set()
