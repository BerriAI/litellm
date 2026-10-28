"""Smoke tests for the proxy_server/ test harness.

Validates that fixtures + scripts work end-to-end before PR1/PR2/PR3 depend
on them. ``_pin_check.py`` skips this file explicitly so it doesn't count
toward behavior pinning.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from .conftest import (  # type: ignore[import-not-found]
    make_acompletion_response,
    make_embedding_response,
    normalize,
)

HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Fixture smoke tests
# ---------------------------------------------------------------------------


def test_app_fixture_returns_fastapi_app(app):
    assert isinstance(app, FastAPI)
    assert app.router is not None


def test_client_fixture_returns_testclient(client):
    assert isinstance(client, TestClient)
    assert hasattr(client, "post")
    assert hasattr(client, "get")


def test_mock_prisma_has_team_table(mock_prisma):
    assert hasattr(mock_prisma.db, "litellm_teamtable")
    assert callable(mock_prisma.db.litellm_teamtable.find_unique)
    assert callable(mock_prisma.db.litellm_teamtable.find_many)


def test_mock_prisma_has_key_table(mock_prisma):
    assert hasattr(mock_prisma.db, "litellm_verificationtoken")
    assert callable(mock_prisma.db.litellm_verificationtoken.find_unique)


def test_auth_as_admin_overrides_dependency(app, auth_as):
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        assert user_api_key_auth in app.dependency_overrides


def test_auth_as_internal_user_overrides_dependency(app, auth_as):
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    with auth_as(LitellmUserRoles.INTERNAL_USER) as fake_auth:
        assert user_api_key_auth in app.dependency_overrides
        assert fake_auth.user_role == LitellmUserRoles.INTERNAL_USER


def test_auth_as_cleans_up_on_exit(app, auth_as):
    from litellm.proxy._types import LitellmUserRoles
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    assert user_api_key_auth not in app.dependency_overrides
    with auth_as(LitellmUserRoles.PROXY_ADMIN):
        pass
    assert user_api_key_auth not in app.dependency_overrides


def test_mock_router_acompletion_callable(mock_router):
    from unittest.mock import AsyncMock

    assert isinstance(mock_router.acompletion, AsyncMock)
    assert isinstance(mock_router.aembedding, AsyncMock)
    assert isinstance(mock_router.aimage_generation, AsyncMock)


@pytest.mark.asyncio
async def test_make_acompletion_response_stream():
    gen = make_acompletion_response(model="gpt-4", stream=True)
    chunks = [chunk async for chunk in gen]
    assert len(chunks) >= 1
    # Last chunk should have finish_reason set
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_make_acompletion_response_tools():
    resp = make_acompletion_response(
        model="gpt-4",
        tools=[{"type": "function", "function": {"name": "fake_tool"}}],
    )
    assert resp.choices[0].message.tool_calls is not None
    assert resp.choices[0].message.tool_calls[0].function.name == "fake_tool"


def test_make_embedding_response_shape():
    resp = make_embedding_response(input=["a", "b", "c"], dimensions=4)
    data = resp.data
    assert len(data) == 3
    assert len(data[0]["embedding"]) == 4


def test_normalize_replaces_volatile_keys():
    out = normalize({"key": "abc", "spend": 0, "nested": {"id": "x", "value": 5}})
    assert out == {
        "key": "<VOLATILE>",
        "spend": 0,
        "nested": {"id": "<VOLATILE>", "value": 5},
    }


def test_normalize_handles_lists():
    out = normalize([{"key": "a"}, {"key": "b"}])
    assert out == [{"key": "<VOLATILE>"}, {"key": "<VOLATILE>"}]


# ---------------------------------------------------------------------------
# Script smoke tests — _coverage_check.py
# ---------------------------------------------------------------------------


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so dataclasses can resolve cls.__module__.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_cov_xml(tmp_path: Path, line_rate: float, branch_rate: float) -> Path:
    xml = textwrap.dedent(f"""\
        <?xml version="1.0" ?>
        <coverage version="7.0">
          <packages>
            <package name="litellm.proxy">
              <classes>
                <class filename="litellm/proxy/proxy_server.py"
                       line-rate="{line_rate}" branch-rate="{branch_rate}"/>
              </classes>
            </package>
          </packages>
        </coverage>
        """)
    path = tmp_path / "cov.xml"
    path.write_text(xml)
    return path


def test_coverage_check_pass_on_synthetic_xml(tmp_path):
    cov_check = _load_script("_coverage_check")
    xml = _write_cov_xml(tmp_path, line_rate=0.75, branch_rate=0.60)
    line_pct, branch_pct = cov_check.parse_coverage_xml(xml)
    assert line_pct == pytest.approx(75.0)
    assert branch_pct == pytest.approx(60.0)


def test_coverage_check_fail_on_low_coverage(tmp_path, monkeypatch, capsys):
    cov_check = _load_script("_coverage_check")
    xml = _write_cov_xml(tmp_path, line_rate=0.10, branch_rate=0.05)
    monkeypatch.setattr(
        sys,
        "argv",
        ["_coverage_check.py", "--pr-target", "3", "--coverage-xml", str(xml)],
    )
    rc = cov_check.main()
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_coverage_check_pass_on_high_coverage(tmp_path, monkeypatch, capsys):
    cov_check = _load_script("_coverage_check")
    xml = _write_cov_xml(tmp_path, line_rate=0.75, branch_rate=0.60)
    monkeypatch.setattr(
        sys,
        "argv",
        ["_coverage_check.py", "--pr-target", "3", "--coverage-xml", str(xml)],
    )
    rc = cov_check.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out


# ---------------------------------------------------------------------------
# Script smoke tests — _pin_check.py
# ---------------------------------------------------------------------------


def _write_pin_list(tmp_path: Path, items: list) -> Path:
    path = tmp_path / "pins.txt"
    path.write_text("\n".join(f"- `{item}`" for item in items) + "\n")
    return path


def _write_test_file(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


def test_pin_check_pass_on_complete_pins(tmp_path):
    pin_check = _load_script("_pin_check")
    _write_pin_list(tmp_path, ["update_cache"])
    _write_test_file(
        tmp_path,
        "test_thing.py",
        """\
        def test_update_cache_happy():
            data = update_cache(value=1)
            assert data == {"key1": 1, "key2": 2, "key3": 3}

        def test_update_cache_error():
            import pytest
            with pytest.raises(ValueError):
                update_cache(value=None)
        """,
    )
    pin_list = pin_check.parse_pin_list(tmp_path / "pins.txt")
    funcs = pin_check.collect_test_functions(tmp_path)
    ok, failures = pin_check.check(pin_list, funcs)
    assert ok, failures


def test_pin_check_fail_on_missing_pin(tmp_path):
    pin_check = _load_script("_pin_check")
    _write_pin_list(tmp_path, ["update_cache", "never_referenced_symbol"])
    _write_test_file(
        tmp_path,
        "test_thing.py",
        """\
        def test_update_cache_happy():
            data = update_cache(value=1)
            assert data == {"key1": 1, "key2": 2, "key3": 3}

        def test_update_cache_error():
            import pytest
            with pytest.raises(ValueError):
                update_cache(value=None)
        """,
    )
    pin_list = pin_check.parse_pin_list(tmp_path / "pins.txt")
    funcs = pin_check.collect_test_functions(tmp_path)
    ok, failures = pin_check.check(pin_list, funcs)
    assert not ok
    assert any("never_referenced_symbol" in f for f in failures)


def test_pin_check_fail_on_status_only_test(tmp_path):
    pin_check = _load_script("_pin_check")
    _write_pin_list(tmp_path, ["some_route"])
    _write_test_file(
        tmp_path,
        "test_thing.py",
        """\
        def test_some_route_happy():
            response = client.get("/some_route")
            assert response.status_code == 200

        def test_some_route_error():
            response = client.get("/some_route")
            assert response.status_code == 404
        """,
    )
    pin_list = pin_check.parse_pin_list(tmp_path / "pins.txt")
    funcs = pin_check.collect_test_functions(tmp_path)
    ok, failures = pin_check.check(pin_list, funcs)
    assert not ok
    assert any("status-only" in f for f in failures)
