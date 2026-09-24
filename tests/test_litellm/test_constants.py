import ast
import importlib
import inspect

import pytest

from litellm import constants


def _build_constant_env_var_map() -> dict[str, str]:
    """
    Build a mapping of CONSTANT_NAME -> ENV_VAR_NAME by parsing constants.py.

    This keeps the test resilient when a constant name and env var name differ
    (e.g., aliases like LITELLM_* env vars).
    """
    env_var_map: dict[str, str] = {}
    constants_source = inspect.getsource(constants)
    parsed = ast.parse(constants_source)

    for node in parsed.body:
        if not isinstance(node, ast.Assign):
            continue

        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue

        constant_name = node.targets[0].id
        env_var_name = None

        for child in ast.walk(node.value):
            if not isinstance(child, ast.Call):
                continue

            # os.getenv("ENV_NAME", default)
            if (
                isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "os"
                and child.func.attr == "getenv"
                and len(child.args) >= 1
                and isinstance(child.args[0], ast.Constant)
                and isinstance(child.args[0].value, str)
            ):
                env_var_name = child.args[0].value
                break

            # get_env_int("ENV_NAME", default)
            if (
                isinstance(child.func, ast.Name)
                and child.func.id == "get_env_int"
                and len(child.args) >= 1
                and isinstance(child.args[0], ast.Constant)
                and isinstance(child.args[0].value, str)
            ):
                env_var_name = child.args[0].value
                break

        if env_var_name:
            env_var_map[constant_name] = env_var_name

    return env_var_map


def _cipher_suites(cipher_string: str) -> list[str]:
    return [suite for suite in cipher_string.split(":") if suite]


def test_default_ssl_ciphers_excludes_chacha20():
    assert all("CHACHA" not in suite for suite in _cipher_suites(constants.DEFAULT_SSL_CIPHERS))


def test_default_ssl_ciphers_tls12_suites_all_offer_forward_secrecy():
    tls12_suites = [suite for suite in _cipher_suites(constants.DEFAULT_SSL_CIPHERS) if not suite.startswith("TLS_")]
    assert tls12_suites
    assert all(suite.startswith("ECDHE-") for suite in tls12_suites)


def test_fips_ssl_ciphers_is_a_subset_of_default():
    assert set(_cipher_suites(constants.FIPS_SSL_CIPHERS)) <= set(_cipher_suites(constants.DEFAULT_SSL_CIPHERS))


def test_fips_ssl_ciphers_tls12_suites_are_all_gcm():
    tls12_suites = [suite for suite in _cipher_suites(constants.FIPS_SSL_CIPHERS) if not suite.startswith("TLS_")]
    assert tls12_suites
    assert all("-GCM-" in suite for suite in tls12_suites)


def test_ssl_ciphers_env_override_applies_to_both_lists(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_SSL_CIPHERS", "ECDHE-RSA-AES128-GCM-SHA256")
    importlib.reload(constants)
    try:
        assert constants.DEFAULT_SSL_CIPHERS == "ECDHE-RSA-AES128-GCM-SHA256"
        assert constants.FIPS_SSL_CIPHERS == "ECDHE-RSA-AES128-GCM-SHA256"
    finally:
        monkeypatch.delenv("LITELLM_SSL_CIPHERS")
        importlib.reload(constants)
    assert constants.DEFAULT_SSL_CIPHERS != "ECDHE-RSA-AES128-GCM-SHA256"
