import pytest
from fastapi import HTTPException

from litellm.proxy.utils import (
    construct_database_url_from_env_vars,
    get_prisma_client_or_throw,
    is_valid_api_key,
)


def normalize(value):
    return value


def test_get_prisma_client_or_throw_happy_path_returns_client(monkeypatch):
    sentinel = object()
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", sentinel, raising=False)
    result = get_prisma_client_or_throw("some message")
    summary = {
        "is_sentinel": result is sentinel,
        "message_arg": "some message",
        "raised": False,
    }
    assert summary == {
        "is_sentinel": True,
        "message_arg": "some message",
        "raised": False,
    }


def test_get_prisma_client_or_throw_raises_when_client_none(monkeypatch):
    import litellm.proxy.proxy_server as ps

    monkeypatch.setattr(ps, "prisma_client", None, raising=False)
    with pytest.raises(HTTPException) as exc_info:
        get_prisma_client_or_throw("db not connected")
    snapshot = {
        "status_code": exc_info.value.status_code,
        "is_dict_detail": isinstance(exc_info.value.detail, dict),
        "error_message": exc_info.value.detail["error"],
    }
    assert snapshot == {
        "status_code": 500,
        "is_dict_detail": True,
        "error_message": "db not connected",
    }


def test_is_valid_api_key_happy_path_sk_prefix():
    summary = {
        "result": is_valid_api_key("sk-abc123_XYZ-456"),
        "key": "sk-abc123_XYZ-456",
        "len": len("sk-abc123_XYZ-456"),
    }
    assert summary == {
        "result": True,
        "key": "sk-abc123_XYZ-456",
        "len": 17,
    }


def test_is_valid_api_key_happy_path_hashed_64_hex():
    key = "a" * 64
    summary = {
        "result": is_valid_api_key(key),
        "key_len": len(key),
        "is_hex": True,
    }
    assert summary == {
        "result": True,
        "key_len": 64,
        "is_hex": True,
    }


def test_is_valid_api_key_happy_path_mixed_case_hex():
    key = "AbCdEf0123456789" * 4
    summary = {
        "result": is_valid_api_key(key),
        "key_len": len(key),
        "first": key[0],
    }
    assert summary == {
        "result": True,
        "key_len": 64,
        "first": "A",
    }


def test_is_valid_api_key_error_path_too_long():
    assert is_valid_api_key("sk-" + "a" * 200) is False


def test_is_valid_api_key_error_path_non_string():
    assert is_valid_api_key(12345) is False  # type: ignore[arg-type]


def test_is_valid_api_key_error_path_invalid_format():
    assert is_valid_api_key("not-a-valid-key-format!!!!") is False


def test_is_valid_api_key_error_path_too_short():
    assert is_valid_api_key("sk") is False


def test_construct_database_url_from_env_vars_happy_path_full(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "db.example.com")
    monkeypatch.setenv("DATABASE_USERNAME", "user")
    monkeypatch.setenv("DATABASE_PASSWORD", "pass")
    monkeypatch.setenv("DATABASE_NAME", "litellm")
    monkeypatch.delenv("DATABASE_SCHEMA", raising=False)
    result = construct_database_url_from_env_vars()
    summary = {
        "result": result,
        "host": "db.example.com",
        "scheme": result.split("://", 1)[0] if result else None,
        "has_password": "pass" in (result or ""),
    }
    assert summary == {
        "result": "postgresql://user:pass@db.example.com/litellm",
        "host": "db.example.com",
        "scheme": "postgresql",
        "has_password": True,
    }


def test_construct_database_url_from_env_vars_happy_path_no_password(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "db.example.com")
    monkeypatch.setenv("DATABASE_USERNAME", "user")
    monkeypatch.delenv("DATABASE_PASSWORD", raising=False)
    monkeypatch.setenv("DATABASE_NAME", "litellm")
    monkeypatch.delenv("DATABASE_SCHEMA", raising=False)
    result = construct_database_url_from_env_vars()
    summary = {
        "result": result,
        "no_colon_password": ":pass@" not in (result or ""),
        "host": "db.example.com",
        "user": "user",
    }
    assert summary == {
        "result": "postgresql://user@db.example.com/litellm",
        "no_colon_password": True,
        "host": "db.example.com",
        "user": "user",
    }


def test_construct_database_url_from_env_vars_special_chars_encoded(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "db.example.com")
    monkeypatch.setenv("DATABASE_USERNAME", "us er@x")
    monkeypatch.setenv("DATABASE_PASSWORD", "p@ss/word")
    monkeypatch.setenv("DATABASE_NAME", "lite/llm")
    monkeypatch.delenv("DATABASE_SCHEMA", raising=False)
    result = construct_database_url_from_env_vars()
    summary = {
        "result": result,
        "username_encoded": "us+er%40x" in result,
        "password_encoded": "p%40ss%2Fword" in result,
        "name_encoded": "lite%2Fllm" in result,
    }
    assert summary == {
        "result": "postgresql://us+er%40x:p%40ss%2Fword@db.example.com/lite%2Fllm",
        "username_encoded": True,
        "password_encoded": True,
        "name_encoded": True,
    }


def test_construct_database_url_from_env_vars_with_schema(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "db.example.com")
    monkeypatch.setenv("DATABASE_USERNAME", "user")
    monkeypatch.setenv("DATABASE_PASSWORD", "pass")
    monkeypatch.setenv("DATABASE_NAME", "litellm")
    monkeypatch.setenv("DATABASE_SCHEMA", "public")
    result = construct_database_url_from_env_vars()
    summary = {
        "result": result,
        "schema_appended": result.endswith("?schema=public"),
        "host": "db.example.com",
    }
    assert summary == {
        "result": "postgresql://user:pass@db.example.com/litellm?schema=public",
        "schema_appended": True,
        "host": "db.example.com",
    }


def test_construct_database_url_from_env_vars_error_path_missing_host(monkeypatch):
    monkeypatch.delenv("DATABASE_HOST", raising=False)
    monkeypatch.setenv("DATABASE_USERNAME", "user")
    monkeypatch.setenv("DATABASE_NAME", "litellm")
    assert construct_database_url_from_env_vars() is None


def test_construct_database_url_from_env_vars_error_path_missing_username(monkeypatch):
    monkeypatch.setenv("DATABASE_HOST", "db.example.com")
    monkeypatch.delenv("DATABASE_USERNAME", raising=False)
    monkeypatch.setenv("DATABASE_NAME", "litellm")
    assert construct_database_url_from_env_vars() is None
