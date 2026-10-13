"""Tests for the HTTP command group."""

import json

import pytest
from click.testing import CliRunner



import responses

from litellm.proxy.client.cli.commands.http import http


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@responses.activate
def test_request_get(runner):
    """Test making a GET request."""
    responses.add(
        responses.GET,
        "http://localhost:4000/models",
        json={"models": []},
        status=200,
    )
    result = runner.invoke(
        http,
        ["request", "GET", "/models"],
        obj={"base_url": "http://localhost:4000", "api_key": "sk-test-key"},
    )
    assert result.exit_code == 0
    assert "models" in result.output


@responses.activate
def test_request_post_with_json(runner):
    """Test making a POST request with JSON data."""
    responses.add(
        responses.POST,
        "http://localhost:4000/chat/completions",
        json={"choices": [{"message": {"content": "Hello!"}}]},
        status=200,
    )
    result = runner.invoke(
        http,
        [
            "request",
            "POST",
            "/chat/completions",
            "-j",
            '{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}',
        ],
        obj={"base_url": "http://localhost:4000", "api_key": "sk-test-key"},
    )
    assert result.exit_code == 0
    assert "choices" in result.output


@responses.activate
def test_request_with_headers(runner):
    """Test making a request with custom headers."""
    responses.add(
        responses.GET,
        "http://localhost:4000/models",
        json={"models": []},
        status=200,
    )
    result = runner.invoke(
        http,
        [
            "request",
            "GET",
            "/models",
            "-H",
            "X-Custom-Header:value",
            "-H",
            "Accept:application/json",
        ],
        obj={"base_url": "http://localhost:4000", "api_key": "sk-test-key"},
    )
    assert result.exit_code == 0
    assert "models" in result.output


def test_request_invalid_json(runner):
    """Test error handling for invalid JSON data."""
    result = runner.invoke(
        http,
        [
            "request",
            "POST",
            "/chat/completions",
            "-j",
            '{"invalid": json}',  # Invalid JSON
        ],
        obj={"base_url": "http://localhost:4000", "api_key": "sk-test-key"},
    )
    assert result.exit_code == 2  # Click error code for invalid parameter
    assert "Invalid JSON format" in result.output


def test_request_invalid_header(runner):
    """Test error handling for invalid header format."""
    result = runner.invoke(
        http,
        [
            "request",
            "GET",
            "/models",
            "-H",
            "invalid-header",  # Invalid header format
        ],
        obj={"base_url": "http://localhost:4000", "api_key": "sk-test-key"},
    )
    assert result.exit_code == 2  # Click error code for invalid parameter
    assert "Invalid header format" in result.output


@pytest.mark.parametrize(
    "raw_body",
    [
        '{"count": "12", "ratio": 1, "whole": 1.0, "enabled": "true", "flag": true, "missing": null}',
        '{"data": [{"id": 1, "tags": ["a", 2, null, false]}, "two", null], "nested": {"deep": {"n": "007"}}}',
        '[{"id": "1"}, 2, "3", null]',
        '"plain string"',
        "12",
        "null",
        "{}",
    ],
)
@responses.activate
def test_request_prints_the_server_json_unchanged(runner, raw_body):
    responses.add(
        responses.GET,
        "http://localhost:4000/models",
        body=raw_body,
        content_type="application/json",
        status=200,
    )
    result = runner.invoke(
        http,
        ["request", "GET", "/models"],
        obj={"base_url": "http://localhost:4000", "api_key": "sk-test-key"},
    )
    assert result.exit_code == 0
    assert json.dumps(json.loads(result.stdout)) == json.dumps(json.loads(raw_body))


@pytest.mark.parametrize(
    "raw_body",
    [
        '{"error": {"message": "model not found", "code": "404", "status": 404, "retry": false}}',
        '["first", {"detail": null}]',
        '"denied"',
    ],
)
@responses.activate
def test_request_http_error_prints_the_json_error_body_unchanged(runner, raw_body):
    responses.add(
        responses.GET,
        "http://localhost:4000/models",
        body=raw_body,
        content_type="application/json",
        status=404,
    )
    result = runner.invoke(
        http,
        ["request", "GET", "/models"],
        obj={"base_url": "http://localhost:4000", "api_key": "sk-test-key"},
    )
    assert result.exit_code == 1
    assert result.stderr.splitlines()[:1] == ["Error: HTTP 404"]
    assert json.dumps(json.loads(result.stdout)) == json.dumps(json.loads(raw_body))


@responses.activate
def test_request_http_error_with_a_non_json_body_echoes_the_text(runner):
    responses.add(
        responses.GET,
        "http://localhost:4000/models",
        body="upstream exploded",
        status=502,
    )
    result = runner.invoke(
        http,
        ["request", "GET", "/models"],
        obj={"base_url": "http://localhost:4000", "api_key": "sk-test-key"},
    )
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.splitlines()[:2] == ["Error: HTTP 502", "upstream exploded"]


@pytest.mark.parametrize(
    ("api_key", "expected_authorization"),
    [
        ("sk-test-key", "Bearer sk-test-key"),
        (None, None),
        ("", None),
    ],
)
@responses.activate
def test_request_sends_the_context_api_key_to_the_context_base_url(runner, api_key, expected_authorization):
    responses.add(responses.GET, "http://proxy.internal:4100/models", json={"models": []}, status=200)
    result = runner.invoke(
        http,
        ["request", "GET", "/models"],
        obj={"base_url": "http://proxy.internal:4100/", "api_key": api_key},
    )
    assert result.exit_code == 0
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == "http://proxy.internal:4100/models"
    assert responses.calls[0].request.headers.get("Authorization") == expected_authorization


@pytest.mark.parametrize(
    ("obj", "missing_key"),
    [
        ({"api_key": "sk-test-key"}, "base_url"),
        ({"base_url": "http://localhost:4000"}, "api_key"),
        ({}, "base_url"),
    ],
)
def test_request_with_an_incomplete_context_fails_on_the_first_missing_value(runner, obj, missing_key):
    result = runner.invoke(http, ["request", "GET", "/models"], obj=obj)
    assert result.exit_code == 1
    assert type(result.exception) is KeyError
    assert result.exception.args == (missing_key,)


def test_request_without_a_context_object_fails_instead_of_assuming_a_default_server(runner):
    result = runner.invoke(http, ["request", "GET", "/models"], obj=None)
    assert result.exit_code == 1
    assert type(result.exception) is TypeError
    assert result.output == ""
