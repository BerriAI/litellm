"""Tests for the HTTP command group."""

import json
import os
import sys

import pytest
from click.testing import CliRunner

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


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
