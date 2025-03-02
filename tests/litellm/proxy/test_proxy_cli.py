import importlib
import json
import os
import socket
import subprocess
import sys
from unittest.mock import MagicMock, mock_open, patch

import click
import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path

import litellm
from litellm.proxy.proxy_cli import ProxyInitializationHelpers


class TestProxyInitializationHelpers:

    @patch("importlib.metadata.version")
    @patch("click.echo")
    def test_echo_litellm_version(self, mock_echo, mock_version):
        # Setup
        mock_version.return_value = "1.0.0"

        # Execute
        ProxyInitializationHelpers._echo_litellm_version()

        # Assert
        mock_version.assert_called_once_with("litellm")
        mock_echo.assert_called_once_with("\nLiteLLM: Current Version = 1.0.0\n")

    @patch("httpx.get")
    @patch("builtins.print")
    @patch("json.dumps")
    def test_run_health_check(self, mock_dumps, mock_print, mock_get):
        # Setup
        mock_response = MagicMock()
        mock_response.json.return_value = {"status": "healthy"}
        mock_get.return_value = mock_response
        mock_dumps.return_value = '{"status": "healthy"}'

        # Execute
        ProxyInitializationHelpers._run_health_check("localhost", 8000)

        # Assert
        mock_get.assert_called_once_with(url="http://localhost:8000/health")
        mock_response.json.assert_called_once()
        mock_dumps.assert_called_once_with({"status": "healthy"}, indent=4)

    @patch("openai.OpenAI")
    @patch("click.echo")
    @patch("builtins.print")
    def test_run_test_chat_completion(self, mock_print, mock_echo, mock_openai):
        # Setup
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        mock_response = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_stream_response = MagicMock()
        mock_stream_response.__iter__.return_value = [MagicMock(), MagicMock()]
        mock_client.chat.completions.create.side_effect = [
            mock_response,
            mock_stream_response,
        ]

        # Execute
        with pytest.raises(ValueError, match="Invalid test value"):
            ProxyInitializationHelpers._run_test_chat_completion(
                "localhost", 8000, "gpt-3.5-turbo", True
            )

        # Test with valid string test value
        ProxyInitializationHelpers._run_test_chat_completion(
            "localhost", 8000, "gpt-3.5-turbo", "http://test-url"
        )

        # Assert
        mock_openai.assert_called_once_with(
            api_key="My API Key", base_url="http://test-url"
        )
        mock_client.chat.completions.create.assert_called()

    def test_get_default_unvicorn_init_args(self):
        # Test without log_config
        args = ProxyInitializationHelpers._get_default_unvicorn_init_args(
            "localhost", 8000
        )
        assert args["app"] == "litellm.proxy.proxy_server:app"
        assert args["host"] == "localhost"
        assert args["port"] == 8000

        # Test with log_config
        args = ProxyInitializationHelpers._get_default_unvicorn_init_args(
            "localhost", 8000, "log_config.json"
        )
        assert args["log_config"] == "log_config.json"

        # Test with json_logs=True
        with patch("litellm.json_logs", True):
            args = ProxyInitializationHelpers._get_default_unvicorn_init_args(
                "localhost", 8000
            )
            assert args["log_config"] is None

    @patch("asyncio.run")
    @patch("builtins.print")
    def test_init_hypercorn_server(self, mock_print, mock_asyncio_run):
        # Setup
        mock_app = MagicMock()

        # Execute
        ProxyInitializationHelpers._init_hypercorn_server(
            mock_app, "localhost", 8000, None, None
        )

        # Assert
        mock_asyncio_run.assert_called_once()

        # Test with SSL
        ProxyInitializationHelpers._init_hypercorn_server(
            mock_app, "localhost", 8000, "cert.pem", "key.pem"
        )

    @patch("subprocess.Popen")
    def test_run_ollama_serve(self, mock_popen):
        # Execute
        ProxyInitializationHelpers._run_ollama_serve()

        # Assert
        mock_popen.assert_called_once()

        # Test exception handling
        mock_popen.side_effect = Exception("Test exception")
        ProxyInitializationHelpers._run_ollama_serve()  # Should not raise

    @patch("socket.socket")
    def test_is_port_in_use(self, mock_socket):
        # Setup for port in use
        mock_socket_instance = MagicMock()
        mock_socket_instance.connect_ex.return_value = 0
        mock_socket.return_value.__enter__.return_value = mock_socket_instance

        # Execute and Assert
        assert ProxyInitializationHelpers._is_port_in_use(8000) is True

        # Setup for port not in use
        mock_socket_instance.connect_ex.return_value = 1

        # Execute and Assert
        assert ProxyInitializationHelpers._is_port_in_use(8000) is False

    def test_get_loop_type(self):
        # Test on Windows
        with patch("sys.platform", "win32"):
            assert ProxyInitializationHelpers._get_loop_type() is None

        # Test on Linux
        with patch("sys.platform", "linux"):
            assert ProxyInitializationHelpers._get_loop_type() == "uvloop"
