import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path

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

    @patch.dict(os.environ, {}, clear=True)
    def test_database_url_construction_with_special_characters(self):
        # Setup environment variables with special characters that need escaping
        test_env = {
            "DATABASE_HOST": "localhost:5432",
            "DATABASE_USERNAME": "user@with+special",
            "DATABASE_PASSWORD": "pass&word!@#$%",
            "DATABASE_NAME": "db_name/test",
        }

        with patch.dict(os.environ, test_env):
            # Call the relevant function - we'll need to extract the database URL construction logic
            # This is simulating what happens in the run_server function when database_url is None
            import urllib.parse

            from litellm.proxy.proxy_cli import append_query_params

            database_host = os.environ["DATABASE_HOST"]
            database_username = os.environ["DATABASE_USERNAME"]
            database_password = os.environ["DATABASE_PASSWORD"]
            database_name = os.environ["DATABASE_NAME"]

            # Test the URL encoding part
            database_username_enc = urllib.parse.quote_plus(database_username)
            database_password_enc = urllib.parse.quote_plus(database_password)
            database_name_enc = urllib.parse.quote_plus(database_name)

            # Construct DATABASE_URL from the provided variables
            database_url = f"postgresql://{database_username_enc}:{database_password_enc}@{database_host}/{database_name_enc}"

            # Assert the correct URL was constructed with properly escaped characters
            expected_url = "postgresql://user%40with%2Bspecial:pass%26word%21%40%23%24%25@localhost:5432/db_name%2Ftest"
            assert database_url == expected_url

            # Test appending query parameters
            params = {"connection_limit": 10, "pool_timeout": 60}
            modified_url = append_query_params(database_url, params)
            assert "connection_limit=10" in modified_url
            assert "pool_timeout=60" in modified_url

    @patch("uvicorn.run")
    @patch("builtins.print")
    def test_skip_server_startup(self, mock_print, mock_uvicorn_run):
        """Test that the skip_server_startup flag prevents server startup when True"""
        from click.testing import CliRunner

        from litellm.proxy.proxy_cli import run_server

        runner = CliRunner()

        mock_app = MagicMock()
        mock_proxy_config = MagicMock()
        mock_key_mgmt = MagicMock()
        mock_save_worker_config = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "proxy_server": MagicMock(
                    app=mock_app,
                    ProxyConfig=mock_proxy_config,
                    KeyManagementSettings=mock_key_mgmt,
                    save_worker_config=mock_save_worker_config,
                )
            },
        ), patch(
            "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
        ) as mock_get_args:
            mock_get_args.return_value = {
                "app": "litellm.proxy.proxy_server:app",
                "host": "localhost",
                "port": 8000,
            }

            result = runner.invoke(run_server, ["--local", "--skip_server_startup"])

            assert result.exit_code == 0
            mock_uvicorn_run.assert_not_called()
            mock_print.assert_any_call(
                "LiteLLM: Setup complete. Skipping server startup as requested."
            )

            mock_uvicorn_run.reset_mock()
            mock_print.reset_mock()

            result = runner.invoke(run_server, ["--local"])

            assert result.exit_code == 0
            mock_uvicorn_run.assert_called_once()
