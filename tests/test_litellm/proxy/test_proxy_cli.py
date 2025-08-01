import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import fastapi

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path

from litellm.proxy.proxy_cli import ProxyInitializationHelpers
from litellm.proxy.health_endpoints.health_app_factory import build_health_app
import builtins
import types


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

        # Test with keepalive_timeout
        args = ProxyInitializationHelpers._get_default_unvicorn_init_args(
            "localhost", 8000, None, 60
        )
        assert args["timeout_keep_alive"] == 60

        # Test with both log_config and keepalive_timeout
        args = ProxyInitializationHelpers._get_default_unvicorn_init_args(
            "localhost", 8000, "log_config.json", 120
        )
        assert args["log_config"] == "log_config.json"
        assert args["timeout_keep_alive"] == 120

    @patch("asyncio.run")
    @patch("builtins.print")
    def test_init_hypercorn_server(self, mock_print, mock_asyncio_run):
        # Setup
        mock_app = MagicMock()

        # Execute
        ProxyInitializationHelpers._init_hypercorn_server(
            mock_app, "localhost", 8000, None, None, None
        )

        # Assert
        mock_asyncio_run.assert_called_once()

        # Test with SSL
        ProxyInitializationHelpers._init_hypercorn_server(
            mock_app, "localhost", 8000, "cert.pem", "key.pem", "ECDHE"
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

    @patch("uvicorn.run")
    @patch("builtins.print")
    def test_keepalive_timeout_flag(self, mock_print, mock_uvicorn_run):
        """Test that the keepalive_timeout flag is properly passed to uvicorn"""
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
                "timeout_keep_alive": 30,
            }

            result = runner.invoke(run_server, ["--local", "--keepalive_timeout", "30"])

            assert result.exit_code == 0
            mock_get_args.assert_called_once_with(
                host="0.0.0.0",
                port=4000,
                log_config=None,
                keepalive_timeout=30,
            )
            mock_uvicorn_run.assert_called_once()
            
            # Check that the uvicorn.run was called with the timeout_keep_alive parameter
            call_args = mock_uvicorn_run.call_args
            assert call_args[1]["timeout_keep_alive"] == 30

    @patch.dict(os.environ, {}, clear=True)
    def test_construct_database_url_from_env_vars(self):
        """Test the construct_database_url_from_env_vars function with various scenarios"""
        from litellm.proxy.utils import construct_database_url_from_env_vars

        # Test with all required variables present
        test_env = {
            "DATABASE_HOST": "localhost:5432",
            "DATABASE_USERNAME": "testuser",
            "DATABASE_PASSWORD": "testpass",
            "DATABASE_NAME": "testdb",
        }

        with patch.dict(os.environ, test_env):
            result = construct_database_url_from_env_vars()
            expected_url = "postgresql://testuser:testpass@localhost:5432/testdb"
            assert result == expected_url

        # Test with special characters that need URL encoding
        test_env_special = {
            "DATABASE_HOST": "localhost:5432",
            "DATABASE_USERNAME": "user@with+special",
            "DATABASE_PASSWORD": "pass&word!@#$%",
            "DATABASE_NAME": "db_name/test",
        }

        with patch.dict(os.environ, test_env_special):
            result = construct_database_url_from_env_vars()
            expected_url = "postgresql://user%40with%2Bspecial:pass%26word%21%40%23%24%25@localhost:5432/db_name%2Ftest"
            assert result == expected_url

        # Test without password (should still work)
        test_env_no_password = {
            "DATABASE_HOST": "localhost:5432",
            "DATABASE_USERNAME": "testuser",
            "DATABASE_NAME": "testdb",
        }

        with patch.dict(os.environ, test_env_no_password):
            result = construct_database_url_from_env_vars()
            expected_url = "postgresql://testuser@localhost:5432/testdb"
            assert result == expected_url

        # Test with missing required variables (should return None)
        test_env_missing = {
            "DATABASE_HOST": "localhost:5432",
            "DATABASE_USERNAME": "testuser",
            # Missing DATABASE_NAME
        }

        with patch.dict(os.environ, test_env_missing):
            result = construct_database_url_from_env_vars()
            assert result is None

        # Test with empty environment (should return None)
        with patch.dict(os.environ, {}, clear=True):
            result = construct_database_url_from_env_vars()
            assert result is None

    @patch("uvicorn.run")
    @patch("builtins.print")
    def test_run_server_no_config_passed(self, mock_print, mock_uvicorn_run):
        """Test that run_server properly handles the case when no config is passed"""
        from click.testing import CliRunner
        from litellm.proxy.proxy_cli import run_server
        import asyncio

        runner = CliRunner()

        mock_app = MagicMock()
        mock_proxy_config = MagicMock()
        mock_key_mgmt = MagicMock()
        mock_save_worker_config = MagicMock()

        # Mock the ProxyConfig.get_config method to return a proper async config
        async def mock_get_config(config_file_path=None):
            return {
                "general_settings": {},
                "litellm_settings": {}
            }
        
        mock_proxy_config_instance = MagicMock()
        mock_proxy_config_instance.get_config = mock_get_config
        mock_proxy_config.return_value = mock_proxy_config_instance

        # Ensure DATABASE_URL is not set in the environment
        with patch.dict(os.environ, {"DATABASE_URL": ""}, clear=True):
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

                # Test with no config parameter (config=None)
                result = runner.invoke(run_server, ["--local"])

                assert result.exit_code == 0
                
                # Verify that uvicorn.run was called
                mock_uvicorn_run.assert_called_once()

                # Reset mocks for second test
                mock_uvicorn_run.reset_mock()

                # Test with explicit --config None (should behave the same)
                result = runner.invoke(run_server, ["--local", "--config", "None"])

                assert result.exit_code == 0
                
                # Verify that uvicorn.run was called again
                mock_uvicorn_run.assert_called_once()


class TestHealthAppFactory:
    """Test cases for the health app factory module"""
    
    def test_build_health_app(self):
        """Test that build_health_app creates a FastAPI app with the correct title and includes the health router"""
        # Execute
        health_app = build_health_app()
        
        # Assert
        assert health_app.title == "LiteLLM Health Endpoints"
        assert isinstance(health_app, fastapi.FastAPI)
        
        # Verify that the app has the expected health endpoints by checking route paths
        # When a router is included, its routes are flattened into the main app's routes
        route_paths = []
        for route in health_app.routes:
            if hasattr(route, 'path'):
                route_paths.append(route.path)
        
        # Check for some expected health endpoints
        expected_paths = [
            "/test",
            "/health/services", 
            "/health",
            "/health/history",
            "/health/latest",
            "/settings",
            "/active/callbacks",
            "/health/readiness",
            "/health/liveliness",
            "/health/liveness",
            "/health/test_connection"
        ]
        
        # At least some of the expected health endpoints should be present
        found_paths = [path for path in expected_paths if path in route_paths]
        assert len(found_paths) > 0, f"Expected to find health endpoints, but found: {route_paths}"
        
        # Verify that the app has routes (indicating the router was included)
        assert len(health_app.routes) > 0, "Health app should have routes from the included router"
        
    def test_build_health_app_returns_different_instances(self):
        """Test that build_health_app returns different FastAPI instances on each call"""
        # Execute
        health_app_1 = build_health_app()
        health_app_2 = build_health_app()
        
        # Assert
        assert health_app_1 is not health_app_2
        assert health_app_1.title == health_app_2.title
        assert isinstance(health_app_1, fastapi.FastAPI)
        assert isinstance(health_app_2, fastapi.FastAPI)
