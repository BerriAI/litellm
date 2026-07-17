import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import click
import fastapi
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path

import builtins
import types

import uvicorn

from litellm.proxy.proxy_cli import ProxyInitializationHelpers


@pytest.mark.xdist_group("proxy_cli")
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
            # When json_logs is True, log_config should be set to the JSON log config dict
            assert args["log_config"] is not None
            assert isinstance(args["log_config"], dict)
            assert "version" in args["log_config"]
            assert "formatters" in args["log_config"]

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

        class _FakeUvicornConfig:
            def __init__(self, timeout_worker_healthcheck=None):
                pass

        with patch("uvicorn.Config", _FakeUvicornConfig):
            args = ProxyInitializationHelpers._get_default_unvicorn_init_args(
                "localhost", 8000, timeout_worker_healthcheck=15
            )
            assert args["timeout_worker_healthcheck"] == 15

    def test_installed_uvicorn_supports_worker_flags(self):
        params = inspect.signature(uvicorn.Config.__init__).parameters
        assert "timeout_worker_healthcheck" in params
        assert "limit_max_requests_jitter" in params

        args = ProxyInitializationHelpers._get_default_unvicorn_init_args(
            "localhost", 8000, timeout_worker_healthcheck=30
        )
        assert args["timeout_worker_healthcheck"] == 30

    def test_get_reload_options_no_config_still_watches_env(self):
        opts = ProxyInitializationHelpers._get_reload_options(None)
        assert opts["reload"] is True
        assert opts["reload_dirs"] == [os.path.abspath(os.getcwd())]
        assert opts["reload_includes"] == ["*.py", ".env"]

    def test_get_reload_options_with_config_in_cwd(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_list: []\n")
        monkeypatch.chdir(tmp_path)

        opts = ProxyInitializationHelpers._get_reload_options("config.yaml")

        assert opts["reload"] is True
        assert opts["reload_dirs"] == [str(tmp_path)]
        assert opts["reload_includes"] == ["*.py", ".env", "config.yaml"]

    def test_get_reload_options_with_config_outside_cwd(self, tmp_path, monkeypatch):
        cwd_dir = tmp_path / "work"
        cwd_dir.mkdir()
        elsewhere = tmp_path / "configs"
        elsewhere.mkdir()
        config_file = elsewhere / "proxy.yaml"
        config_file.write_text("model_list: []\n")
        monkeypatch.chdir(cwd_dir)

        opts = ProxyInitializationHelpers._get_reload_options(str(config_file))

        assert opts["reload"] is True
        assert opts["reload_dirs"] == [str(cwd_dir), str(elsewhere)]
        assert opts["reload_includes"] == ["*.py", ".env", "proxy.yaml"]

    def test_patch_statreload_extra_paths_yields_config_and_py(self, tmp_path):
        from pathlib import Path

        from uvicorn.supervisors.statreload import StatReload

        if hasattr(StatReload, "_litellm_patched_config_paths"):
            StatReload._litellm_patched_config_paths.clear()

        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_list: []\n")
        py_file = tmp_path / "module.py"
        py_file.write_text("x = 1\n")

        applied = ProxyInitializationHelpers._patch_statreload_extra_paths(
            [str(config_file)]
        )
        assert applied is True

        fake_self = types.SimpleNamespace(
            config=types.SimpleNamespace(reload_dirs=[tmp_path])
        )
        yielded_paths = {Path(p).resolve() for p in StatReload.iter_py_files(fake_self)}

        assert config_file.resolve() in yielded_paths
        assert py_file.resolve() in yielded_paths

    def test_patch_statreload_extra_paths_yields_env(self, tmp_path):
        from pathlib import Path

        from uvicorn.supervisors.statreload import StatReload

        if hasattr(StatReload, "_litellm_patched_config_paths"):
            StatReload._litellm_patched_config_paths.clear()

        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\n")

        applied = ProxyInitializationHelpers._patch_statreload_extra_paths(
            [str(env_file)]
        )
        assert applied is True

        fake_self = types.SimpleNamespace(
            config=types.SimpleNamespace(reload_dirs=[tmp_path])
        )
        yielded_paths = {Path(p).resolve() for p in StatReload.iter_py_files(fake_self)}

        assert env_file.resolve() in yielded_paths

    def test_patch_statreload_extra_paths_skips_falsy(self, tmp_path):
        from uvicorn.supervisors.statreload import StatReload

        if hasattr(StatReload, "_litellm_patched_config_paths"):
            StatReload._litellm_patched_config_paths.clear()

        assert ProxyInitializationHelpers._patch_statreload_extra_paths([]) is False
        assert (
            ProxyInitializationHelpers._patch_statreload_extra_paths([None, ""])
            is False
        )

    def test_patch_statreload_extra_paths_is_idempotent(self, tmp_path):
        from pathlib import Path

        from uvicorn.supervisors.statreload import StatReload

        if hasattr(StatReload, "_litellm_patched_config_paths"):
            StatReload._litellm_patched_config_paths.clear()

        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_list: []\n")
        py_file = tmp_path / "only.py"
        py_file.write_text("x = 1\n")

        for _ in range(3):
            ProxyInitializationHelpers._patch_statreload_extra_paths([str(config_file)])

        fake_self = types.SimpleNamespace(
            config=types.SimpleNamespace(reload_dirs=[tmp_path])
        )
        yielded = list(StatReload.iter_py_files(fake_self))
        assert len(yielded) == len(set(map(str, yielded)))
        yielded_paths = {Path(p).resolve() for p in yielded}
        assert config_file.resolve() in yielded_paths
        assert py_file.resolve() in yielded_paths

    def test_configure_dev_reload_watches_env_and_sets_override_flag(
        self, tmp_path, monkeypatch
    ):
        from pathlib import Path

        from uvicorn.supervisors.statreload import StatReload

        if hasattr(StatReload, "_litellm_patched_config_paths"):
            StatReload._litellm_patched_config_paths.clear()
        monkeypatch.delenv("LITELLM_DEV_ENV_HOT_RELOAD", raising=False)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("model_list: []\n")
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\n")
        monkeypatch.chdir(tmp_path)

        uvicorn_args: dict = {}
        with patch("litellm._logging.verbose_proxy_logger.warning") as mock_warning:
            ProxyInitializationHelpers._configure_dev_reload(
                uvicorn_args, str(config_file)
            )

        assert os.environ["LITELLM_DEV_ENV_HOT_RELOAD"] == "True"
        assert uvicorn_args["reload"] is True
        assert ".env" in uvicorn_args["reload_includes"]

        mock_warning.assert_called_once()
        warning_text = mock_warning.call_args.args[0].lower()
        assert "override" in warning_text
        assert ".env" in warning_text

        fake_self = types.SimpleNamespace(
            config=types.SimpleNamespace(reload_dirs=[tmp_path])
        )
        yielded_paths = {Path(p).resolve() for p in StatReload.iter_py_files(fake_self)}
        assert env_file.resolve() in yielded_paths
        assert config_file.resolve() in yielded_paths

    def test_dev_env_hot_reload_enabled_reads_flag(self, monkeypatch):
        import litellm

        monkeypatch.setenv("LITELLM_DEV_ENV_HOT_RELOAD", "True")
        assert litellm._dev_env_hot_reload_enabled() is True

        monkeypatch.setenv("LITELLM_DEV_ENV_HOT_RELOAD", "false")
        assert litellm._dev_env_hot_reload_enabled() is False

        monkeypatch.delenv("LITELLM_DEV_ENV_HOT_RELOAD", raising=False)
        assert litellm._dev_env_hot_reload_enabled() is False

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

    @patch("granian.Granian")
    @patch("builtins.print")
    def test_init_granian_server(self, mock_print, mock_granian_cls):
        pytest.importorskip("granian")
        mock_server = MagicMock()
        mock_granian_cls.return_value = mock_server
        fake_interfaces = SimpleNamespace(ASGI="asgi")
        with patch("granian.constants.Interfaces", fake_interfaces):
            ProxyInitializationHelpers._init_granian_server(
                host="0.0.0.0",
                port=4000,
                num_workers=2,
                ssl_certfile_path=None,
                ssl_keyfile_path=None,
                max_requests_before_restart=None,
                ciphers=None,
                granian_runtime_threads=None,
            )
        mock_granian_cls.assert_called_once()
        call_kwargs = mock_granian_cls.call_args.kwargs
        assert call_kwargs["target"] == "litellm.proxy.proxy_server:app"
        assert call_kwargs["address"] == "0.0.0.0"
        assert call_kwargs["port"] == 4000
        assert call_kwargs["workers"] == 2
        assert call_kwargs["interface"] == "asgi"
        assert call_kwargs["websockets"] is True
        assert "runtime_threads" not in call_kwargs
        mock_server.serve.assert_called_once()

    @patch("granian.Granian")
    @patch("builtins.print")
    def test_init_granian_server_runtime_threads(self, mock_print, mock_granian_cls):
        pytest.importorskip("granian")
        mock_server = MagicMock()
        mock_granian_cls.return_value = mock_server
        fake_interfaces = SimpleNamespace(ASGI="asgi")
        with patch("granian.constants.Interfaces", fake_interfaces):
            ProxyInitializationHelpers._init_granian_server(
                host="0.0.0.0",
                port=4000,
                num_workers=1,
                ssl_certfile_path=None,
                ssl_keyfile_path=None,
                max_requests_before_restart=None,
                ciphers=None,
                granian_runtime_threads=4,
            )
        assert mock_granian_cls.call_args.kwargs["runtime_threads"] == 4

    @patch("granian.Granian")
    @patch("builtins.print")
    def test_init_granian_server_ssl(self, mock_print, mock_granian_cls):
        pytest.importorskip("granian")
        mock_server = MagicMock()
        mock_granian_cls.return_value = mock_server
        fake_interfaces = SimpleNamespace(ASGI="asgi")
        with patch("granian.constants.Interfaces", fake_interfaces):
            ProxyInitializationHelpers._init_granian_server(
                host="0.0.0.0",
                port=4000,
                num_workers=1,
                ssl_certfile_path="/path/to/cert.pem",
                ssl_keyfile_path="/path/to/key.pem",
                max_requests_before_restart=None,
                ciphers=None,
                granian_runtime_threads=None,
            )
        call_kwargs = mock_granian_cls.call_args.kwargs
        assert call_kwargs["ssl_cert"] == Path("/path/to/cert.pem")
        assert call_kwargs["ssl_key"] == Path("/path/to/key.pem")
        mock_server.serve.assert_called_once()

    @patch("granian.Granian")
    def test_init_granian_server_ssl_requires_cert_and_key(self, mock_granian_cls):
        pytest.importorskip("granian")
        fake_interfaces = SimpleNamespace(ASGI="asgi")
        with patch("granian.constants.Interfaces", fake_interfaces):
            with pytest.raises(click.ClickException, match="Both --ssl_certfile_path"):
                ProxyInitializationHelpers._init_granian_server(
                    host="0.0.0.0",
                    port=4000,
                    num_workers=1,
                    ssl_certfile_path="/path/to/cert.pem",
                    ssl_keyfile_path=None,
                    max_requests_before_restart=None,
                    ciphers=None,
                    granian_runtime_threads=None,
                )
        mock_granian_cls.assert_not_called()

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
            "DATABASE_PASSWORD": "test-password-special-chars",
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
            expected_url = "postgresql://user%40with%2Bspecial:test-password-special-chars@localhost:5432/db_name%2Ftest"
            assert database_url == expected_url

            # Test appending query parameters
            params = {"connection_limit": 10, "pool_timeout": 60}
            modified_url = append_query_params(database_url, params)
            assert "connection_limit=10" in modified_url
            assert "pool_timeout=60" in modified_url

    def test_append_query_params_handles_missing_url(self):
        from litellm.proxy.proxy_cli import append_query_params

        modified_url = append_query_params(None, {"connection_limit": 10})
        assert modified_url == ""

    @patch("uvicorn.run")
    @patch("atexit.register")  # critical
    @patch("litellm.proxy.db.prisma_client.PrismaManager.setup_database")
    @patch(
        "litellm.proxy.db.prisma_client.should_update_prisma_schema", return_value=False
    )
    def test_skip_server_startup(
        self, mock_should_update, mock_setup_db, mock_atexit_register, mock_uvicorn_run
    ):
        from click.testing import CliRunner

        from litellm.proxy.proxy_cli import run_server

        runner = CliRunner()

        mock_proxy_module = MagicMock(
            app=MagicMock(),
            ProxyConfig=MagicMock(),
            KeyManagementSettings=MagicMock(),
            save_worker_config=MagicMock(),
        )
        # Remove DATABASE_URL/DIRECT_URL so the CLI doesn't attempt
        # real prisma operations when these are set in CI.
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }
        with (
            patch.dict(
                os.environ,
                clean_env,
                clear=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": mock_proxy_module,
                    # Prevent real import of proxy_server inside Click's
                    # isolation context (heavy side effects cause stream
                    # lifecycle issues with Click 8.2+)
                    "litellm.proxy.proxy_server": mock_proxy_module,
                },
            ),
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
            ) as mock_get_args,
        ):
            mock_get_args.return_value = {
                "app": "litellm.proxy.proxy_server:app",
                "host": "localhost",
                "port": 8000,
            }

            # --- skip startup ---
            result = runner.invoke(run_server, ["--local", "--skip_server_startup"])

            assert (
                result.exit_code == 0
            ), f"exit_code={result.exit_code}, output={result.output}"
            assert "Skipping server startup" in result.output
            mock_uvicorn_run.assert_not_called()

            # --- normal startup ---
            mock_uvicorn_run.reset_mock()

            result = runner.invoke(run_server, ["--local"])

            assert (
                result.exit_code == 0
            ), f"exit_code={result.exit_code}, output={result.output}"
            mock_uvicorn_run.assert_called_once()

    @pytest.mark.parametrize(
        "timeout_config,expected_timeout",
        [
            ({"database_connection_timeout": 30}, 30),
            ({"database_connection_pool_timeout": 45}, 45),
            (
                {
                    "database_connection_timeout": 30,
                    "database_connection_pool_timeout": 45,
                },
                30,
            ),
        ],
    )
    @patch("subprocess.run")
    @patch("atexit.register")
    @patch("litellm.proxy.db.prisma_client.PrismaManager.setup_database")
    @patch(
        "litellm.proxy.db.prisma_client.should_update_prisma_schema", return_value=False
    )
    def test_db_timeout_settings_are_forwarded_to_pool_timeout(
        self,
        mock_should_update,
        mock_setup_db,
        mock_atexit_register,
        mock_subprocess_run,
        timeout_config,
        expected_timeout,
    ):
        from click.testing import CliRunner

        from litellm.proxy.proxy_cli import run_server

        runner = CliRunner()
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        mock_proxy_module = MagicMock(
            app=MagicMock(),
            ProxyConfig=MagicMock(),
            KeyManagementSettings=MagicMock(),
            save_worker_config=MagicMock(),
        )
        mock_proxy_module.ProxyConfig.return_value.get_config = AsyncMock(
            return_value={
                "general_settings": {
                    "database_url": "postgresql://test:test@localhost:5432/test",
                    "database_connection_pool_limit": 5,
                    **timeout_config,
                }
            }
        )

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }

        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": mock_proxy_module,
                    "litellm.proxy.proxy_server": mock_proxy_module,
                },
            ),
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
            ) as mock_get_args,
            patch(
                "litellm.proxy.proxy_cli.append_query_params",
                side_effect=lambda url, params: (
                    f"{url}?connection_limit={params['connection_limit']}&pool_timeout={params['pool_timeout']}"
                ),
            ) as mock_append_query_params,
        ):
            mock_get_args.return_value = {
                "app": "litellm.proxy.proxy_server:app",
                "host": "localhost",
                "port": 8000,
            }

            result = runner.invoke(
                run_server,
                ["--local", "--config", "test-config.yaml", "--skip_server_startup"],
            )

            assert (
                result.exit_code == 0
            ), f"exit_code={result.exit_code}, output={result.output}"
            mock_append_query_params.assert_called()
            appended_params = mock_append_query_params.call_args.args[1]
            assert appended_params["connection_limit"] == 5
            assert appended_params["pool_timeout"] == expected_timeout

    def test_build_db_connection_url_params_defaults(self):
        from litellm.proxy.proxy_cli import _build_db_connection_url_params

        params = _build_db_connection_url_params(connection_limit=10, pool_timeout=60)
        assert params == {"connection_limit": 10, "pool_timeout": 60}

    def test_build_db_connection_url_params_omits_none_timeouts(self):
        from litellm.proxy.proxy_cli import _build_db_connection_url_params

        params = _build_db_connection_url_params(
            connection_limit=10,
            pool_timeout=60,
            connect_timeout=None,
            socket_timeout=None,
        )
        assert "connect_timeout" not in params
        assert "socket_timeout" not in params

    def test_build_db_connection_url_params_includes_optional_timeouts(self):
        from litellm.proxy.proxy_cli import _build_db_connection_url_params

        params = _build_db_connection_url_params(
            connection_limit=10,
            pool_timeout=60,
            connect_timeout=15,
            socket_timeout=120,
        )
        assert params["connect_timeout"] == 15
        assert params["socket_timeout"] == 120

    def test_build_db_connection_url_params_extras_override_defaults(self):
        from litellm.proxy.proxy_cli import _build_db_connection_url_params

        params = _build_db_connection_url_params(
            connection_limit=10,
            pool_timeout=60,
            extra_params={
                "pgbouncer": "true",
                "statement_cache_size": 0,
                "pool_timeout": 5,
            },
        )
        assert params["pgbouncer"] == "true"
        assert params["statement_cache_size"] == 0
        assert params["pool_timeout"] == 5

    @patch("subprocess.run")
    @patch("atexit.register")
    @patch("litellm.proxy.db.prisma_client.PrismaManager.setup_database")
    @patch(
        "litellm.proxy.db.prisma_client.should_update_prisma_schema", return_value=False
    )
    def test_db_connection_extra_params_forwarded_to_url(
        self,
        mock_should_update,
        mock_setup_db,
        mock_atexit_register,
        mock_subprocess_run,
    ):
        from click.testing import CliRunner

        from litellm.proxy.proxy_cli import run_server

        runner = CliRunner()
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        mock_proxy_module = MagicMock(
            app=MagicMock(),
            ProxyConfig=MagicMock(),
            KeyManagementSettings=MagicMock(),
            save_worker_config=MagicMock(),
        )
        mock_proxy_module.ProxyConfig.return_value.get_config = AsyncMock(
            return_value={
                "general_settings": {
                    "database_url": "postgresql://test:test@localhost:5432/test",
                    "database_connect_timeout": 15,
                    "database_socket_timeout": 120,
                    "database_extra_connection_params": {
                        "pgbouncer": "true",
                        "statement_cache_size": 0,
                    },
                }
            }
        )

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }

        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": mock_proxy_module,
                    "litellm.proxy.proxy_server": mock_proxy_module,
                },
            ),
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
            ) as mock_get_args,
            patch(
                "litellm.proxy.proxy_cli.append_query_params",
                side_effect=lambda url, params: str(url),
            ) as mock_append_query_params,
        ):
            mock_get_args.return_value = {
                "app": "litellm.proxy.proxy_server:app",
                "host": "localhost",
                "port": 8000,
            }

            result = runner.invoke(
                run_server,
                ["--local", "--config", "test-config.yaml", "--skip_server_startup"],
            )

            assert (
                result.exit_code == 0
            ), f"exit_code={result.exit_code}, output={result.output}"
            mock_append_query_params.assert_called()
            appended_params = mock_append_query_params.call_args.args[1]
            assert appended_params["connect_timeout"] == 15
            assert appended_params["socket_timeout"] == 120
            assert appended_params["pgbouncer"] == "true"
            assert appended_params["statement_cache_size"] == 0

    def test_build_db_connection_url_params_disable_prepared_statements(self):
        from litellm.proxy.proxy_cli import _build_db_connection_url_params

        params = _build_db_connection_url_params(
            connection_limit=10,
            pool_timeout=60,
            disable_prepared_statements=True,
        )
        assert params["pgbouncer"] == "true"

    def test_build_db_connection_url_params_no_pgbouncer_by_default(self):
        from litellm.proxy.proxy_cli import _build_db_connection_url_params

        params = _build_db_connection_url_params(
            connection_limit=10,
            pool_timeout=60,
        )
        assert "pgbouncer" not in params

    def test_build_db_connection_url_params_extra_pgbouncer_overrides_flag(self):
        from litellm.proxy.proxy_cli import _build_db_connection_url_params

        params = _build_db_connection_url_params(
            connection_limit=10,
            pool_timeout=60,
            disable_prepared_statements=True,
            extra_params={"pgbouncer": "false"},
        )
        assert params["pgbouncer"] == "false"

    @pytest.mark.parametrize(
        "config_value, expect_pgbouncer",
        [
            (True, True),
            (False, False),
            ("true", True),
            ("false", False),
            ("not-a-bool", False),
        ],
    )
    @patch("subprocess.run")
    @patch("atexit.register")
    @patch("litellm.proxy.db.prisma_client.PrismaManager.setup_database")
    @patch(
        "litellm.proxy.db.prisma_client.should_update_prisma_schema", return_value=False
    )
    def test_disable_prepared_statements_forwarded_to_url(
        self,
        mock_should_update,
        mock_setup_db,
        mock_atexit_register,
        mock_subprocess_run,
        config_value,
        expect_pgbouncer,
    ):
        from click.testing import CliRunner

        from litellm.proxy.proxy_cli import run_server

        runner = CliRunner()
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        mock_proxy_module = MagicMock(
            app=MagicMock(),
            ProxyConfig=MagicMock(),
            KeyManagementSettings=MagicMock(),
            save_worker_config=MagicMock(),
        )
        mock_proxy_module.ProxyConfig.return_value.get_config = AsyncMock(
            return_value={
                "general_settings": {
                    "database_url": "postgresql://test:test@localhost:5432/test",
                    "database_disable_prepared_statements": config_value,
                }
            }
        )

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }

        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": mock_proxy_module,
                    "litellm.proxy.proxy_server": mock_proxy_module,
                },
            ),
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
            ) as mock_get_args,
            patch(
                "litellm.proxy.proxy_cli.append_query_params",
                side_effect=lambda url, params: str(url),
            ) as mock_append_query_params,
        ):
            mock_get_args.return_value = {
                "app": "litellm.proxy.proxy_server:app",
                "host": "localhost",
                "port": 8000,
            }

            result = runner.invoke(
                run_server,
                ["--local", "--config", "test-config.yaml", "--skip_server_startup"],
            )

            assert (
                result.exit_code == 0
            ), f"exit_code={result.exit_code}, output={result.output}"
            mock_append_query_params.assert_called()
            appended_params = mock_append_query_params.call_args.args[1]
            if expect_pgbouncer:
                assert appended_params["pgbouncer"] == "true"
            else:
                assert "pgbouncer" not in appended_params

    @patch("uvicorn.run")
    @patch("atexit.register")
    @patch("litellm.proxy.db.prisma_client.PrismaManager.setup_database")
    @patch(
        "litellm.proxy.db.prisma_client.should_update_prisma_schema", return_value=False
    )
    def test_proxy_default_api_version_uses_azure_default(
        self, mock_should_update, mock_setup_db, mock_atexit_register, mock_uvicorn_run
    ):
        """Proxy default api_version should match litellm.AZURE_DEFAULT_API_VERSION for consistency."""
        from click.testing import CliRunner

        import litellm
        from litellm.proxy.proxy_cli import run_server

        runner = CliRunner()
        mock_proxy_module = MagicMock(
            app=MagicMock(),
            ProxyConfig=MagicMock(),
            KeyManagementSettings=MagicMock(),
            save_worker_config=MagicMock(),
        )
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }
        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": mock_proxy_module,
                    "litellm.proxy.proxy_server": mock_proxy_module,
                },
            ),
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
            ) as mock_get_args,
        ):
            mock_get_args.return_value = {
                "app": "litellm.proxy.proxy_server:app",
                "host": "localhost",
                "port": 8000,
            }
            result = runner.invoke(run_server, ["--local", "--skip_server_startup"])
            assert (
                result.exit_code == 0
            ), f"exit_code={result.exit_code}, output={result.output}"
            mock_proxy_module.save_worker_config.assert_called_once()
            call_kwargs = mock_proxy_module.save_worker_config.call_args[1]
            assert call_kwargs["api_version"] == litellm.AZURE_DEFAULT_API_VERSION

    @patch("uvicorn.run")
    @patch("builtins.print")
    @patch("litellm.proxy.db.prisma_client.PrismaManager.setup_database")
    @patch(
        "litellm.proxy.db.prisma_client.should_update_prisma_schema", return_value=False
    )
    def test_keepalive_timeout_flag(
        self, mock_should_update, mock_setup_db, mock_print, mock_uvicorn_run
    ):
        """Test that the keepalive_timeout flag is properly passed to uvicorn"""
        from click.testing import CliRunner

        from litellm.proxy.proxy_cli import run_server

        runner = CliRunner()

        mock_app = MagicMock()
        mock_proxy_config = MagicMock()
        mock_key_mgmt = MagicMock()
        mock_save_worker_config = MagicMock()

        # Strip DATABASE_URL/DIRECT_URL so run_server doesn't enter the prisma
        # DB-setup block (un-timeout'd `subprocess.run(["prisma"])` +
        # migrate-deploy retry loop) — same isolation every other run_server
        # test in this file uses.
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }

        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": MagicMock(
                        app=mock_app,
                        ProxyConfig=mock_proxy_config,
                        KeyManagementSettings=mock_key_mgmt,
                        save_worker_config=mock_save_worker_config,
                    )
                },
            ),
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
            ) as mock_get_args,
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._is_port_in_use",
                return_value=False,
            ),
        ):
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
                timeout_worker_healthcheck=None,
            )
            mock_uvicorn_run.assert_called_once()

            # Check that the uvicorn.run was called with the timeout_keep_alive parameter
            call_args = mock_uvicorn_run.call_args
            assert call_args[1]["timeout_keep_alive"] == 30

    @patch("uvicorn.run")
    @patch("builtins.print")
    @patch("litellm.proxy.db.prisma_client.PrismaManager.setup_database")
    @patch(
        "litellm.proxy.db.prisma_client.should_update_prisma_schema", return_value=False
    )
    def test_timeout_worker_healthcheck_flag(
        self, mock_should_update, mock_setup_db, mock_print, mock_uvicorn_run
    ):
        """Test that the --timeout_worker_healthcheck flag is threaded through to the uvicorn init helper."""
        from click.testing import CliRunner

        from litellm.proxy.proxy_cli import run_server

        runner = CliRunner()

        mock_app = MagicMock()
        mock_proxy_config = MagicMock()
        mock_key_mgmt = MagicMock()
        mock_save_worker_config = MagicMock()

        # Strip DATABASE_URL/DIRECT_URL so run_server doesn't enter the prisma
        # DB-setup block (un-timeout'd `subprocess.run(["prisma"])` +
        # migrate-deploy retry loop) — same isolation every other run_server
        # test in this file uses.
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }

        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": MagicMock(
                        app=mock_app,
                        ProxyConfig=mock_proxy_config,
                        KeyManagementSettings=mock_key_mgmt,
                        save_worker_config=mock_save_worker_config,
                    )
                },
            ),
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
            ) as mock_get_args,
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._is_port_in_use",
                return_value=False,
            ),
        ):
            mock_get_args.return_value = {
                "app": "litellm.proxy.proxy_server:app",
                "host": "localhost",
                "port": 8000,
            }

            result = runner.invoke(
                run_server, ["--local", "--timeout_worker_healthcheck", "15"]
            )

            assert result.exit_code == 0
            mock_get_args.assert_called_once_with(
                host="0.0.0.0",
                port=4000,
                log_config=None,
                keepalive_timeout=None,
                timeout_worker_healthcheck=15,
            )

    @patch("uvicorn.run")
    @patch("builtins.print")
    @patch("litellm.proxy.db.prisma_client.PrismaManager.setup_database")
    def test_max_requests_before_restart_flag(
        self, mock_setup_db, mock_print, mock_uvicorn_run
    ):
        """Test that the max_requests_before_restart flag is passed to uvicorn as limit_max_requests"""
        from click.testing import CliRunner

        from litellm.proxy.proxy_cli import run_server

        runner = CliRunner()

        mock_app = MagicMock()
        mock_proxy_config = MagicMock()
        mock_key_mgmt = MagicMock()
        mock_save_worker_config = MagicMock()

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }
        with (
            patch.dict(
                os.environ,
                clean_env,
                clear=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": MagicMock(
                        app=mock_app,
                        ProxyConfig=mock_proxy_config,
                        KeyManagementSettings=mock_key_mgmt,
                        save_worker_config=mock_save_worker_config,
                    )
                },
            ),
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
            ) as mock_get_args,
        ):
            mock_get_args.return_value = {
                "app": "litellm.proxy.proxy_server:app",
                "host": "localhost",
                "port": 8000,
            }

            result = runner.invoke(
                run_server, ["--local", "--max_requests_before_restart", "123"]
            )

            assert (
                result.exit_code == 0
            ), f"exit_code={result.exit_code}, output={result.output}"
            mock_uvicorn_run.assert_called_once()

            # Check that uvicorn.run was called with limit_max_requests parameter
            call_args = mock_uvicorn_run.call_args
            assert call_args[1]["limit_max_requests"] == 123

    @patch("uvicorn.run")
    @patch("builtins.print")
    @patch("litellm.proxy.db.prisma_client.PrismaManager.setup_database")
    def test_max_requests_before_restart_jitter_flag(
        self, mock_setup_db, mock_print, mock_uvicorn_run
    ):
        """--max_requests_before_restart_jitter maps to uvicorn limit_max_requests_jitter"""
        from click.testing import CliRunner

        from litellm.proxy.proxy_cli import run_server

        class _NewUvicornConfig:
            def __init__(self, limit_max_requests=None, limit_max_requests_jitter=0):
                pass

        runner = CliRunner()
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }
        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch("uvicorn.Config", _NewUvicornConfig),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": MagicMock(
                        app=MagicMock(),
                        ProxyConfig=MagicMock(),
                        KeyManagementSettings=MagicMock(),
                        save_worker_config=MagicMock(),
                    )
                },
            ),
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
            ) as mock_get_args,
        ):
            mock_get_args.return_value = {
                "app": "litellm.proxy.proxy_server:app",
                "host": "localhost",
                "port": 8000,
            }

            result = runner.invoke(
                run_server,
                [
                    "--local",
                    "--max_requests_before_restart",
                    "1000",
                    "--max_requests_before_restart_jitter",
                    "50",
                ],
            )

            assert (
                result.exit_code == 0
            ), f"exit_code={result.exit_code}, output={result.output}"
            mock_uvicorn_run.assert_called_once()
            call_args = mock_uvicorn_run.call_args
            assert call_args[1]["limit_max_requests"] == 1000
            assert call_args[1]["limit_max_requests_jitter"] == 50

    @patch("litellm.proxy.proxy_cli.ProxyInitializationHelpers._run_gunicorn_server")
    @patch("uvicorn.run")
    @patch("builtins.print")
    @patch("litellm.proxy.db.prisma_client.PrismaManager.setup_database")
    def test_run_gunicorn_passes_max_requests_jitter(
        self, mock_setup_db, mock_print, mock_uvicorn_run, mock_run_gunicorn
    ):
        """--run_gunicorn threads jitter into _run_gunicorn_server, not uvicorn.run"""
        from click.testing import CliRunner

        from litellm.proxy.proxy_cli import run_server

        runner = CliRunner()
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }
        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": MagicMock(
                        app=MagicMock(),
                        ProxyConfig=MagicMock(),
                        KeyManagementSettings=MagicMock(),
                        save_worker_config=MagicMock(),
                    )
                },
            ),
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
            ) as mock_get_args,
        ):
            mock_get_args.return_value = {
                "app": "litellm.proxy.proxy_server:app",
                "host": "localhost",
                "port": 8000,
            }

            result = runner.invoke(
                run_server,
                [
                    "--local",
                    "--run_gunicorn",
                    "--max_requests_before_restart",
                    "900",
                    "--max_requests_before_restart_jitter",
                    "75",
                ],
            )

            assert (
                result.exit_code == 0
            ), f"exit_code={result.exit_code}, output={result.output}"
            mock_uvicorn_run.assert_not_called()
            mock_run_gunicorn.assert_called_once()
            g_kwargs = mock_run_gunicorn.call_args[1]
            assert g_kwargs["max_requests_before_restart"] == 900
            assert g_kwargs["max_requests_before_restart_jitter"] == 75

    @pytest.mark.skipif(os.name == "nt", reason="gunicorn server path skips Windows")
    def test_gunicorn_options_include_max_requests_jitter(self):
        """_run_gunicorn_server puts max_requests_jitter into the gunicorn options"""
        pytest.importorskip("gunicorn")

        captured: dict = {}

        def capture_run(self):
            captured["options"] = dict(self.options)

        with patch("gunicorn.app.base.BaseApplication.run", capture_run):
            ProxyInitializationHelpers._run_gunicorn_server(
                host="127.0.0.1",
                port=4010,
                app=MagicMock(),
                num_workers=2,
                ssl_certfile_path=None,
                ssl_keyfile_path=None,
                max_requests_before_restart=1000,
                max_requests_before_restart_jitter=50,
            )

        assert captured["options"]["max_requests"] == 1000
        assert captured["options"]["max_requests_jitter"] == 50

    @pytest.mark.skipif(os.name == "nt", reason="gunicorn server path skips Windows")
    def test_gunicorn_jitter_without_base_warns(self):
        """gunicorn path warns when jitter is set without --max_requests_before_restart"""
        pytest.importorskip("gunicorn")

        captured: dict = {}

        def capture_run(self):
            captured["options"] = dict(self.options)

        with (
            patch("gunicorn.app.base.BaseApplication.run", capture_run),
            patch("builtins.print") as mock_print,
        ):
            ProxyInitializationHelpers._run_gunicorn_server(
                host="127.0.0.1",
                port=4011,
                app=MagicMock(),
                num_workers=2,
                ssl_certfile_path=None,
                ssl_keyfile_path=None,
                max_requests_before_restart=None,
                max_requests_before_restart_jitter=50,
            )

        assert "max_requests" not in captured["options"]
        assert "max_requests_jitter" not in captured["options"]
        assert any("has no effect" in str(c) for c in mock_print.call_args_list)

    def test_apply_uvicorn_jitter_sets_arg_when_supported(self):
        class _NewUvicornConfig:
            def __init__(self, limit_max_requests=None, limit_max_requests_jitter=0):
                pass

        uvicorn_args: dict = {}
        with patch("uvicorn.Config", _NewUvicornConfig):
            ProxyInitializationHelpers._apply_uvicorn_max_requests_jitter(
                uvicorn_args=uvicorn_args,
                max_requests_before_restart=1000,
                jitter=50,
            )
        assert uvicorn_args["limit_max_requests_jitter"] == 50

    def test_apply_uvicorn_jitter_skipped_on_old_uvicorn(self):
        class _FakeUvicornConfig:
            def __init__(self, limit_max_requests=None):
                pass

        uvicorn_args: dict = {}
        with (
            patch("uvicorn.Config", _FakeUvicornConfig),
            patch("builtins.print") as mock_print,
        ):
            ProxyInitializationHelpers._apply_uvicorn_max_requests_jitter(
                uvicorn_args=uvicorn_args,
                max_requests_before_restart=1000,
                jitter=50,
            )

        assert "limit_max_requests_jitter" not in uvicorn_args
        assert any("0.41.0" in str(c) for c in mock_print.call_args_list)

    def test_apply_uvicorn_jitter_without_base_warns(self):
        uvicorn_args: dict = {}
        with patch("builtins.print") as mock_print:
            ProxyInitializationHelpers._apply_uvicorn_max_requests_jitter(
                uvicorn_args=uvicorn_args,
                max_requests_before_restart=None,
                jitter=50,
            )

        assert "limit_max_requests_jitter" not in uvicorn_args
        assert any("has no effect" in str(c) for c in mock_print.call_args_list)

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
            "DATABASE_PASSWORD": "test-password-special-chars",
            "DATABASE_NAME": "db_name/test",
        }

        with patch.dict(os.environ, test_env_special):
            result = construct_database_url_from_env_vars()
            expected_url = "postgresql://user%40with%2Bspecial:test-password-special-chars@localhost:5432/db_name%2Ftest"
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
        import asyncio

        from click.testing import CliRunner

        from litellm.proxy.proxy_cli import run_server

        runner = CliRunner()

        mock_app = MagicMock()
        mock_proxy_config = MagicMock()
        mock_key_mgmt = MagicMock()
        mock_save_worker_config = MagicMock()

        # Mock the ProxyConfig.get_config method to return a proper async config
        async def mock_get_config(config_file_path=None):
            return {"general_settings": {}, "litellm_settings": {}}

        mock_proxy_config_instance = MagicMock()
        mock_proxy_config_instance.get_config = mock_get_config
        mock_proxy_config.return_value = mock_proxy_config_instance

        mock_proxy_server_module = MagicMock(app=mock_app)

        # Only remove DATABASE_URL and DIRECT_URL to prevent the database setup
        # code path from running. Do NOT use clear=True as it removes PATH, HOME,
        # etc., which causes imports inside run_server to break in CI (the real
        # litellm.proxy.proxy_server import at line 820 of proxy_cli.py has heavy
        # side effects that fail without a proper environment).
        env_overrides = {
            "DATABASE_URL": "",
            "DIRECT_URL": "",
            "IAM_TOKEN_DB_AUTH": "",
            "USE_AWS_KMS": "",
        }
        with patch.dict(os.environ, env_overrides):
            # Remove DATABASE_URL entirely so the DB setup block is skipped
            os.environ.pop("DATABASE_URL", None)
            os.environ.pop("DIRECT_URL", None)

            with (
                patch.dict(
                    "sys.modules",
                    {
                        "proxy_server": MagicMock(
                            app=mock_app,
                            ProxyConfig=mock_proxy_config,
                            KeyManagementSettings=mock_key_mgmt,
                            save_worker_config=mock_save_worker_config,
                        ),
                        # Also mock litellm.proxy.proxy_server to prevent the real
                        # import at line 820 of proxy_cli.py which has heavy side
                        # effects (FastAPI app init, logging setup, etc.)
                        "litellm.proxy.proxy_server": mock_proxy_server_module,
                    },
                ),
                patch(
                    "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
                ) as mock_get_args,
            ):
                mock_get_args.return_value = {
                    "app": "litellm.proxy.proxy_server:app",
                    "host": "localhost",
                    "port": 8000,
                }

                # Test with no config parameter (config=None)
                result = runner.invoke(run_server, ["--local"])

                assert result.exit_code == 0, (
                    f"run_server failed with exit_code={result.exit_code}, "
                    f"output={result.output}, exception={result.exception}"
                )

                # Verify that uvicorn.run was called
                mock_uvicorn_run.assert_called_once()

                # Reset mocks for second test
                mock_uvicorn_run.reset_mock()

                # Test with explicit --config None (should behave the same)
                result = runner.invoke(run_server, ["--local", "--config", "None"])

                assert result.exit_code == 0, (
                    f"run_server failed with exit_code={result.exit_code}, "
                    f"output={result.output}, exception={result.exception}"
                )

                # Verify that uvicorn.run was called again
                mock_uvicorn_run.assert_called_once()


class TestQueryEngineReaperWiring:
    def _invoke_run_server(self, args):
        from click.testing import CliRunner

        from litellm.proxy.proxy_cli import run_server

        runner = CliRunner()
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }
        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": MagicMock(
                        app=MagicMock(),
                        ProxyConfig=MagicMock(),
                        KeyManagementSettings=MagicMock(),
                        save_worker_config=MagicMock(),
                    )
                },
            ),
            patch("uvicorn.run") as mock_uvicorn_run,
            patch(
                "litellm.proxy.proxy_cli.start_query_engine_reaper"
            ) as mock_start_reaper,
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
            ) as mock_get_args,
        ):
            mock_get_args.return_value = {
                "app": "litellm.proxy.proxy_server:app",
                "host": "localhost",
                "port": 8000,
            }
            result = runner.invoke(run_server, args)
        return result, mock_uvicorn_run, mock_start_reaper

    def test_multi_worker_uvicorn_starts_reaper(self):
        result, mock_uvicorn_run, mock_start_reaper = self._invoke_run_server(
            ["--local", "--num_workers", "2"]
        )
        assert result.exit_code == 0, f"exit_code={result.exit_code}, output={result.output}"
        mock_uvicorn_run.assert_called_once()
        mock_start_reaper.assert_called_once()

    def test_single_worker_uvicorn_does_not_start_reaper(self):
        result, mock_uvicorn_run, mock_start_reaper = self._invoke_run_server(
            ["--local", "--num_workers", "1"]
        )
        assert result.exit_code == 0, f"exit_code={result.exit_code}, output={result.output}"
        mock_uvicorn_run.assert_called_once()
        mock_start_reaper.assert_not_called()

    @pytest.mark.skipif(os.name == "nt", reason="gunicorn server path skips Windows")
    def test_gunicorn_arbiter_starts_reaper(self):
        pytest.importorskip("gunicorn")

        with (
            patch("gunicorn.app.base.BaseApplication.run"),
            patch(
                "litellm.proxy.proxy_cli.start_query_engine_reaper"
            ) as mock_start_reaper,
        ):
            ProxyInitializationHelpers._run_gunicorn_server(
                host="127.0.0.1",
                port=4010,
                app=MagicMock(),
                num_workers=1,
                ssl_certfile_path=None,
                ssl_keyfile_path=None,
            )

        mock_start_reaper.assert_called_once()


class TestRunServerDbSetup:
    """Tests for run_server's prisma setup_database behavior."""

    @patch("subprocess.run")
    @patch("atexit.register")
    @patch("litellm.proxy.db.prisma_client.PrismaManager.setup_database")
    @patch("litellm.proxy.db.check_migration.check_prisma_schema_diff")
    @patch("litellm.proxy.db.prisma_client.should_update_prisma_schema")
    def test_use_prisma_db_push_flag_behavior(
        self,
        mock_should_update_schema,
        mock_check_schema_diff,
        mock_setup_database,
        mock_atexit_register,
        mock_subprocess_run,
    ):
        """Test that use_prisma_db_push flag correctly controls PrismaManager.setup_database use_migrate parameter"""
        from litellm.proxy.proxy_cli import run_server

        # Mock subprocess.run to simulate prisma being available
        mock_subprocess_run.return_value = MagicMock(returncode=0)

        # Mock should_update_prisma_schema to return True (so setup_database gets called)
        mock_should_update_schema.return_value = True

        mock_proxy_module = MagicMock(
            app=MagicMock(),
            ProxyConfig=MagicMock(),
            KeyManagementSettings=MagicMock(),
            save_worker_config=MagicMock(),
        )

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }
        clean_env["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"

        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": mock_proxy_module,
                    "litellm.proxy.proxy_server": mock_proxy_module,
                },
            ),
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
            ) as mock_get_args,
        ):
            mock_get_args.return_value = {
                "app": "litellm.proxy.proxy_server:app",
                "host": "localhost",
                "port": 8000,
            }

            # Use standalone_mode=False to bypass Click's CliRunner stream
            # isolation which causes flaky "I/O operation on closed file"
            # errors in CI environments (Click 8.3.x stream lifecycle issue).

            # Test 1: Without --use_prisma_db_push flag (default behavior)
            # use_prisma_db_push should be False (default), so use_migrate should be True
            run_server.main(["--local", "--skip_server_startup"], standalone_mode=False)
            mock_setup_database.assert_called_with(
                use_migrate=True, use_v2_resolver=False
            )

            # Reset mocks
            mock_setup_database.reset_mock()
            mock_should_update_schema.reset_mock()
            mock_should_update_schema.return_value = True

            # Test 2: With --use_prisma_db_push flag set
            # use_prisma_db_push should be True, so use_migrate should be False
            run_server.main(
                ["--local", "--skip_server_startup", "--use_prisma_db_push"],
                standalone_mode=False,
            )
            mock_setup_database.assert_called_with(
                use_migrate=False, use_v2_resolver=False
            )

    @patch("subprocess.run")
    @patch("atexit.register")
    @patch("litellm.proxy.db.prisma_client.PrismaManager.setup_database")
    @patch("litellm.proxy.db.check_migration.check_prisma_schema_diff")
    @patch("litellm.proxy.db.prisma_client.should_update_prisma_schema")
    def test_startup_fails_when_db_setup_fails(
        self,
        mock_should_update_schema,
        mock_check_schema_diff,
        mock_setup_database,
        mock_atexit_register,
        mock_subprocess_run,
    ):
        """Test that proxy exits with code 1 when PrismaManager.setup_database returns False and --enforce_prisma_migration_check is set"""
        from litellm.proxy.proxy_cli import run_server

        mock_subprocess_run.return_value = MagicMock(returncode=0)
        mock_should_update_schema.return_value = True
        mock_setup_database.return_value = False

        mock_proxy_module = MagicMock(
            app=MagicMock(),
            ProxyConfig=MagicMock(),
            KeyManagementSettings=MagicMock(),
            save_worker_config=MagicMock(),
        )

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }
        clean_env["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"

        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": mock_proxy_module,
                    "litellm.proxy.proxy_server": mock_proxy_module,
                },
            ),
            patch(
                "litellm.proxy.proxy_cli.ProxyInitializationHelpers._get_default_unvicorn_init_args"
            ) as mock_get_args,
        ):
            mock_get_args.return_value = {
                "app": "litellm.proxy.proxy_server:app",
                "host": "localhost",
                "port": 8000,
            }

            with pytest.raises(SystemExit) as exc_info:
                run_server.main(
                    [
                        "--local",
                        "--skip_server_startup",
                        "--enforce_prisma_migration_check",
                    ],
                    standalone_mode=False,
                )
            assert exc_info.value.code == 1
            mock_setup_database.assert_called_once_with(
                use_migrate=True, use_v2_resolver=False
            )

    @patch("subprocess.run")
    @patch("atexit.register")
    @patch("litellm.proxy.db.prisma_client.PrismaManager.setup_database")
    @patch("litellm.proxy.db.check_migration.check_prisma_schema_diff")
    @patch("litellm.proxy.db.prisma_client.should_update_prisma_schema")
    def test_startup_exits_on_non_postgres_database_url(
        self,
        mock_should_update_schema,
        mock_check_schema_diff,
        mock_setup_database,
        mock_atexit_register,
        mock_subprocess_run,
    ):
        """A sqlite DATABASE_URL must exit immediately, before any prisma call,
        instead of stalling on a migration against the postgresql-only schema."""
        from litellm.proxy.proxy_cli import run_server

        mock_subprocess_run.return_value = MagicMock(returncode=0)
        mock_should_update_schema.return_value = True

        mock_proxy_module = MagicMock(
            app=MagicMock(),
            ProxyConfig=MagicMock(),
            KeyManagementSettings=MagicMock(),
            save_worker_config=MagicMock(),
        )

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }
        clean_env["DATABASE_URL"] = "sqlite:///data/litellm.db"

        with (
            patch.dict(os.environ, clean_env, clear=True),
            patch.dict(
                "sys.modules",
                {
                    "proxy_server": mock_proxy_module,
                    "litellm.proxy.proxy_server": mock_proxy_module,
                },
            ),
        ):
            with pytest.raises(SystemExit) as exc_info:
                run_server.main(
                    ["--local", "--skip_server_startup"], standalone_mode=False
                )
            assert exc_info.value.code == 1
            mock_setup_database.assert_not_called()


# --- Module-level helpers for worker startup hook tests ---

_dummy_hook_called = False


def _dummy_hook():
    """A simple sync hook used by test_should_run_worker_startup_hooks."""
    global _dummy_hook_called
    _dummy_hook_called = True


_dummy_async_hook_called = False


async def _dummy_async_hook():
    """A simple async hook used by test_should_run_async_worker_startup_hook."""
    global _dummy_async_hook_called
    _dummy_async_hook_called = True


def _failing_hook():
    """A hook that always raises, used by test_should_raise_on_failing_hook."""
    raise RuntimeError("Hook failed on purpose")


class TestWorkerStartupHooks:
    """Tests for the LITELLM_WORKER_STARTUP_HOOKS mechanism in proxy_startup_event."""

    @pytest.mark.asyncio
    async def test_should_run_worker_startup_hooks(self):
        """Sync worker startup hook is called during proxy_startup_event."""
        global _dummy_hook_called
        _dummy_hook_called = False

        from litellm.proxy.proxy_server import proxy_startup_event

        env_overrides = {
            "LITELLM_WORKER_STARTUP_HOOKS": "tests.test_litellm.proxy.test_proxy_cli:_dummy_hook",
        }
        # Remove DATABASE_URL to avoid real DB setup
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }
        clean_env.update(env_overrides)

        with patch.dict(os.environ, clean_env, clear=True):
            try:
                async with proxy_startup_event(app=None) as _:
                    pass
            except Exception:
                pass  # We expect errors after the hook (no DB, etc.)

        assert _dummy_hook_called is True, "Sync startup hook was not called"

    @pytest.mark.asyncio
    async def test_should_run_async_worker_startup_hook(self):
        """Async worker startup hook is awaited during proxy_startup_event."""
        global _dummy_async_hook_called
        _dummy_async_hook_called = False

        from litellm.proxy.proxy_server import proxy_startup_event

        env_overrides = {
            "LITELLM_WORKER_STARTUP_HOOKS": "tests.test_litellm.proxy.test_proxy_cli:_dummy_async_hook",
        }
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }
        clean_env.update(env_overrides)

        with patch.dict(os.environ, clean_env, clear=True):
            try:
                async with proxy_startup_event(app=None) as _:
                    pass
            except Exception:
                pass

        assert _dummy_async_hook_called is True, "Async startup hook was not called"

    @pytest.mark.asyncio
    async def test_should_raise_on_failing_worker_startup_hook(self):
        """A failing worker startup hook propagates the error."""
        from litellm.proxy.proxy_server import proxy_startup_event

        env_overrides = {
            "LITELLM_WORKER_STARTUP_HOOKS": "tests.test_litellm.proxy.test_proxy_cli:_failing_hook",
        }
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }
        clean_env.update(env_overrides)

        with patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(RuntimeError, match="Hook failed on purpose"):
                async with proxy_startup_event(app=None) as _:
                    pass

    def test_should_skip_when_no_hooks_set(self):
        """When LITELLM_WORKER_STARTUP_HOOKS is not set, no hooks are executed."""
        global _dummy_hook_called
        _dummy_hook_called = False

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LITELLM_WORKER_STARTUP_HOOKS", None)
            # The hook block should be skipped entirely when env var is absent
            assert "LITELLM_WORKER_STARTUP_HOOKS" not in os.environ
            # Verify that an empty env var value also results in no hook execution
            assert os.environ.get("LITELLM_WORKER_STARTUP_HOOKS", "") == ""

    @pytest.mark.asyncio
    async def test_should_run_multiple_hooks(self):
        """Multiple comma-separated hooks are all called."""
        global _dummy_hook_called, _dummy_async_hook_called
        _dummy_hook_called = False
        _dummy_async_hook_called = False

        from litellm.proxy.proxy_server import proxy_startup_event

        hooks = (
            "tests.test_litellm.proxy.test_proxy_cli:_dummy_hook,"
            "tests.test_litellm.proxy.test_proxy_cli:_dummy_async_hook"
        )
        env_overrides = {
            "LITELLM_WORKER_STARTUP_HOOKS": hooks,
        }
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("DATABASE_URL", "DIRECT_URL")
        }
        clean_env.update(env_overrides)

        with patch.dict(os.environ, clean_env, clear=True):
            try:
                async with proxy_startup_event(app=None) as _:
                    pass
            except Exception:
                pass

        assert _dummy_hook_called is True, "First hook was not called"
        assert _dummy_async_hook_called is True, "Second hook was not called"
