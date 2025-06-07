# ruff: noqa: T201
import importlib
import json
import os
import random
import subprocess
import sys
import urllib.parse
import urllib.parse as urlparse
from typing import TYPE_CHECKING, Any, Optional, Union

import click
import httpx
from dotenv import load_dotenv

if TYPE_CHECKING:
    from fastapi import FastAPI
else:
    FastAPI = Any

sys.path.append(os.getcwd())

# Check if rich is available for enhanced CLI output
IS_RICH_AVAILABLE = False
try:
    importlib.import_module("rich")
    IS_RICH_AVAILABLE = True
except ImportError:
    IS_RICH_AVAILABLE = False
config_filename = "litellm.secrets"

litellm_mode = os.getenv("LITELLM_MODE", "DEV")  # "PRODUCTION", "DEV"
if litellm_mode == "DEV":
    load_dotenv()
from enum import Enum

telemetry = None
if IS_RICH_AVAILABLE:
    from rich.align import Align
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    console = Console()

    class LiteLLMDatabaseConnectionPool(Enum):
        database_connection_pool_limit = 10
        database_connection_pool_timeout = 60

    def append_query_params(url, params) -> str:
        from litellm._logging import verbose_proxy_logger

        verbose_proxy_logger.debug(f"url: {url}")
        verbose_proxy_logger.debug(f"params: {params}")
        parsed_url = urlparse.urlparse(url)
        parsed_query = urlparse.parse_qs(parsed_url.query)
        parsed_query.update(params)
        encoded_query = urlparse.urlencode(parsed_query, doseq=True)
        modified_url = urlparse.urlunparse(parsed_url._replace(query=encoded_query))
        return modified_url  # type: ignore

    class ProxyInitializationHelpers:
        @staticmethod
        def _echo_litellm_version():
            """Display LiteLLM version with rich formatting"""
            try:
                pkg_version = importlib.metadata.version("litellm")  # type: ignore

                # Create a beautiful version display
                version_panel = Panel(
                    Align.center(
                        f"[bold cyan]LiteLLM[/bold cyan]\n[green]Version: {pkg_version}[/green]"
                    ),
                    title="[bold blue]LiteLLM Proxy[/bold blue]",
                    border_style="cyan",
                    padding=(1, 2),
                )
                console.print()
                console.print(version_panel)
                console.print()
            except Exception as e:
                console.print(f"[red]Error getting version: {e}[/red]")

        @staticmethod
        def _run_health_check(host, port):
            """Run health check with rich progress indicators"""
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running health check...", total=None)

                try:
                    response = httpx.get(url=f"http://{host}:{port}/health")
                    progress.update(task, completed=True)

                    if response.status_code == 200:
                        console.print("[green]✅ Health check passed![/green]")

                        # Create a formatted table for health check results
                        health_data = response.json()

                        if isinstance(health_data, dict):
                            table = Table(
                                title="Health Check Results",
                                show_header=True,
                                header_style="bold magenta",
                            )
                            table.add_column("Model", style="cyan")
                            table.add_column("Status", style="green")
                            table.add_column("Response Time", style="yellow")

                            for model, details in health_data.items():
                                if isinstance(details, dict):
                                    status = (
                                        "✅ Healthy"
                                        if details.get("status") == "healthy"
                                        else "❌ Unhealthy"
                                    )
                                    response_time = details.get("response_time", "N/A")
                                    table.add_row(model, status, str(response_time))

                            console.print(table)
                        else:
                            console.print_json(data=health_data)
                    else:
                        console.print(
                            f"[red]❌ Health check failed with status {response.status_code}[/red]"
                        )

                except Exception as e:
                    progress.update(task, completed=True)
                    console.print(f"[red]❌ Health check failed: {e}[/red]")

        @staticmethod
        def _run_test_chat_completion(
            host: str,
            port: int,
            model: str,
            test: Union[bool, str],
        ):
            """Run test chat completion with rich formatting and progress"""
            request_model = model or "gpt-3.5-turbo"

            # Create test info panel
            test_panel = Panel(
                f"[cyan]Model:[/cyan] {request_model}\n[cyan]Endpoint:[/cyan] http://{host}:{port}",
                title="[bold yellow]Test Configuration[/bold yellow]",
                border_style="yellow",
            )
            console.print(test_panel)

            import openai

            api_base = f"http://{host}:{port}"
            if isinstance(test, str):
                api_base = test
            else:
                raise ValueError("Invalid test value")

            client = openai.OpenAI(api_key="My API Key", base_url=api_base)

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                # Test 1: Regular completion
                task1 = progress.add_task("Testing chat completion...", total=None)
                try:
                    response = client.chat.completions.create(
                        model=request_model,
                        messages=[
                            {
                                "role": "user",
                                "content": "this is a test request, write a short poem",
                            }
                        ],
                        max_tokens=256,
                    )
                    progress.update(task1, completed=True)
                    console.print("[green]✅ Chat completion test passed![/green]")

                    # Display response in a nice format
                    if response.choices and response.choices[0].message:
                        response_panel = Panel(
                            response.choices[0].message.content or "No content",
                            title="[bold green]Response[/bold green]",
                            border_style="green",
                        )
                        console.print(response_panel)

                except Exception as e:
                    progress.update(task1, completed=True)
                    console.print(f"[red]❌ Chat completion test failed: {e}[/red]")

                # Test 2: Streaming completion
                task2 = progress.add_task("Testing streaming completion...", total=None)
                try:
                    stream_response = client.chat.completions.create(
                        model=request_model,
                        messages=[
                            {
                                "role": "user",
                                "content": "this is a test request, write a short poem",
                            }
                        ],
                        stream=True,
                    )

                    console.print("[cyan]Streaming response:[/cyan]")
                    for chunk in stream_response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            console.print(chunk.choices[0].delta.content, end="")

                    progress.update(task2, completed=True)
                    console.print(
                        "\n[green]✅ Streaming completion test passed![/green]"
                    )

                except Exception as e:
                    progress.update(task2, completed=True)
                    console.print(
                        f"[red]❌ Streaming completion test failed: {e}[/red]"
                    )

                # Test 3: Legacy completion
                task3 = progress.add_task("Testing legacy completion...", total=None)
                try:
                    completion_response = client.completions.create(
                        model=request_model,
                        prompt="this is a test request, write a short poem",
                    )
                    progress.update(task3, completed=True)
                    console.print("[green]✅ Legacy completion test passed![/green]")

                    # Display response in a nice format
                    if (
                        completion_response.choices
                        and completion_response.choices[0].text
                    ):
                        legacy_response_panel = Panel(
                            completion_response.choices[0].text or "No content",
                            title="[bold green]Legacy Response[/bold green]",
                            border_style="green",
                        )
                        console.print(legacy_response_panel)

                except Exception as e:
                    progress.update(task3, completed=True)
                    console.print(f"[red]❌ Legacy completion test failed: {e}[/red]")

        @staticmethod
        def _get_default_unvicorn_init_args(
            host: str,
            port: int,
            log_config: Optional[str] = None,
        ) -> dict:
            """
            Get the arguments for `uvicorn` worker
            """
            import litellm

            uvicorn_args = {
                "app": "litellm.proxy.proxy_server:app",
                "host": host,
                "port": port,
            }
            if log_config is not None:
                console.print(f"[cyan]Using log_config:[/cyan] {log_config}")
                uvicorn_args["log_config"] = log_config
            elif litellm.json_logs:
                console.print(
                    "[cyan]Using JSON logs. Setting log_config to None.[/cyan]"
                )
                uvicorn_args["log_config"] = None
            return uvicorn_args

        @staticmethod
        def _init_hypercorn_server(
            app: FastAPI,
            host: str,
            port: int,
            ssl_certfile_path: str,
            ssl_keyfile_path: str,
        ):
            """
            Initialize litellm with `hypercorn`
            """
            import asyncio

            from hypercorn.asyncio import serve
            from hypercorn.config import Config

            # Display server start message with rich formatting
            server_panel = Panel(
                f"[green]Starting LiteLLM Proxy Server[/green]\n"
                f"[cyan]Server:[/cyan] Hypercorn\n"
                f"[cyan]Host:[/cyan] {host}\n"
                f"[cyan]Port:[/cyan] {port}",
                title="[bold blue]Server Configuration[/bold blue]",
                border_style="blue",
            )
            console.print(server_panel)

            config = Config()
            config.bind = [f"{host}:{port}"]

            if ssl_certfile_path is not None and ssl_keyfile_path is not None:
                ssl_panel = Panel(
                    f"[cyan]Certificate:[/cyan] {ssl_certfile_path}\n"
                    f"[cyan]Key File:[/cyan] {ssl_keyfile_path}",
                    title="[bold green]SSL Configuration[/bold green]",
                    border_style="green",
                )
                console.print(ssl_panel)
                config.certfile = ssl_certfile_path
                config.keyfile = ssl_keyfile_path

            # hypercorn serve raises a type warning when passing a fast api app - even though fast API is a valid type
            asyncio.run(serve(app, config))  # type: ignore

        @staticmethod
        def _run_gunicorn_server(
            host: str,
            port: int,
            app: FastAPI,
            num_workers: int,
            ssl_certfile_path: str,
            ssl_keyfile_path: str,
        ):
            """
            Run litellm with `gunicorn`
            """
            if os.name == "nt":
                pass
            else:
                import gunicorn.app.base

            # Gunicorn Application Class
            class StandaloneApplication(gunicorn.app.base.BaseApplication):
                def __init__(self, app, options=None):
                    self.options = options or {}  # gunicorn options
                    self.application = app  # FastAPI app
                    super().__init__()

                    # Create beautiful server info display
                    server_info = Panel(
                        f"[green]LiteLLM Proxy Server Starting[/green]\n"
                        f"[cyan]Server:[/cyan] Gunicorn\n"
                        f"[cyan]Host:[/cyan] {host}\n"
                        f"[cyan]Port:[/cyan] {port}\n"
                        f"[cyan]Workers:[/cyan] {num_workers}",
                        title="[bold blue]Server Configuration[/bold blue]",
                        border_style="blue",
                    )
                    console.print(server_info)

                    # Create testing instructions
                    curl_command = f"""curl --location 'http://0.0.0.0:{port}/chat/completions' \\
    --header 'Content-Type: application/json' \\
    --data '{{
        "model": "gpt-3.5-turbo",
        "messages": [
            {{
                "role": "user",
                "content": "what llm are you"
            }}
        ]
    }}'"""

                    test_panel = Panel(
                        f"[yellow]Quick Test:[/yellow] litellm --test\n\n"
                        f"[yellow]cURL Test:[/yellow]\n{curl_command}",
                        title="[bold yellow]Testing Instructions[/bold yellow]",
                        border_style="yellow",
                    )
                    console.print(test_panel)

                    links_panel = Panel(
                        f"[blue]Documentation:[/blue] https://docs.litellm.ai/docs/simple_proxy\n"
                        f"[blue]Swagger UI:[/blue] http://0.0.0.0:{port}",
                        title="[bold blue]Useful Links[/bold blue]",
                        border_style="blue",
                    )
                    console.print(links_panel)

                def load_config(self):
                    # note: This Loads the gunicorn config - has nothing to do with LiteLLM Proxy config
                    if self.cfg is not None:
                        config = {
                            key: value
                            for key, value in self.options.items()
                            if key in self.cfg.settings and value is not None
                        }
                    else:
                        config = {}
                    for key, value in config.items():
                        if self.cfg is not None:
                            self.cfg.set(key.lower(), value)

                def load(self):
                    # gunicorn app function
                    return self.application

            gunicorn_options = {
                "bind": f"{host}:{port}",
                "workers": num_workers,  # default is 1
                "worker_class": "uvicorn.workers.UvicornWorker",
                "preload": True,  # Add the preload flag,
                "accesslog": "-",  # Log to stdout
                "timeout": 600,  # default to very high number, bedrock/anthropic.claude-v2:1 can take 30+ seconds for the 1st chunk to come in
                "access_log_format": '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s',
            }

            if ssl_certfile_path is not None and ssl_keyfile_path is not None:
                ssl_panel = Panel(
                    f"[cyan]Certificate:[/cyan] {ssl_certfile_path}\n"
                    f"[cyan]Key File:[/cyan] {ssl_keyfile_path}",
                    title="[bold green]SSL Configuration[/bold green]",
                    border_style="green",
                )
                console.print(ssl_panel)
                gunicorn_options["certfile"] = ssl_certfile_path
                gunicorn_options["keyfile"] = ssl_keyfile_path

            StandaloneApplication(
                app=app, options=gunicorn_options
            ).run()  # Run gunicorn

        @staticmethod
        def _run_ollama_serve():
            try:
                command = ["ollama", "serve"]

                with open(os.devnull, "w") as devnull:
                    subprocess.Popen(command, stdout=devnull, stderr=devnull)
                    console.print("[green]✅ Ollama serve started successfully[/green]")
            except Exception as e:
                console.print(
                    Panel(
                        f"[red]Failed to start Ollama serve[/red]\n"
                        f"[yellow]Error:[/yellow] {e}\n"
                        f"[yellow]Please ensure Ollama is installed and run:[/yellow] ollama serve",
                        title="[bold red]Ollama Warning[/bold red]",
                        border_style="red",
                    )
                )

        @staticmethod
        def _is_port_in_use(port):
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(("localhost", port)) == 0

        @staticmethod
        def _get_loop_type():
            """Helper function to determine the event loop type based on platform"""
            if sys.platform in ("win32", "cygwin", "cli"):
                return None  # Let uvicorn choose the default loop on Windows
            return "uvloop"

        @staticmethod
        def _display_startup_banner(
            host: str, port: int, config_path: Optional[str] = None
        ):
            """Display a beautiful startup banner"""
            # Create configuration info
            config_info = f"[cyan]Host:[/cyan] {host}\n[cyan]Port:[/cyan] {port}"
            if config_path:
                config_info += f"\n[cyan]Config:[/cyan] {config_path}"

            # Create startup panel
            startup_panel = Panel(
                Align.center(
                    f"[bold cyan]🚄 LiteLLM Proxy[/bold cyan]\n\n{config_info}"
                ),
                title="[bold green]Starting Server[/bold green]",
                border_style="green",
                padding=(1, 2),
            )

            console.print()
            console.print(startup_panel)
            console.print()

    def create_help_panel():
        """Create a beautiful help panel with grouped options"""
        help_content = """
    [bold cyan]Server Options:[/bold cyan]
    --host, --port          Server binding configuration
    --num_workers          Number of worker processes
    --config, -c           Configuration file path
    
    [bold cyan]Model Options:[/bold cyan]
    --model, -m            Model name
    --alias                Model alias
    --api_base             API base URL
    
    [bold cyan]Logging & Debug:[/bold cyan]
    --debug                Enable debug mode
    --detailed_debug       Enable detailed debugging
    --log_config           Logging configuration file
    
    [bold cyan]Testing:[/bold cyan]
    --test                 Run test completion
    --health               Run health check
    --version, -v          Show version
    
    [bold cyan]Security:[/bold cyan]
    --ssl_certfile_path    SSL certificate file
    --ssl_keyfile_path     SSL key file
    """

        help_panel = Panel(
            help_content.strip(),
            title="[bold blue]LiteLLM Proxy CLI Help[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
        return help_panel

    # Enhanced click command with rich help
    class RichCommand(click.Command):
        def format_help(self, ctx, formatter):
            console.print(create_help_panel())

    @click.command(cls=RichCommand)
    @click.option(
        "--host",
        default="0.0.0.0",
        help="🌐 Host for the server to listen on",
        envvar="HOST",
        show_default=True,
    )
    @click.option(
        "--port",
        default=4000,
        help="🔌 Port to bind the server to",
        envvar="PORT",
        show_default=True,
    )
    @click.option(
        "--num_workers",
        default=1,
        help="👥 Number of uvicorn/gunicorn workers",
        envvar="NUM_WORKERS",
        show_default=True,
    )
    @click.option("--api_base", default=None, help="🔗 API base URL")
    @click.option(
        "--api_version",
        default="2024-07-01-preview",
        help="📅 Azure API version",
        show_default=True,
    )
    @click.option("--model", "-m", default=None, help="🤖 Model name to use")
    @click.option(
        "--alias",
        default=None,
        help='📝 Model alias (e.g., "codellama" for long model names)',
    )
    @click.option("--add_key", default=None, help="🔑 Add API key")
    @click.option("--headers", default=None, help="📋 Headers for API calls")
    @click.option("--save", is_flag=True, help="💾 Save model-specific configuration")
    @click.option(
        "--debug",
        default=False,
        is_flag=True,
        help="🐛 Enable debug mode",
        envvar="DEBUG",
    )
    @click.option(
        "--detailed_debug",
        default=False,
        is_flag=True,
        help="🔍 Enable detailed debug logs",
        envvar="DETAILED_DEBUG",
    )
    @click.option(
        "--use_queue",
        default=False,
        is_flag=True,
        help="⚡ Use celery workers for async endpoints",
    )
    @click.option(
        "--temperature", default=None, type=float, help="🌡️  Model temperature"
    )
    @click.option("--max_tokens", default=None, type=int, help="📏 Maximum tokens")
    @click.option(
        "--request_timeout",
        default=None,
        type=int,
        help="⏱️  Request timeout (seconds)",
    )
    @click.option("--drop_params", is_flag=True, help="🗑️  Drop unmapped parameters")
    @click.option(
        "--add_function_to_prompt",
        is_flag=True,
        help="🔧 Add unsupported functions to prompt",
    )
    @click.option(
        "--config",
        "-c",
        default=None,
        help="⚙️  Configuration file path (e.g., config.yaml)",
    )
    @click.option(
        "--max_budget",
        default=None,
        type=float,
        help="💰 Maximum budget for API calls",
    )
    @click.option(
        "--telemetry",
        default=True,
        type=bool,
        help="📊 Enable telemetry (helps improve LiteLLM)",
        show_default=True,
    )
    @click.option(
        "--log_config",
        default=None,
        type=str,
        help="📝 Logging configuration file path",
    )
    @click.option(
        "--version",
        "-v",
        default=False,
        is_flag=True,
        help="📋 Show LiteLLM version",
    )
    @click.option(
        "--health",
        flag_value=True,
        help="🏥 Run health check on all models",
    )
    @click.option(
        "--test",
        flag_value=True,
        help="🧪 Run test chat completion",
    )
    @click.option(
        "--test_async",
        default=False,
        is_flag=True,
        help="⚡ Test async endpoints",
    )
    @click.option(
        "--iam_token_db_auth",
        default=False,
        is_flag=True,
        help="🔐 Use IAM token for database authentication",
    )
    @click.option(
        "--num_requests",
        default=10,
        type=int,
        help="🔢 Number of requests for async testing",
        show_default=True,
    )
    @click.option(
        "--run_gunicorn",
        default=False,
        is_flag=True,
        help="🦄 Use Gunicorn instead of Uvicorn",
    )
    @click.option(
        "--run_hypercorn",
        default=False,
        is_flag=True,
        help="🚄 Use Hypercorn (HTTP/2 support)",
    )
    @click.option(
        "--ssl_keyfile_path",
        default=None,
        type=str,
        help="🔐 SSL private key file path",
        envvar="SSL_KEYFILE_PATH",
    )
    @click.option(
        "--ssl_certfile_path",
        default=None,
        type=str,
        help="📜 SSL certificate file path",
        envvar="SSL_CERTFILE_PATH",
    )
    @click.option(
        "--use_prisma_migrate",
        is_flag=True,
        default=False,
        help="🗃️  Use Prisma migrate for schema updates",
    )
    @click.option(
        "--local", is_flag=True, default=False, help="🏠 Local debugging mode"
    )
    @click.option(
        "--skip_server_startup",
        is_flag=True,
        default=False,
        help="⏭️  Skip server startup (migrations only)",
    )
    def run_server(  # noqa: PLR0915
        host,
        port,
        api_base,
        api_version,
        model,
        alias,
        add_key,
        headers,
        save,
        debug,
        detailed_debug,
        temperature,
        max_tokens,
        request_timeout,
        drop_params,
        add_function_to_prompt,
        config,
        max_budget,
        telemetry,
        test,
        local,
        num_workers,
        test_async,
        iam_token_db_auth,
        num_requests,
        use_queue,
        health,
        version,
        run_gunicorn,
        run_hypercorn,
        ssl_keyfile_path,
        ssl_certfile_path,
        log_config,
        use_prisma_migrate,
        skip_server_startup,
    ):
        """
        🚄 LiteLLM Proxy Server - A unified interface for 100+ LLMs

        Start a proxy server that provides OpenAI-compatible endpoints for various LLM providers.
        """
        args = locals()

        # Handle imports
        if local:
            from proxy_server import (
                KeyManagementSettings,
                ProxyConfig,
                app,
                save_worker_config,
            )
        else:
            try:
                from .proxy_server import (
                    KeyManagementSettings,
                    ProxyConfig,
                    app,
                    save_worker_config,
                )
            except ImportError as e:
                if "litellm[proxy]" in str(e):
                    # user is missing a proxy dependency, ask them to pip install litellm[proxy]
                    console.print(
                        Panel(
                            "[red]Missing proxy dependencies![/red]\n\n"
                            "Please install with: [cyan]pip install 'litellm[proxy]'[/cyan]",
                            title="[bold red]Installation Error[/bold red]",
                            border_style="red",
                        )
                    )
                    raise e
                else:
                    # this is just a local/relative import error, user git cloned litellm
                    from proxy_server import (
                        KeyManagementSettings,
                        ProxyConfig,
                        app,
                        save_worker_config,
                    )

        # Handle version display
        if version is True:
            ProxyInitializationHelpers._echo_litellm_version()
            return

        # Handle Ollama setup
        if model and "ollama" in model and api_base is None:
            ProxyInitializationHelpers._run_ollama_serve()

        # Handle health check
        if health is True:
            ProxyInitializationHelpers._run_health_check(host, port)
            return

        # Handle test completion
        if test is True:
            ProxyInitializationHelpers._run_test_chat_completion(
                host, port, model, test
            )
            return

        # Main server startup flow
        else:
            # Display startup banner
            ProxyInitializationHelpers._display_startup_banner(host, port, config)

            if headers:
                headers = json.loads(headers)

            save_worker_config(
                model=model,
                alias=alias,
                api_base=api_base,
                api_version=api_version,
                debug=debug,
                detailed_debug=detailed_debug,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                max_budget=max_budget,
                telemetry=telemetry,
                drop_params=drop_params,
                add_function_to_prompt=add_function_to_prompt,
                headers=headers,
                save=save,
                config=config,
                use_queue=use_queue,
            )

            try:
                import uvicorn
            except Exception:
                console.print(
                    Panel(
                        "[red]Missing server dependencies![/red]\n\n"
                        "Please install with: [cyan]pip install 'litellm[proxy]'[/cyan]",
                        title="[bold red]Import Error[/bold red]",
                        border_style="red",
                    )
                )
                raise ImportError(
                    "uvicorn, gunicorn needs to be imported. Run - `pip install 'litellm[proxy]'`"
                )

            db_connection_pool_limit = 100
            db_connection_timeout = 60
            general_settings = {}

            ### GET DB TOKEN FOR IAM AUTH ###
            if iam_token_db_auth:
                from litellm.proxy.auth.rds_iam_token import generate_iam_auth_token

                db_host = os.getenv("DATABASE_HOST")
                db_port = os.getenv("DATABASE_PORT")
                db_user = os.getenv("DATABASE_USER")
                db_name = os.getenv("DATABASE_NAME")
                db_schema = os.getenv("DATABASE_SCHEMA")

                token = generate_iam_auth_token(
                    db_host=db_host, db_port=db_port, db_user=db_user
                )

                _db_url = (
                    f"postgresql://{db_user}:{token}@{db_host}:{db_port}/{db_name}"
                )
                if db_schema:
                    _db_url += f"?schema={db_schema}"

                os.environ["DATABASE_URL"] = _db_url
                os.environ["IAM_TOKEN_DB_AUTH"] = "True"

            ### DECRYPT ENV VAR ###
            from litellm.secret_managers.aws_secret_manager import decrypt_env_var

            if (
                os.getenv("USE_AWS_KMS", None) is not None
                and os.getenv("USE_AWS_KMS") == "True"
            ):
                ## V2 IMPLEMENTATION OF AWS KMS - USER WANTS TO DECRYPT MULTIPLE KEYS IN THEIR ENV
                new_env_var = decrypt_env_var()

                for k, v in new_env_var.items():
                    os.environ[k] = v

            if config is not None:
                """
                Allow user to pass in db url via config

                read from there and save it to os.env['DATABASE_URL']
                """
                try:
                    import asyncio

                except Exception:
                    raise ImportError(
                        "yaml needs to be imported. Run - `pip install 'litellm[proxy]'`"
                    )

                proxy_config = ProxyConfig()
                _config = asyncio.run(proxy_config.get_config(config_file_path=config))

                ### LITELLM SETTINGS ###
                litellm_settings = _config.get("litellm_settings", None)
                if (
                    litellm_settings is not None
                    and "json_logs" in litellm_settings
                    and litellm_settings["json_logs"] is True
                ):
                    import litellm

                    litellm.json_logs = True
                    litellm._turn_on_json()

                ### GENERAL SETTINGS ###
                general_settings = _config.get("general_settings", {})
                if general_settings is None:
                    general_settings = {}
                if general_settings:
                    ### LOAD SECRET MANAGER ###
                    key_management_system = general_settings.get(
                        "key_management_system", None
                    )
                    proxy_config.initialize_secret_manager(key_management_system)

                key_management_settings = general_settings.get(
                    "key_management_settings", None
                )
                if key_management_settings is not None:
                    import litellm

                    litellm._key_management_settings = KeyManagementSettings(
                        **key_management_settings
                    )

                database_url = general_settings.get("database_url", None)
                if database_url is None and os.getenv("DATABASE_URL") is None:
                    # Check if all required variables are provided
                    database_host = os.getenv("DATABASE_HOST")
                    database_username = os.getenv("DATABASE_USERNAME")
                    database_password = os.getenv("DATABASE_PASSWORD")
                    database_name = os.getenv("DATABASE_NAME")

                    if (
                        database_host
                        and database_username
                        and database_password
                        and database_name
                    ):
                        # Handle the problem of special character escaping in the database URL
                        database_username_enc = urllib.parse.quote_plus(
                            database_username
                        )
                        database_password_enc = urllib.parse.quote_plus(
                            database_password
                        )
                        database_name_enc = urllib.parse.quote_plus(database_name)

                        # Construct DATABASE_URL from the provided variables
                        database_url = f"postgresql://{database_username_enc}:{database_password_enc}@{database_host}/{database_name_enc}"

                        os.environ["DATABASE_URL"] = database_url

                db_connection_pool_limit = general_settings.get(
                    "database_connection_pool_limit",
                    LiteLLMDatabaseConnectionPool.database_connection_pool_limit.value,
                )
                db_connection_timeout = general_settings.get(
                    "database_connection_pool_timeout",
                    LiteLLMDatabaseConnectionPool.database_connection_pool_timeout.value,
                )
                if database_url and database_url.startswith("os.environ/"):
                    original_dir = os.getcwd()
                    # set the working directory to where this script is
                    sys.path.insert(
                        0, os.path.abspath("../..")
                    )  # Adds the parent directory to the system path - for litellm local dev
                    import litellm
                    from litellm import get_secret_str

                    database_url = get_secret_str(database_url, default_value=None)
                    os.chdir(original_dir)
                if database_url is not None and isinstance(database_url, str):
                    os.environ["DATABASE_URL"] = database_url

            if (
                os.getenv("DATABASE_URL", None) is not None
                or os.getenv("DIRECT_URL", None) is not None
            ):
                # Display a nice message before database setup
                db_setup_panel = Panel(
                    "[cyan]Setting up database connection and schema...[/cyan]",
                    title="[bold blue]Database Setup[/bold blue]",
                    border_style="blue",
                )
                console.print(db_setup_panel)

                try:
                    from litellm.secret_managers.main import get_secret

                    if os.getenv("DATABASE_URL", None) is not None:
                        ### add connection pool + pool timeout args
                        params = {
                            "connection_limit": db_connection_pool_limit,
                            "pool_timeout": db_connection_timeout,
                        }
                        database_url = get_secret("DATABASE_URL", default_value=None)
                        modified_url = append_query_params(database_url, params)
                        os.environ["DATABASE_URL"] = modified_url
                    if os.getenv("DIRECT_URL", None) is not None:
                        ### add connection pool + pool timeout args
                        params = {
                            "connection_limit": db_connection_pool_limit,
                            "pool_timeout": db_connection_timeout,
                        }
                        database_url = os.getenv("DIRECT_URL")
                        modified_url = append_query_params(database_url, params)
                        os.environ["DIRECT_URL"] = modified_url
                        ###
                    subprocess.run(["prisma"], capture_output=True)
                    is_prisma_runnable = True
                except FileNotFoundError:
                    is_prisma_runnable = False

                if is_prisma_runnable:
                    from litellm.proxy.db.check_migration import (
                        check_prisma_schema_diff,
                    )
                    from litellm.proxy.db.prisma_client import (
                        PrismaManager,
                        should_update_prisma_schema,
                    )

                    if (
                        should_update_prisma_schema(
                            general_settings.get("disable_prisma_schema_update")
                        )
                        is False
                    ):
                        check_prisma_schema_diff(db_url=None)
                    else:
                        PrismaManager.setup_database(use_migrate=use_prisma_migrate)
                else:
                    console.print(
                        "[yellow]⚠️  Unable to connect to DB. DATABASE_URL found but Prisma not available.[/yellow]"
                    )

            # Check port availability
            if port == 4000 and ProxyInitializationHelpers._is_port_in_use(port):
                old_port = port
                port = random.randint(1024, 49152)
                console.print(
                    f"[yellow]⚠️  Port {old_port} is in use. Using port {port} instead.[/yellow]"
                )

            import litellm

            if detailed_debug is True:
                litellm._turn_on_debug()
                console.print("[cyan]🔍 Detailed debugging enabled[/cyan]")

            # DO NOT DELETE - enables global variables to work across files
            from litellm.proxy.proxy_server import app  # noqa

            # Skip server startup if requested (after all setup is done)
            if skip_server_startup:
                console.print(
                    "[yellow]⏭️  Setup complete. Skipping server startup as requested.[/yellow]"
                )
                return

            # Final server startup
            uvicorn_args = ProxyInitializationHelpers._get_default_unvicorn_init_args(
                host=host,
                port=port,
                log_config=log_config,
            )

            if run_gunicorn is False and run_hypercorn is False:
                if ssl_certfile_path is not None and ssl_keyfile_path is not None:
                    ssl_panel = Panel(
                        f"[cyan]Certificate:[/cyan] {ssl_certfile_path}\n"
                        f"[cyan]Key File:[/cyan] {ssl_keyfile_path}",
                        title="[bold green]SSL Configuration[/bold green]",
                        border_style="green",
                    )
                    console.print(ssl_panel)
                    uvicorn_args["ssl_keyfile"] = ssl_keyfile_path
                    uvicorn_args["ssl_certfile"] = ssl_certfile_path

                loop_type = ProxyInitializationHelpers._get_loop_type()
                if loop_type:
                    uvicorn_args["loop"] = loop_type

                # Final startup message
                startup_msg = Panel(
                    f"[green]🚄 Starting LiteLLM Proxy Server[/green]\n"
                    f"[cyan]Server:[/cyan] Uvicorn\n"
                    f"[cyan]Host:[/cyan] {host}\n"
                    f"[cyan]Port:[/cyan] {port}\n"
                    f"[cyan]Workers:[/cyan] {num_workers}",
                    title="[bold blue]Server Starting[/bold blue]",
                    border_style="blue",
                )
                console.print(startup_msg)

                uvicorn.run(
                    **uvicorn_args,
                    workers=num_workers,
                )
            elif run_gunicorn is True:
                ProxyInitializationHelpers._run_gunicorn_server(
                    host=host,
                    port=port,
                    app=app,
                    num_workers=num_workers,
                    ssl_certfile_path=ssl_certfile_path,
                    ssl_keyfile_path=ssl_keyfile_path,
                )
            elif run_hypercorn is True:
                ProxyInitializationHelpers._init_hypercorn_server(
                    app=app,
                    host=host,
                    port=port,
                    ssl_certfile_path=ssl_certfile_path,
                    ssl_keyfile_path=ssl_keyfile_path,
                )

    if __name__ == "__main__":
        try:
            run_server()
        except KeyboardInterrupt:
            console.print("\n[yellow]👋 Server stopped by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Error: {e}[/red]")
            raise
