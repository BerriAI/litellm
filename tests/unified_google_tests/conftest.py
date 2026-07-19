# conftest.py

import asyncio
import importlib
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Tuple

import pytest
import uvicorn
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm  # noqa: E402,F401

from tests._vcr_conftest_common import (  # noqa: E402,F401
    VerboseReporterState,
    _pin_multipart_boundary,
    apply_vcr_auto_marker_to_items,
    emit_cassette_cache_session_banner,
    emit_vcr_classification_summary,
    emit_vcr_diagnostic_log,
    install_live_call_probe,
    record_vcr_outcome,
    register_persister_if_enabled,
    reset_vcr_diag_dir,
    vcr_config_dict,
)

_verbose_state = VerboseReporterState()

PROXY_CONFIG_PATH = Path(__file__).parent / "google_genai_proxy_test_config.yaml"
PROXY_MASTER_KEY = "sk-1234"
PROXY_START_TIMEOUT_S = 30.0


def _start_proxy_server(
    config_path: str,
) -> Tuple[str, uvicorn.Server, threading.Thread, socket.socket]:
    from litellm.proxy.proxy_server import (
        app as proxy_app,
        cleanup_router_config_variables,
        initialize,
    )

    cleanup_router_config_variables()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()

    config = uvicorn.Config(proxy_app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(initialize(config=config_path, debug=True))
        loop.run_until_complete(server.serve(sockets=[sock]))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    start_time = time.time()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("LiteLLM proxy failed to start")
        if time.time() - start_time > PROXY_START_TIMEOUT_S:
            raise TimeoutError("LiteLLM proxy did not start in time")
        time.sleep(0.05)

    return f"http://{host}:{port}", server, thread, sock


@pytest.fixture(scope="session")
def google_genai_proxy_url() -> Iterator[str]:
    from base_google_genai_proxy_sdk_test import has_vertex_credentials
    from base_google_test import load_vertex_ai_credentials

    saved_env = {
        key: os.environ.get(key)
        for key in (
            "DATABASE_URL",
            "DIRECT_URL",
            "LITELLM_MASTER_KEY",
            "STORE_MODEL_IN_DB",
            "GOOGLE_APPLICATION_CREDENTIALS",
        )
    }
    temp_credentials_path: str | None = None
    os.environ.pop("DATABASE_URL", None)
    os.environ.pop("DIRECT_URL", None)
    os.environ["LITELLM_MASTER_KEY"] = PROXY_MASTER_KEY
    os.environ["STORE_MODEL_IN_DB"] = "False"

    if has_vertex_credentials():
        credentials_file = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        if not (credentials_file and os.path.isfile(credentials_file)):
            vertex_credentials_path = load_vertex_ai_credentials(
                model="vertex_ai/gemini-2.5-flash-lite"
            )
            if vertex_credentials_path:
                temp_credentials_path = vertex_credentials_path
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = vertex_credentials_path

    server_url, server, thread, sock = _start_proxy_server(str(PROXY_CONFIG_PATH))
    try:
        yield server_url
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
        if temp_credentials_path:
            try:
                os.unlink(temp_credentials_path)
            except OSError:
                pass
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown(request):
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    sys.path.insert(
        0, os.path.abspath("../..")
    )  # Adds the project directory to the system path

    import litellm

    if "google_genai_proxy_url" not in request.fixturenames:
        importlib.reload(litellm)

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    print(litellm)
    yield

    # Teardown code (executes after the yield point)
    loop.close()  # Close the loop created earlier
    asyncio.set_event_loop(None)  # Remove the reference to the loop


@pytest.fixture(scope="module")
def vcr_config():
    return vcr_config_dict()


def pytest_recording_configure(config, vcr):
    register_persister_if_enabled(vcr)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


def pytest_configure(config):
    _verbose_state.remember_pluginmanager(config)
    reset_vcr_diag_dir()


def pytest_runtest_logreport(report):
    _verbose_state.maybe_emit_verdict(report)


def pytest_collection_modifyitems(config, items):
    apply_vcr_auto_marker_to_items(
        items,
        skip_nodeid_suffixes=(
            "test_proxy_genai_sdk_non_streaming",
            "test_proxy_genai_sdk_streaming_completes_without_errors",
            "test_proxy_genai_sdk_streaming_dict_style",
        ),
    )

    # Separate tests in 'test_amazing_proxy_custom_logger.py' and other tests
    custom_logger_tests = [
        item for item in items if "custom_logger" in item.parent.name
    ]
    other_tests = [item for item in items if "custom_logger" not in item.parent.name]

    # Sort tests based on their names
    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    # Reorder the items list
    items[:] = custom_logger_tests + other_tests


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    emit_cassette_cache_session_banner(terminalreporter)
    emit_vcr_classification_summary(terminalreporter)
    emit_vcr_diagnostic_log(terminalreporter)
