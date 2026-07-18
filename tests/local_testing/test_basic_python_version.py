import asyncio
import os
import subprocess
import sys
import time
import traceback

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


def _run_uv(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["uv", *args], check=True, cwd=PROJECT_ROOT, **kwargs)


def test_using_litellm():
    try:
        import litellm

        print("litellm imported successfully")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}. Installing litellm failed please retry")


def test_litellm_proxy_server():
    # Sync the local litellm[proxy] dependencies into the project environment
    _run_uv("sync", "--frozen", "--extra", "proxy")

    # Import through the uv-managed interpreter that uv sync populated.
    try:
        _run_uv("run", "--no-sync", "python", "-c", "import litellm.proxy.proxy_server")
    except subprocess.CalledProcessError:
        pytest.fail("Failed to import litellm.proxy.proxy_server")

    # Assertion to satisfy the test, you can add other checks as needed
    assert True


def test_package_dependencies():
    """
    Test that all optional dependency entries are exposed via project optional-dependencies.
    """
    try:
        import pathlib
        import litellm
        from packaging.requirements import Requirement

        # Try to import tomllib (Python 3.11+) or tomli (older versions)
        try:
            import tomllib as tomli
        except ImportError:
            try:
                import tomli
            except ImportError:
                pytest.skip("tomli/tomllib not available - skipping dependency check")

        # Get the litellm package root path
        litellm_path = pathlib.Path(litellm.__file__).parent.parent
        pyproject_path = litellm_path / "pyproject.toml"

        # Read and parse pyproject.toml
        with open(pyproject_path, "rb") as f:
            pyproject = tomli.load(f)

        optional_deps = pyproject["project"]["optional-dependencies"]
        assert optional_deps, "Expected project.optional-dependencies to be defined"

        parsed_requirements = set()
        for extra_name, requirements in optional_deps.items():
            assert requirements, f"Optional dependency group '{extra_name}' is empty"
            for requirement in requirements:
                assert isinstance(
                    requirement, str
                ), f"Expected string requirement in extra '{extra_name}'"
                parsed = Requirement(requirement)
                parsed_requirements.add(parsed.name.lower())

        print(parsed_requirements)
        print(
            f"Validated {len(parsed_requirements)} optional dependencies across {len(optional_deps)} extras groups"
        )

    except Exception as e:
        pytest.fail(
            f"Error occurred while checking dependencies: {str(e)}\n"
            + traceback.format_exc()
        )


def test_cli_extra_is_a_thin_client_install():
    """The `cli` extra must install a working `lite` client without dragging in the
    proxy server runtime. It therefore has to declare the CLI's real third-party
    deps (rich, pyyaml, requests) and must never contain a server-only dependency
    from the `proxy` extra; a leak there silently re-bloats the laptop install.
    """
    import pathlib

    import litellm
    from packaging.requirements import Requirement

    try:
        import tomllib as tomli
    except ImportError:
        try:
            import tomli
        except ImportError:
            pytest.skip("tomli/tomllib not available - skipping dependency check")

    pyproject_path = pathlib.Path(litellm.__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        optional_deps = tomli.load(f)["project"]["optional-dependencies"]

    assert "cli" in optional_deps, "Expected a `cli` extra for the thin lite install"

    cli_names = {Requirement(req).name.lower() for req in optional_deps["cli"]}

    missing = {"rich", "pyyaml", "requests"} - cli_names
    assert not missing, f"`cli` extra is missing deps the lite CLI imports: {missing}"

    server_only = {
        "fastapi",
        "uvicorn",
        "gunicorn",
        "granian",
        "starlette",
        "boto3",
        "polars",
        "soundfile",
        "mcp",
        "cryptography",
        "apscheduler",
        "rq",
        "litellm-enterprise",
        "litellm-proxy-extras",
    }
    leaked = cli_names & server_only
    assert not leaked, f"`cli` extra leaks proxy-server deps onto laptops: {leaked}"


def test_aiohttp_constraint_excludes_314():
    """aiohttp 3.14.x re-arms the sock_read timeout on a keep-alive connection after it
    is returned to the pool (aio-libs/aiohttp#12953), poisoning it so the next request
    fails instantly with a sub-millisecond `Connection timed out`. This surfaces as
    sporadic cross-provider timeouts (issue #33820). Until a 3.14.x release ships the
    aio-libs/aiohttp#12954 fix, the uv constraint must keep aiohttp below 3.14.
    """
    import pathlib

    import litellm
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    try:
        import tomllib as tomli
    except ImportError:
        try:
            import tomli
        except ImportError:
            pytest.skip("tomli/tomllib not available - skipping dependency check")

    pyproject_path = pathlib.Path(litellm.__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        constraints = tomli.load(f)["tool"]["uv"]["constraint-dependencies"]

    aiohttp_constraint = next(
        (c for c in constraints if c.replace(" ", "").lower().startswith("aiohttp")),
        None,
    )
    assert (
        aiohttp_constraint is not None
    ), "Expected an aiohttp entry in [tool.uv].constraint-dependencies"

    specifier = SpecifierSet(aiohttp_constraint.split("aiohttp", 1)[1].strip())
    poisoned = [v for v in ("3.14.0", "3.14.1") if specifier.contains(Version(v))]
    assert not poisoned, (
        f"aiohttp constraint '{aiohttp_constraint}' allows {poisoned}, which poison "
        "pooled keep-alive connections (aio-libs/aiohttp#12953, litellm #33820). "
        "Keep the pin below 3.14 until a release with the aio-libs/aiohttp#12954 fix ships"
    )


import os
import subprocess
import time

import pytest
import requests


def _run_proxy_server_smoke_test(extra_proxy_args=None):
    """Sync deps, generate Prisma client, start proxy with optional extra args,
    send a health check + chat/completions request, and tear down."""
    if extra_proxy_args is None:
        extra_proxy_args = []

    server_process = None
    try:
        _run_uv(
            "sync",
            "--frozen",
            "--group",
            "proxy-dev",
            "--extra",
            "proxy",
            "--extra",
            "extra_proxy",
        )

        # Ensure Prisma client is generated
        try:
            print(f"Running prisma generate from: {PROJECT_ROOT}")

            result = _run_uv(
                "run",
                "--no-sync",
                "prisma",
                "generate",
                capture_output=True,
                text=True,
            )
            print(f"Prisma generate stdout: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Prisma generate failed: {e}")
            print(f"Prisma generate stderr: {e.stderr}")
            raise
        filepath = os.path.dirname(os.path.abspath(__file__))
        config_fp = f"{filepath}/test_configs/test_config_no_auth.yaml"
        server_process = subprocess.Popen(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-m",
                "litellm.proxy.proxy_cli",
                "--config",
                config_fp,
                *extra_proxy_args,
            ],
            cwd=PROJECT_ROOT,
        )

        # Allow some time for the server to start (increased for CI environments)
        time.sleep(90)  # Increased from 60s for slower CI runners

        # Send a request to the /health/liveliness endpoint
        response = requests.get("http://localhost:4000/health/liveliness")

        # Check if the response is successful
        assert response.status_code == 200
        assert response.json() == "I'm alive!"

        # Test /chat/completions
        response = requests.post(
            "http://localhost:4000/chat/completions",
            headers={"Authorization": "Bearer 1234567890"},
            json={
                "model": "test_openai_models",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
            },
        )

        assert response.status_code == 200

    except ImportError:
        pytest.fail("Failed to import litellm.proxy_server")
    except requests.ConnectionError:
        pytest.fail("Failed to connect to the server")
    finally:
        # Shut down the server
        if server_process:
            server_process.terminate()
            server_process.wait()

    # Additional assertions can be added here
    assert True


def test_litellm_proxy_server_config_no_general_settings():
    """Exercises the default (v1) migration resolver."""
    _run_proxy_server_smoke_test()


def test_litellm_proxy_server_config_no_general_settings_v2_resolver():
    """Exercises the opt-in v2 migration resolver.

    Runs in a separate CI job against a local Postgres to avoid collisions
    with the v1 variant when they share a database.
    """
    _run_proxy_server_smoke_test(extra_proxy_args=["--use_v2_migration_resolver"])
