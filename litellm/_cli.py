from importlib.util import find_spec
from typing import Final

from litellm._version import get_distribution_name


def _require_cli(extra: str, modules: tuple[str, ...]) -> None:
    missing: Final = tuple(module for module in modules if find_spec(module) is None)
    if missing:
        raise SystemExit(f'Install the {extra} commands with pip install "{get_distribution_name()}[{extra}]"')


def run_server() -> None:
    _require_cli("proxy", ("click", "dotenv", "pydantic_settings"))
    from litellm.proxy.proxy_cli import run_server as command

    command()


def cli() -> None:
    _require_cli("cli", ("click", "filelock", "rich", "yaml", "InquirerPy", "tomlkit"))
    from litellm.proxy.client.cli import cli as command

    command()


def litellm_proxy_cli() -> None:
    _require_cli("cli", ("click", "filelock", "rich", "yaml", "InquirerPy", "tomlkit"))
    from litellm.proxy.client.cli import litellm_proxy_cli as command

    command()
