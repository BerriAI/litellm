from typing import Final

import click
import pytest
from click.testing import CliRunner, Result

from litellm.proxy.client.cli import cli
from litellm.proxy.client.cli.interface import run_shell_line

GATEWAY: Final = "https://gateway.example.com"


def _shell(*lines: str) -> Result:
    """Type each line into a bare `lite` shell against the gateway, then quit"""
    typed: Final = "".join(f"{line}\n" for line in (*lines, "quit"))
    return CliRunner().invoke(cli, ["--base-url", GATEWAY, "--api-key", "sk-shell"], input=typed, prog_name="lite")


@pytest.fixture
def credentials_endpoint(requests_mock):
    return requests_mock.get(
        f"{GATEWAY}/credentials",
        json={"credentials": [{"credential_name": "shared-openai", "credential_info": {}}]},
    )


def test_a_group_typed_alone_prints_its_own_usage_and_opens_a_prompt_inside_it(credentials_endpoint):
    result: Final = _shell("credentials", "list")

    assert "Usage: lite credentials [OPTIONS] COMMAND [ARGS]..." in result.output
    assert "lite lite" not in result.output
    assert "credentials > " in result.output
    assert "Unknown command" not in result.output
    assert credentials_endpoint.call_count == 1
    assert credentials_endpoint.last_request.headers["Authorization"] == "Bearer sk-shell"


@pytest.mark.parametrize(
    "lines",
    [
        ("credentials list",),
        ("lite credentials list",),
        ("credentials", "list"),
        ("credentials", "credentials list"),
        ("credentials", "lite credentials list"),
    ],
)
def test_every_spelling_of_a_group_subcommand_runs_it(lines, credentials_endpoint):
    result: Final = _shell(*lines)

    assert "Unknown command" not in result.output
    assert "shared-openai" in result.output
    assert credentials_endpoint.call_count == 1


@pytest.mark.parametrize(
    "create_line",
    [
        """create shell-made --info '{"custom_llm_provider": "openai"}' --values '{"api_key": "sk-placeholder"}'""",
        """create shell-made --info {"custom_llm_provider":"openai"} --values {"api_key":"sk-placeholder"}""",
        """create shell-made --info='{"custom_llm_provider": "openai"}' --values='{"api_key": "sk-placeholder"}'""",
    ],
)
def test_quoted_and_compact_json_arguments_reach_the_gateway_as_typed(requests_mock, create_line):
    create: Final = requests_mock.post(f"{GATEWAY}/credentials", json={"credential_name": "shell-made"})

    _shell("credentials", create_line)

    assert create.call_count == 1
    assert create.last_request.json() == {
        "credential_name": "shell-made",
        "credential_info": {"custom_llm_provider": "openai"},
        "credential_values": {"api_key": "sk-placeholder"},
    }


def test_an_unclosed_quote_runs_nothing_and_keeps_the_group(requests_mock, credentials_endpoint):
    create: Final = requests_mock.post(f"{GATEWAY}/credentials", json={})

    result: Final = _shell("credentials", "create shell-made --info '{", "list")

    assert "Could not read that command" in result.output
    assert create.call_count == 0
    assert credentials_endpoint.call_count == 1


def test_an_unknown_command_inside_a_group_keeps_the_group(credentials_endpoint):
    result: Final = _shell("credentials", "frobnicate", "list")

    assert "Unknown command: frobnicate" in result.output
    assert credentials_endpoint.call_count == 1


@pytest.mark.parametrize("leaving_line", ["models list", "help", "lite"])
def test_a_top_level_line_inside_a_group_leaves_the_group(leaving_line, requests_mock, credentials_endpoint):
    requests_mock.get(f"{GATEWAY}/models", json={"data": []})

    result: Final = _shell("credentials", leaving_line, "list")

    assert "Unknown command: list" in result.output
    assert credentials_endpoint.call_count == 0


def test_help_inside_a_group_shows_that_group_and_keeps_it(credentials_endpoint):
    result: Final = _shell("credentials", "--help", "list")

    assert "Unknown command" not in result.output
    assert result.output.count("Usage: lite credentials [OPTIONS] COMMAND [ARGS]...") == 2
    assert credentials_endpoint.call_count == 1


def test_lite_with_its_own_options_inside_the_shell_runs_against_the_server_it_names(
    requests_mock, credentials_endpoint
):
    other: Final = requests_mock.get(
        "https://other-gateway.example.com/credentials", json={"credentials": [{"credential_name": "other-openai"}]}
    )

    result: Final = _shell("lite --base-url https://other-gateway.example.com --api-key sk-other credentials list")

    assert "other-openai" in result.output
    assert other.last_request.headers["Authorization"] == "Bearer sk-other"
    assert credentials_endpoint.call_count == 0


def test_a_group_that_runs_on_its_own_does_not_become_the_scope(capsys):
    @click.group()
    def root() -> None:
        pass

    @root.group(invoke_without_command=True)
    def wizard() -> None:
        click.echo("wizard ran")

    @wizard.command()
    def undo() -> None:
        pass

    scope: Final = run_shell_line("wizard", (), click.Context(root, info_name="lite", obj={}), root)

    assert scope == ()
    assert "wizard ran" in capsys.readouterr().out


def test_a_line_splits_on_spaces_and_unwraps_quotes_that_open_a_word_or_follow_an_equals_sign(capsys):
    @click.group()
    def root() -> None:
        pass

    @root.command()
    @click.argument("words", nargs=-1)
    def echo(words: tuple[str, ...]) -> None:
        click.echo("\n".join(words))

    run_shell_line(
        r"""echo C:\Users\me\notes.txt "two words" '{"a": 1}' info='{"a": 1}' {"b":2} it's issue#1""",
        (),
        click.Context(root, info_name="lite", obj={}),
        root,
    )

    assert capsys.readouterr().out.splitlines() == [
        r"C:\Users\me\notes.txt",
        "two words",
        '{"a": 1}',
        'info={"a": 1}',
        '{"b":2}',
        "it's",
        "issue#1",
    ]
