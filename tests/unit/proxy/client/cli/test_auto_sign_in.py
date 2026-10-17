import json
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

import pytest
import requests
from click.testing import CliRunner, Result
from requests_mock import Mocker

from litellm.proxy.client.cli import cli
from litellm.proxy.client.cli.commands.auth import CliTokenData, get_stored_api_key, save_token
from litellm.proxy.client.cli.commands.auto_sign_in import pkce_browser_sign_in
from litellm.proxy.client.cli.commands.pkce_login import PkceCredential, PkceFailure, pkce_token_record
from tests.unit.litellm_core_utils.fake_secret_vault import FakeSecretVault

BASE_URL: Final = "https://proxy.example.com"
SIGNED_IN_KEY: Final = "sk-signed-in"
GENERATED: Final = {"key": "sk-generated", "key_alias": "experiment-1"}
REJECTED: Final = {"error": {"message": "Authentication Error, No api key passed in.", "type": "auth_error"}}
SIGNED_IN_AT: Final = 1_790_000_000.0
EXPIRES_AT: Final = 4_102_444_800.0


class FakeBrowserSignIn:
    def __init__(self, outcome: PkceCredential | PkceFailure) -> None:
        self.outcome = outcome
        self.calls: list[str] = []

    def __call__(self, base_url: str, echo: Callable[[str], None]) -> PkceCredential | PkceFailure:
        self.calls.append(base_url)
        echo(f"Opening browser to: {base_url}/v1/cli/authorize?state=fake")
        return self.outcome


def _credential(access_token: str = SIGNED_IN_KEY) -> PkceCredential:
    return PkceCredential(
        access_token=access_token,
        refresh_token="refresh-token",
        expires_at=EXPIRES_AT,
        client_id="cli-client",
        token_endpoint=f"{BASE_URL}/v1/cli/token",
        revocation_endpoint=f"{BASE_URL}/v1/cli/revoke",
        resource=BASE_URL,
        user_id="agent-user",
        team_id=None,
    )


def _classic_login(key: str) -> CliTokenData:
    return {
        "base_url": BASE_URL,
        "key": key,
        "user_id": "agent-user",
        "user_email": "unknown",
        "user_role": "cli",
        "auth_header_name": "Authorization",
        "jwt_token": "",
        "timestamp": SIGNED_IN_AT,
    }


@pytest.fixture(autouse=True)
def no_key_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)


def _lite(args: Sequence[str], vault: object, sign_in: FakeBrowserSignIn, global_args: Sequence[str] = ()) -> Result:
    return CliRunner().invoke(
        cli,
        [*global_args, "--base-url", BASE_URL, *args],
        obj={"secret_vault": vault, "browser_sign_in": sign_in},
    )


def _bearers(route) -> list[str | None]:
    return [request.headers.get("Authorization") for request in route.request_history]


def _key_check(requests_mock, status_code: int):
    return requests_mock.get(f"{BASE_URL}/key/info", status_code=status_code, json={"info": {}})


def test_a_401_without_a_login_signs_in_once_stores_the_key_and_finishes_the_command(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault]
) -> None:
    route: Final = requests_mock.post(
        f"{BASE_URL}/key/generate", [{"status_code": 401, "json": REJECTED}, {"status_code": 200, "json": GENERATED}]
    )
    vault: Final = secret_vault_factory()
    sign_in: Final = FakeBrowserSignIn(_credential())

    result: Final = _lite(
        ["keys", "generate", "--models", "gpt-5.4-mini", "--key-alias", "experiment-1"], vault, sign_in
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == GENERATED
    assert sign_in.calls == [BASE_URL]
    assert _bearers(route) == [None, f"Bearer {SIGNED_IN_KEY}"]
    assert get_stored_api_key(BASE_URL, vault=vault) == SIGNED_IN_KEY
    assert f"The proxy at {BASE_URL} needs a sign-in" in result.stderr
    assert f"Opening browser to: {BASE_URL}/v1/cli/authorize" in result.stderr
    assert "Credential stored in your OS keychain." in result.stderr
    assert "Continuing `cli keys generate`" in result.stderr
    assert SIGNED_IN_KEY not in result.output


@pytest.mark.parametrize(
    "key_check",
    [{"status_code": 401}, {"status_code": 503}, {"exc": requests.exceptions.ConnectionError}],
    ids=["rejected", "unavailable", "unreachable"],
)
def test_a_stored_login_the_proxy_rejects_is_replaced_by_one_sign_in_with_only_json_on_stdout(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault], key_check: dict[str, object]
) -> None:
    vault: Final = secret_vault_factory()
    save_token(pkce_token_record(BASE_URL, _credential("sk-stale")), vault=vault)
    requests_mock.get(f"{BASE_URL}/key/info", **key_check)
    revoke: Final = requests_mock.post(f"{BASE_URL}/v1/cli/revoke", status_code=500)
    route: Final = requests_mock.post(
        f"{BASE_URL}/key/generate", [{"status_code": 401, "json": REJECTED}, {"status_code": 200, "json": GENERATED}]
    )
    sign_in: Final = FakeBrowserSignIn(_credential())

    result: Final = _lite(["keys", "generate"], vault, sign_in)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == GENERATED
    assert sign_in.calls == [BASE_URL]
    assert _bearers(route) == ["Bearer sk-stale", f"Bearer {SIGNED_IN_KEY}"]
    assert revoke.call_count == 1
    assert "Could not revoke the previous login's refresh token" in result.stderr
    assert get_stored_api_key(BASE_URL, vault=vault) == SIGNED_IN_KEY


def test_a_login_for_another_proxy_is_never_replaced_by_an_automatic_sign_in(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault]
) -> None:
    other_proxy: Final = "https://other.example.com"
    vault: Final = secret_vault_factory()
    save_token({**_classic_login("sk-other"), "base_url": other_proxy}, vault=vault)
    route: Final = requests_mock.post(f"{BASE_URL}/key/generate", status_code=401, json=REJECTED)
    sign_in: Final = FakeBrowserSignIn(_credential())

    result: Final = _lite(["keys", "generate"], vault, sign_in)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert f"lite keeps one stored login and it is for {other_proxy}" in result.stderr
    assert f"Run `lite --base-url {BASE_URL} login --pkce`" in result.stderr
    assert sign_in.calls == []
    assert _bearers(route) == [None]
    assert get_stored_api_key(other_proxy, vault=vault) == "sk-other"


def test_the_browser_sign_in_fails_at_once_on_a_machine_without_a_browser(requests_mock: Mocker) -> None:
    def no_browser() -> webbrowser.BaseBrowser:
        raise webbrowser.Error("could not locate runnable browser")

    echoed: list[str] = []

    outcome: Final = pkce_browser_sign_in(BASE_URL, echoed.append, find_browser=no_browser, environ={})

    assert outcome == PkceFailure("no web browser is available on this machine")
    assert requests_mock.call_count == 0
    assert echoed == []


@pytest.mark.parametrize("ci", ["true", "1", "True"])
def test_the_browser_sign_in_never_looks_for_a_browser_in_ci(requests_mock: Mocker, ci: str) -> None:
    def browser_lookup() -> webbrowser.BaseBrowser:
        raise AssertionError("looked for a browser while CI is set")

    echoed: list[str] = []

    outcome: Final = pkce_browser_sign_in(BASE_URL, echoed.append, find_browser=browser_lookup, environ={"CI": ci})

    assert outcome == PkceFailure("the CI environment variable is set, so lite does not open a browser")
    assert requests_mock.call_count == 0
    assert echoed == []


@pytest.mark.parametrize("name", ["www-browser", "links", "elinks", "lynx", "w3m"])
def test_a_text_mode_browser_counts_as_no_browser(requests_mock: Mocker, name: str) -> None:
    echoed: list[str] = []

    outcome: Final = pkce_browser_sign_in(
        BASE_URL, echoed.append, find_browser=lambda: webbrowser.GenericBrowser(name), environ={"TERM": "xterm"}
    )

    assert outcome == PkceFailure("only a text-mode web browser is available on this machine")
    assert requests_mock.call_count == 0
    assert echoed == []


@pytest.mark.parametrize("ssh_variable", ["SSH_CONNECTION", "SSH_TTY"])
def test_an_ssh_session_without_a_display_never_opens_a_browser_on_the_remote_screen(
    requests_mock: Mocker, ssh_variable: str
) -> None:
    def browser_lookup() -> webbrowser.BaseBrowser:
        raise AssertionError("looked for a browser in an SSH session without a display")

    echoed: list[str] = []

    outcome: Final = pkce_browser_sign_in(
        BASE_URL, echoed.append, find_browser=browser_lookup, environ={ssh_variable: "203.0.113.7 52144 10.0.0.5 22"}
    )

    assert outcome == PkceFailure(
        "this is an SSH session without a display, so a browser would open where you cannot see it"
    )
    assert requests_mock.call_count == 0
    assert echoed == []


@pytest.mark.parametrize(
    "environ",
    [
        {"BROWSER": "w3m"},
        {"CI": "false", "BROWSER": "w3m"},
        {"CI": "0", "BROWSER": "w3m"},
        {"SSH_CONNECTION": "203.0.113.7 52144 10.0.0.5 22", "BROWSER": "w3m"},
        {"SSH_CONNECTION": "203.0.113.7 52144 10.0.0.5 22", "DISPLAY": "localhost:10.0", "BROWSER": "w3m"},
    ],
    ids=["browser-picks-it", "ci-false", "ci-zero", "ssh-with-a-browser-forwarder", "ssh-with-x11-forwarding"],
)
def test_a_browser_the_user_picked_starts_the_sign_in_and_names_how_long_it_waits(
    requests_mock: Mocker, environ: dict[str, str]
) -> None:
    discovery: Final = requests_mock.get(f"{BASE_URL}/.well-known/litellm-cli-auth", status_code=404)
    echoed: list[str] = []

    outcome: Final = pkce_browser_sign_in(
        BASE_URL, echoed.append, find_browser=lambda: webbrowser.GenericBrowser("w3m"), environ=environ
    )

    assert isinstance(outcome, PkceFailure)
    assert discovery.call_count == 1
    assert echoed == [
        "lite is opening your browser to sign in and waits up to 5 minutes for you to approve it. "
        "A shell or agent that stops commands sooner should run this one with a longer timeout."
    ]


@pytest.mark.parametrize(
    ("command", "method", "path", "body"),
    [
        (["keys", "list", "--format", "json"], "GET", "/key/list", {"keys": []}),
        (["models", "list", "--format", "json"], "GET", "/models", {"data": []}),
        (["model-groups", "list", "--format", "json"], "GET", "/model_group/info", {"data": []}),
        (["users", "list"], "GET", "/user/list", {"users": []}),
        (["teams", "list"], "GET", "/team/list", []),
        (["teams", "available"], "GET", "/team/available", []),
        (["credentials", "list", "--format", "json"], "GET", "/credentials", {"credentials": []}),
    ],
)
def test_every_proxy_command_group_signs_in_on_a_401_and_reruns_with_the_new_key(
    requests_mock: Mocker,
    secret_vault_factory: type[FakeSecretVault],
    command: list[str],
    method: str,
    path: str,
    body: object,
) -> None:
    route: Final = requests_mock.register_uri(
        method, f"{BASE_URL}{path}", [{"status_code": 401, "json": REJECTED}, {"status_code": 200, "json": body}]
    )
    sign_in: Final = FakeBrowserSignIn(_credential())

    result: Final = _lite(command, secret_vault_factory(), sign_in)

    assert result.exit_code == 0, result.output
    assert sign_in.calls == [BASE_URL]
    assert _bearers(route) == [None, f"Bearer {SIGNED_IN_KEY}"]


def test_models_import_signs_in_instead_of_reporting_models_it_never_added(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault], tmp_path: Path
) -> None:
    models_file: Final = tmp_path / "models.yaml"
    models_file.write_text(
        "model_list:\n  - model_name: gpt-5.4-mini\n    litellm_params:\n      model: openai/gpt-5.4-mini\n"
    )
    route: Final = requests_mock.post(
        f"{BASE_URL}/model/new", [{"status_code": 401, "json": REJECTED}, {"status_code": 200, "json": {}}]
    )
    sign_in: Final = FakeBrowserSignIn(_credential())

    result: Final = _lite(["models", "import", str(models_file)], secret_vault_factory(), sign_in)

    assert result.exit_code == 0, result.output
    assert sign_in.calls == [BASE_URL]
    assert _bearers(route) == [None, f"Bearer {SIGNED_IN_KEY}"]


def test_a_401_partway_through_models_import_stops_without_signing_in_or_adding_models_twice(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault], tmp_path: Path
) -> None:
    models_file: Final = tmp_path / "models.yaml"
    models_file.write_text(
        "model_list:\n"
        "  - model_name: gpt-5.4-mini\n    litellm_params:\n      model: openai/gpt-5.4-mini\n"
        "  - model_name: gpt-5.4\n    litellm_params:\n      model: openai/gpt-5.4\n"
    )
    route: Final = requests_mock.post(
        f"{BASE_URL}/model/new", [{"status_code": 200, "json": {}}, {"status_code": 401, "json": REJECTED}]
    )
    sign_in: Final = FakeBrowserSignIn(_credential())

    result: Final = _lite(["models", "import", str(models_file)], secret_vault_factory(), sign_in)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "rejected the API key after 1 of 2 models were sent" in result.stderr
    assert sign_in.calls == []
    assert [request.json()["model_name"] for request in route.request_history] == ["gpt-5.4-mini", "gpt-5.4"]


def test_a_401_from_the_keys_import_source_never_signs_in_to_the_destination(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault]
) -> None:
    source: Final = "https://source.example.com"
    requests_mock.get(f"{source}/key/list", status_code=401, json=REJECTED)
    sign_in: Final = FakeBrowserSignIn(_credential())

    result: Final = _lite(
        ["keys", "import", "--source-base-url", source, "--source-api-key", "sk-source"],
        secret_vault_factory(),
        sign_in,
    )

    assert result.exit_code != 0
    assert sign_in.calls == []
    assert {request.netloc for request in requests_mock.request_history} == {"source.example.com"}


@pytest.mark.parametrize(
    ("global_args", "env"),
    [(["--api-key", "sk-explicit"], {}), ([], {"LITELLM_PROXY_API_KEY": "sk-explicit"})],
    ids=["flag", "env"],
)
def test_a_rejected_explicit_key_fails_cleanly_without_opening_a_browser(
    requests_mock: Mocker,
    secret_vault_factory: type[FakeSecretVault],
    monkeypatch: pytest.MonkeyPatch,
    global_args: list[str],
    env: dict[str, str],
) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    route: Final = requests_mock.post(f"{BASE_URL}/key/generate", status_code=401, json=REJECTED)
    vault: Final = secret_vault_factory()
    sign_in: Final = FakeBrowserSignIn(_credential())

    result: Final = _lite(["keys", "generate"], vault, sign_in, global_args)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "with HTTP 401 for the API key from --api-key or LITELLM_PROXY_API_KEY" in result.stderr
    assert "has the role this command needs" in result.stderr
    assert sign_in.calls == []
    assert _bearers(route) == ["Bearer sk-explicit"]
    assert [request.path for request in requests_mock.request_history] == ["/key/generate"]
    assert vault.writes == []


def test_a_failed_sign_in_names_the_reason_and_points_at_lite_login(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault]
) -> None:
    route: Final = requests_mock.post(f"{BASE_URL}/key/generate", status_code=401, json=REJECTED)
    vault: Final = secret_vault_factory()
    sign_in: Final = FakeBrowserSignIn(PkceFailure("timed out waiting for the browser sign-in to finish"))

    result: Final = _lite(["keys", "generate"], vault, sign_in)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "timed out waiting for the browser sign-in to finish" in result.stderr
    assert "Run `lite login --pkce` from a terminal on a machine with a browser" in result.stderr
    assert sign_in.calls == [BASE_URL]
    assert route.call_count == 1
    assert get_stored_api_key(BASE_URL, vault=vault) is None


@pytest.mark.parametrize(
    ("discovery_status", "advice"),
    [
        (404, "so it is likely older than v1.99.0. Upgrade it, run `lite login` if it uses SSO"),
        (503, "Run `lite login --pkce` from a terminal on a machine with a browser"),
    ],
    ids=["proxy_without_browser_sign_in", "proxy_erroring"],
)
def test_a_proxy_without_browser_sign_in_is_named_as_too_old_instead_of_pointing_at_lite_login_pkce(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault], discovery_status: int, advice: str
) -> None:
    requests_mock.post(f"{BASE_URL}/key/generate", status_code=401, json=REJECTED)
    requests_mock.get(f"{BASE_URL}/.well-known/litellm-cli-auth", status_code=discovery_status)

    def sign_in_with_a_picked_browser(base_url: str, echo: Callable[[str], None]) -> PkceCredential | PkceFailure:
        return pkce_browser_sign_in(
            base_url, echo, find_browser=lambda: webbrowser.GenericBrowser("w3m"), environ={"BROWSER": "w3m"}
        )

    result: Final = CliRunner().invoke(
        cli,
        ["--base-url", BASE_URL, "keys", "generate"],
        obj={"secret_vault": secret_vault_factory(), "browser_sign_in": sign_in_with_a_picked_browser},
    )

    assert result.exit_code == 1
    assert advice in result.stderr


def test_a_signed_in_key_the_proxy_also_rejects_is_not_followed_by_another_sign_in(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault]
) -> None:
    route: Final = requests_mock.post(f"{BASE_URL}/key/generate", status_code=401, json=REJECTED)
    sign_in: Final = FakeBrowserSignIn(_credential())

    result: Final = _lite(["keys", "generate"], secret_vault_factory(), sign_in)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "refused `cli keys generate` with HTTP 401 again after the browser sign-in" in result.stderr
    assert sign_in.calls == [BASE_URL]
    assert _bearers(route) == [None, f"Bearer {SIGNED_IN_KEY}"]


def _rejected_while_a_sibling_signs_in(vault):
    def respond(request, context):
        save_token(_classic_login("sk-sibling"), vault=vault)
        context.status_code = 401
        return REJECTED

    return respond


def test_a_key_a_sibling_process_stored_meanwhile_is_used_before_any_browser(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault]
) -> None:
    vault: Final = secret_vault_factory()
    save_token(_classic_login("sk-stale"), vault=vault)
    _key_check(requests_mock, 401)
    route: Final = requests_mock.post(
        f"{BASE_URL}/key/generate",
        [{"json": _rejected_while_a_sibling_signs_in(vault)}, {"status_code": 200, "json": GENERATED}],
    )
    sign_in: Final = FakeBrowserSignIn(_credential())

    result: Final = _lite(["keys", "generate"], vault, sign_in)

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == GENERATED
    assert sign_in.calls == []
    assert _bearers(route) == ["Bearer sk-stale", "Bearer sk-sibling"]


def test_a_sibling_key_the_proxy_also_rejects_falls_through_to_one_sign_in(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault]
) -> None:
    vault: Final = secret_vault_factory()
    save_token(_classic_login("sk-stale"), vault=vault)
    _key_check(requests_mock, 401)
    route: Final = requests_mock.post(
        f"{BASE_URL}/key/generate",
        [
            {"json": _rejected_while_a_sibling_signs_in(vault)},
            {"status_code": 401, "json": REJECTED},
            {"status_code": 200, "json": GENERATED},
        ],
    )
    sign_in: Final = FakeBrowserSignIn(_credential())

    result: Final = _lite(["keys", "generate"], vault, sign_in)

    assert result.exit_code == 0, result.output
    assert sign_in.calls == [BASE_URL]
    assert _bearers(route) == ["Bearer sk-stale", "Bearer sk-sibling", f"Bearer {SIGNED_IN_KEY}"]
    assert get_stored_api_key(BASE_URL, vault=vault) == SIGNED_IN_KEY


@pytest.mark.parametrize("key_check_status", [200, 404], ids=["virtual_key", "browser_login_without_a_key_row"])
def test_a_stored_login_the_proxy_accepts_but_a_command_refuses_reports_a_missing_role_without_a_browser(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault], key_check_status: int
) -> None:
    vault: Final = secret_vault_factory()
    save_token(_classic_login("sk-member"), vault=vault)
    requests_mock.get(f"{BASE_URL}/v1/models", status_code=200, json={"data": []})
    route: Final = requests_mock.get(f"{BASE_URL}/credentials", status_code=401, json=REJECTED)
    key_check: Final = _key_check(requests_mock, key_check_status)
    sign_in: Final = FakeBrowserSignIn(_credential())
    writes_before: Final = list(vault.writes)

    result: Final = _lite(["credentials", "list"], vault, sign_in)

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "accepts your stored login but refused `cli credentials list` with HTTP 401" in result.stderr
    assert "Set LITELLM_PROXY_API_KEY to a key that has it" in result.stderr
    assert sign_in.calls == []
    assert route.call_count == 1
    assert _bearers(key_check) == ["Bearer sk-member"]
    assert vault.writes == writes_before
    assert "sk-member" not in result.output


class SiblingSignsInWhileThisCommandWaits:
    def __init__(self, vault) -> None:
        self.vault = vault
        self.held = False

    def __call__(self) -> "SiblingSignsInWhileThisCommandWaits":
        return self

    def __enter__(self) -> None:
        save_token(_classic_login("sk-sibling"), vault=self.vault)
        self.held = True

    def __exit__(self, *exc_info: object) -> None:
        self.held = False


def test_a_command_that_waits_for_another_commands_sign_in_uses_its_login_instead_of_a_second_browser(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault]
) -> None:
    vault: Final = secret_vault_factory()
    route: Final = requests_mock.post(
        f"{BASE_URL}/key/generate", [{"status_code": 401, "json": REJECTED}, {"status_code": 200, "json": GENERATED}]
    )
    sign_in: Final = FakeBrowserSignIn(_credential())
    sibling: Final = SiblingSignsInWhileThisCommandWaits(vault)

    result: Final = CliRunner().invoke(
        cli,
        ["--base-url", BASE_URL, "keys", "generate"],
        obj={"secret_vault": vault, "browser_sign_in": sign_in, "sign_in_lock": sibling},
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == GENERATED
    assert sign_in.calls == []
    assert _bearers(route) == [None, "Bearer sk-sibling"]
    assert sibling.held is False


def test_help_never_signs_in(requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault]) -> None:
    sign_in: Final = FakeBrowserSignIn(_credential())

    result: Final = _lite(["keys", "generate", "--help"], secret_vault_factory(), sign_in)

    assert result.exit_code == 0
    assert "Generate a new API key" in result.stdout
    assert sign_in.calls == []
    assert requests_mock.call_count == 0


@pytest.mark.parametrize("status", [403, 500])
def test_an_error_other_than_401_never_signs_in(
    requests_mock: Mocker, secret_vault_factory: type[FakeSecretVault], status: int
) -> None:
    route: Final = requests_mock.post(f"{BASE_URL}/key/generate", status_code=status, json={"error": "refused"})
    sign_in: Final = FakeBrowserSignIn(_credential())

    result: Final = _lite(["keys", "generate"], secret_vault_factory(), sign_in)

    assert result.exit_code == 1
    assert f"Error: HTTP {status}" in result.stderr
    assert sign_in.calls == []
    assert route.call_count == 1
