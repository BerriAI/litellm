import json
import stat
import sys
import time

import pytest

from litellm.constants import CLI_JWT_EXPIRATION_HOURS
from litellm.litellm_core_utils.cli_keyring import (
    DISABLE_KEYRING_ENV_VAR,
    KEYRING_ACCOUNT,
    KEYRING_SERVICE,
    KeyringVault,
    SecretFound,
    SecretMissing,
    KeyringDisabled,
    KeyringNotInstalled,
    KeyringUnreachable,
    SecretStored,
)
from litellm.litellm_core_utils.cli_token_utils import (
    CliTokenRecord,
    clear_cli_token,
    get_cli_token_file_path,
    get_litellm_gateway_api_key,
    is_cli_token_fresh,
    load_cli_token,
    save_cli_token,
)

SERVER = "https://proxy.example.com"
OTHER_SERVER = "https://other-proxy.example.com"


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return tmp_path


def _token_file(home):
    return home / ".litellm" / "token.json"


def _write_legacy_file(home, **overrides):
    payload = {
        "base_url": SERVER,
        "key": "sk-legacy",
        "user_id": "u-1",
        "user_email": "user@example.com",
        "user_role": "cli",
        "timestamp": time.time(),
        **overrides,
    }
    path = _token_file(home)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    return path


def _write_metadata_only_file(home):
    """What a post-migration token.json looks like: everything except the secret material."""
    path = _token_file(home)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({"base_url": SERVER, "user_id": "u-1", "timestamp": time.time()}))
    path.chmod(0o600)
    return path


def _blob(base_url=SERVER, key="sk-vault", jwt_token=""):
    return json.dumps({"base_url": base_url, "key": key, "jwt_token": jwt_token})


class TestGetCliTokenFilePath:
    def test_points_at_the_home_config_file(self, isolated_home):
        assert get_cli_token_file_path() == str(isolated_home / ".litellm" / "token.json")

    def test_does_not_create_the_directory(self, isolated_home):
        """Merely asking for the path must not leave a directory behind, so an SDK import that
        never logs in cannot create a ~/.litellm on someone's machine."""
        get_cli_token_file_path()

        assert not (isolated_home / ".litellm").exists()


class TestLoadCliToken:
    def test_no_token_file_never_touches_the_keychain(self, isolated_home, secret_vault_factory):
        """The SDK calls this on machines that never ran `lite login`; it must not prompt for
        keychain access there."""
        vault = secret_vault_factory(blob=_blob())

        assert load_cli_token(vault=vault) is None
        assert vault.reads == 0

    def test_secret_comes_from_the_vault_when_the_file_holds_only_metadata(self, isolated_home, secret_vault_factory):
        _write_metadata_only_file(isolated_home)
        vault = secret_vault_factory(blob=_blob(key="sk-from-keychain"))

        record = load_cli_token(vault=vault)

        assert record.key == "sk-from-keychain"
        assert "sk-from-keychain" not in _token_file(isolated_home).read_text()

    def test_jwt_token_round_trips_through_the_vault(self, isolated_home, secret_vault_factory):
        _write_metadata_only_file(isolated_home)
        vault = secret_vault_factory(blob=_blob(key="sk-a", jwt_token="jwt-a"))

        record = load_cli_token(vault=vault)

        assert (record.key, record.jwt_token) == ("sk-a", "jwt-a")

    def test_legacy_plaintext_file_still_authenticates_and_is_migrated(self, isolated_home, secret_vault_factory):
        """A token.json written by an older `lite` keeps working, and reading it moves the secret
        into the keychain and scrubs it from disk."""
        path = _write_legacy_file(isolated_home)
        vault = secret_vault_factory()

        record = load_cli_token(vault=vault)

        assert record.key == "sk-legacy"
        assert json.loads(vault.blob)["key"] == "sk-legacy"
        on_disk = json.loads(path.read_text())
        assert "key" not in on_disk
        assert on_disk["user_email"] == "user@example.com"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_legacy_file_survives_a_vault_that_refuses_to_store(self, isolated_home, secret_vault_factory):
        """Scrubbing the only copy of the secret after a failed keychain write would log the user
        out for good."""
        path = _write_legacy_file(isolated_home)
        before = path.read_text()

        record = load_cli_token(vault=secret_vault_factory(writable=False))

        assert record.key == "sk-legacy"
        assert path.read_text() == before

    def test_a_secret_left_on_disk_outranks_a_stale_keychain_entry(self, isolated_home, secret_vault_factory):
        """A failed keychain write leaves the fresh secret on disk while the vault still holds the
        previous one; the next read must serve the file's secret and move it into the vault, never
        resurrect the stale key or scrub the only copy of the fresh one."""
        path = _write_legacy_file(isolated_home, key="sk-fresh")
        vault = secret_vault_factory(blob=_blob(key="sk-stale"))

        record = load_cli_token(vault=vault)

        assert record.key == "sk-fresh"
        assert json.loads(vault.blob)["key"] == "sk-fresh"
        assert "key" not in json.loads(path.read_text())

    def test_a_disk_secret_survives_when_the_stale_vault_refuses_the_rewrite(
        self, isolated_home, secret_vault_factory
    ):
        path = _write_legacy_file(isolated_home, key="sk-fresh")
        before = path.read_text()

        record = load_cli_token(vault=secret_vault_factory(blob=_blob(key="sk-stale"), writable=False))

        assert record.key == "sk-fresh"
        assert path.read_text() == before

    def test_legacy_file_survives_an_unreachable_vault_without_write_attempts(
        self, isolated_home, secret_vault_factory
    ):
        path = _write_legacy_file(isolated_home)
        before = path.read_text()
        vault = secret_vault_factory(available=False)

        record = load_cli_token(vault=vault)

        assert record.key == "sk-legacy"
        assert vault.writes == []
        assert path.read_text() == before

    def test_metadata_only_file_with_an_empty_vault_is_not_a_login(self, isolated_home, secret_vault_factory):
        _write_metadata_only_file(isolated_home)

        assert load_cli_token(vault=secret_vault_factory()) is None

    def test_metadata_only_file_with_an_unreachable_vault_reports_a_missing_secret(
        self, isolated_home, secret_vault_factory
    ):
        """The caller needs to tell "never logged in" apart from "locked keychain", so the record
        comes back with no key rather than as None."""
        _write_metadata_only_file(isolated_home)

        record = load_cli_token(vault=secret_vault_factory(available=False))

        assert record.key is None
        assert record.user_id == "u-1"

    def test_a_secret_minted_for_another_server_is_never_handed_out(self, isolated_home, secret_vault_factory):
        _write_metadata_only_file(isolated_home)

        assert load_cli_token(vault=secret_vault_factory(blob=_blob(base_url=OTHER_SERVER))) is None

    def test_a_secret_minted_for_another_server_loses_to_the_file(self, isolated_home, secret_vault_factory):
        _write_legacy_file(isolated_home)
        vault = secret_vault_factory(blob=_blob(base_url=OTHER_SERVER, key="sk-elsewhere"))

        record = load_cli_token(vault=vault)

        assert record.key == "sk-legacy"
        assert json.loads(vault.blob)["key"] == "sk-legacy"

    def test_unreadable_vault_blob_falls_back_to_the_file_secret(self, isolated_home, secret_vault_factory):
        _write_legacy_file(isolated_home)

        record = load_cli_token(vault=secret_vault_factory(blob="not json at all {{{"))

        assert record.key == "sk-legacy"

    def test_corrupt_token_file_is_not_a_login(self, isolated_home, secret_vault_factory):
        _token_file(isolated_home).parent.mkdir()
        _token_file(isolated_home).write_text("not json at all {{{")

        assert load_cli_token(vault=secret_vault_factory(blob=_blob())) is None


class TestGetLitellmGatewayApiKey:
    def test_returns_the_vault_secret_when_the_origin_matches(self, isolated_home, secret_vault_factory):
        _write_metadata_only_file(isolated_home)

        key = get_litellm_gateway_api_key(expected_base_url=SERVER, vault=secret_vault_factory(blob=_blob()))

        assert key == "sk-vault"

    def test_trailing_slash_on_the_expected_url_is_normalised(self, isolated_home, secret_vault_factory):
        _write_metadata_only_file(isolated_home)

        key = get_litellm_gateway_api_key(expected_base_url=SERVER + "/", vault=secret_vault_factory(blob=_blob()))

        assert key == "sk-vault"

    def test_origin_mismatch_returns_nothing_without_reading_the_keychain(self, isolated_home, secret_vault_factory):
        """Pointing the SDK at a different server must fail before the keychain is even consulted,
        so a hostile base_url cannot provoke an unlock prompt."""
        _write_legacy_file(isolated_home)
        vault = secret_vault_factory(blob=_blob())

        assert get_litellm_gateway_api_key(expected_base_url=OTHER_SERVER, vault=vault) is None
        assert vault.reads == 0

    def test_no_token_file_returns_nothing(self, isolated_home, secret_vault_factory):
        assert get_litellm_gateway_api_key(vault=secret_vault_factory(blob=_blob())) is None


class TestSaveCliToken:
    def test_secret_goes_to_the_keychain_and_never_to_the_file(self, isolated_home, secret_vault_factory):
        vault = secret_vault_factory()

        stored = save_cli_token(
            CliTokenRecord(base_url=SERVER, key="sk-new", user_id="u-1", timestamp=time.time()),
            vault=vault,
        )

        assert stored == SecretStored()
        assert "sk-new" not in _token_file(isolated_home).read_text()
        assert json.loads(vault.blob)["key"] == "sk-new"
        assert load_cli_token(vault=vault).key == "sk-new"

    def test_falls_back_to_the_owner_only_file_when_there_is_no_keychain(self, isolated_home, secret_vault_factory):
        stored = save_cli_token(
            CliTokenRecord(base_url=SERVER, key="sk-new", timestamp=time.time()),
            vault=secret_vault_factory(available=False),
        )

        path = _token_file(isolated_home)
        assert stored == KeyringUnreachable()
        assert json.loads(path.read_text())["key"] == "sk-new"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert list(path.parent.glob(".tmp-*")) == []

    def test_creates_the_config_directory_owner_only(self, isolated_home, secret_vault_factory):
        """A 0755 ~/.litellm lets any local process list, and in the fallback case read, the
        credential's directory."""
        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-new"), vault=secret_vault_factory())

        assert stat.S_IMODE((isolated_home / ".litellm").stat().st_mode) == 0o700

    def test_tightens_a_directory_left_group_readable_by_an_older_cli(self, isolated_home, secret_vault_factory):
        config_dir = isolated_home / ".litellm"
        config_dir.mkdir(mode=0o755)

        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-new"), vault=secret_vault_factory())

        assert stat.S_IMODE(config_dir.stat().st_mode) == 0o700

    def test_a_failed_write_leaves_the_previous_credential_intact(self, isolated_home, secret_vault_factory, monkeypatch):
        path = _write_legacy_file(isolated_home)
        before = path.read_text()

        def _explode(*args, **kwargs):
            raise TypeError("not serialisable")

        monkeypatch.setattr("litellm.litellm_core_utils.private_json.json.dump", _explode)

        with pytest.raises(TypeError):
            save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-new"), vault=secret_vault_factory(available=False))

        assert path.read_text() == before
        assert list(path.parent.glob(".tmp-*")) == []


class TestClearCliToken:
    def test_removes_the_credential_from_both_stores(self, isolated_home, secret_vault_factory):
        vault = secret_vault_factory()
        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-new"), vault=vault)

        assert clear_cli_token(vault=vault) is True
        assert vault.blob is None
        assert not _token_file(isolated_home).exists()
        assert load_cli_token(vault=vault) is None

    def test_reports_a_keychain_that_will_not_release_the_secret(self, isolated_home, secret_vault_factory):
        _write_legacy_file(isolated_home)
        vault = secret_vault_factory(blob=_blob(), erasable=False)

        assert clear_cli_token(vault=vault) is False
        assert not _token_file(isolated_home).exists()

    def test_is_safe_when_nothing_was_ever_stored(self, isolated_home, secret_vault_factory):
        assert clear_cli_token(vault=secret_vault_factory()) is True


class TestIsCliTokenFresh:
    def test_a_just_issued_token_is_fresh(self):
        assert is_cli_token_fresh(CliTokenRecord(timestamp=time.time())) is True

    def test_a_token_past_its_expiry_is_stale(self):
        stale = CliTokenRecord(timestamp=time.time() - (CLI_JWT_EXPIRATION_HOURS + 1) * 3600)

        assert is_cli_token_fresh(stale) is False

    def test_the_buffer_retires_a_token_just_before_it_expires(self):
        almost = CliTokenRecord(timestamp=time.time() - (CLI_JWT_EXPIRATION_HOURS * 3600 - 60))

        assert is_cli_token_fresh(almost, buffer_hours=0.1) is False


class _FakeKeyringModule:
    def __init__(self, stored=None, *, get_error=None, set_error=None, delete_error=None):
        self.stored = stored
        self.get_error = get_error
        self.set_error = set_error
        self.delete_error = delete_error
        self.calls = []

    def get_password(self, service_name, username):
        self.calls.append(("get", service_name, username))
        if self.get_error is not None:
            raise self.get_error
        return self.stored

    def set_password(self, service_name, username, password):
        self.calls.append(("set", service_name, username))
        if self.set_error is not None:
            raise self.set_error
        self.stored = password

    def delete_password(self, service_name, username):
        self.calls.append(("delete", service_name, username))
        if self.delete_error is not None:
            raise self.delete_error
        self.stored = None


@pytest.fixture
def install_fake_keyring(monkeypatch):
    def _install(fake):
        monkeypatch.delenv(DISABLE_KEYRING_ENV_VAR, raising=False)
        monkeypatch.setitem(sys.modules, "keyring", fake)
        return fake

    return _install


class TestKeyringVault:
    def test_round_trips_through_the_installed_keyring(self, install_fake_keyring):
        fake = install_fake_keyring(_FakeKeyringModule())
        vault = KeyringVault()

        assert vault.write("blob-1") == SecretStored()
        assert vault.read() == SecretFound("blob-1")
        assert vault.erase() is True
        assert vault.read() == SecretMissing()
        assert {call[1:] for call in fake.calls} == {(KEYRING_SERVICE, KEYRING_ACCOUNT)}

    def test_the_kill_switch_reports_no_keychain(self, monkeypatch):
        """`LITELLM_CLI_DISABLE_KEYRING` has to work without importing keyring, because keyring
        caches its backend on first use and cannot be reconfigured later. Erase still fails: a
        credential stored before the switch was set may be in the keychain, and with reads
        disabled `lite logout` cannot verify it is gone, so it must warn instead."""
        monkeypatch.setenv(DISABLE_KEYRING_ENV_VAR, "1")
        vault = KeyringVault()

        assert vault.read() == KeyringDisabled()
        assert vault.write("blob-1") == KeyringDisabled()
        assert vault.erase() is False

    def test_an_uninstalled_keyring_library_degrades_to_the_file(self, monkeypatch):
        """keyring is an optional extra, so the SDK must survive its absence rather than raise on
        the hot path."""
        monkeypatch.delenv(DISABLE_KEYRING_ENV_VAR, raising=False)
        monkeypatch.setitem(sys.modules, "keyring", None)
        vault = KeyringVault()

        assert vault.read() == KeyringNotInstalled()
        assert vault.write("blob-1") == KeyringNotInstalled()
        assert vault.erase() is True

    def test_a_locked_keychain_is_reported_not_raised(self, install_fake_keyring):
        install_fake_keyring(_FakeKeyringModule(get_error=RuntimeError("keyring is locked")))

        assert KeyringVault().read() == KeyringUnreachable()

    def test_a_refused_write_is_reported_not_raised(self, install_fake_keyring):
        install_fake_keyring(_FakeKeyringModule(set_error=RuntimeError("no backend")))

        assert KeyringVault().write("blob-1") == KeyringUnreachable()

    def test_a_refused_delete_is_reported_so_logout_can_warn(self, install_fake_keyring):
        install_fake_keyring(_FakeKeyringModule(stored="blob-1", delete_error=RuntimeError("locked")))

        assert KeyringVault().erase() is False

    def test_erasing_a_locked_keychain_is_a_failure(self, install_fake_keyring):
        install_fake_keyring(_FakeKeyringModule(get_error=RuntimeError("locked")))

        assert KeyringVault().erase() is False
