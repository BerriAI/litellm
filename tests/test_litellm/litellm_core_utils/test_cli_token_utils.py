import json
import os
import stat
import sys
import threading
import time

import pytest

from litellm.constants import CLI_JWT_EXPIRATION_HOURS
from litellm.litellm_core_utils.cli_keyring import (
    DISABLE_KEYRING_ENV_VAR,
    KEYRING_ACCOUNT,
    KEYRING_PREFLIGHT_ACCOUNT,
    KEYRING_SERVICE,
    KeyringVault,
    SecretFound,
    SecretMissing,
    KeyringDisabled,
    KeyringDiscardsWrites,
    KeyringNotInstalled,
    KeyringUnreachable,
    SecretErased,
    SecretStored,
    SecretStranded,
)
from litellm.litellm_core_utils.cli_token_utils import (
    CliTokenRecord,
    CredentialNotRecorded,
    CredentialNotSaved,
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

    def test_migration_tightens_a_world_readable_legacy_file(self, isolated_home, secret_vault_factory):
        """An older `lite`, a loose umask, or a restored backup can leave token.json readable by
        every account on the box. Migrating it must not preserve those permissions."""
        path = _write_legacy_file(isolated_home)
        path.chmod(0o644)

        load_cli_token(vault=secret_vault_factory())

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

    def test_a_token_file_that_is_not_text_is_not_a_login(self, isolated_home, secret_vault_factory):
        """A truncated write or a half-synced backup can leave bytes that are not UTF-8 at all.
        Reading them must fail the way an absent file does, not crash every `lite` command."""
        path = _token_file(isolated_home)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xff\xfe not utf-8 at all")

        assert load_cli_token(vault=secret_vault_factory()) is None

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

    def test_a_credential_no_store_would_keep_is_reported_rather_than_raised(
        self, isolated_home, secret_vault_factory, monkeypatch
    ):
        """`lite login` catches whatever escapes here and calls it an authentication failure, which
        is the one thing that did not happen: the proxy minted a real credential. Saying so lets the
        user act on the actual problem instead of retrying a sign-in that already worked."""

        def _explode(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr("litellm.litellm_core_utils.private_json.json.dump", _explode)

        outcome = save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-new"), vault=secret_vault_factory())

        assert isinstance(outcome, CredentialNotSaved)
        assert "read-only file system" in outcome.detail

    def test_a_file_that_will_not_be_written_stops_the_save_before_the_keychain_is_touched(
        self, isolated_home, secret_vault_factory, monkeypatch
    ):
        """The token file is what makes a keychain entry findable again, so it is staged first.
        Handing the keychain a secret and only then finding out that nothing will point at it
        would strand a live credential under a machine with no idea it is there."""
        vault = secret_vault_factory()

        def _explode(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr("litellm.litellm_core_utils.private_json.json.dump", _explode)

        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-new"), vault=vault)

        assert vault.blob is None

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    def test_a_login_that_cannot_be_saved_leaves_the_working_one_alone(
        self, isolated_home, secret_vault_factory
    ):
        """Signing in again on a machine whose ~/.litellm has gone read-only must not cost the user
        the credential they already had. Overwriting the keychain and then failing to record it, or
        undoing that write afterwards, would take a login that still works out from under them."""
        _write_legacy_file(isolated_home, key=None)
        vault = secret_vault_factory(blob=_blob(key="sk-in-use"))
        path = _token_file(isolated_home)
        path.parent.chmod(0o500)
        try:
            outcome = save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-new"), vault=vault)
        finally:
            path.parent.chmod(0o700)

        assert isinstance(outcome, CredentialNotSaved)
        assert json.loads(vault.blob)["key"] == "sk-in-use"
        assert load_cli_token(vault=vault).key == "sk-in-use"

    def test_a_keychain_write_the_file_cannot_be_pointed_at_is_reported_as_that(
        self, isolated_home, secret_vault_factory
    ):
        """Staging the file can succeed and the replacement still fail, and that is the one path
        where the keychain already took the new secret. Reporting it as a save that kept nothing
        would send the user looking for a credential that is sitting in their keychain."""
        vault = secret_vault_factory()
        path = _token_file(isolated_home)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()

        outcome = save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-new"), vault=vault)

        assert isinstance(outcome, CredentialNotRecorded)
        assert json.loads(vault.blob)["key"] == "sk-new"

    def test_the_credential_the_file_cannot_name_is_left_in_the_keychain(
        self, isolated_home, secret_vault_factory
    ):
        """The keychain holds one entry, so the secret that was there went the moment this one
        landed. Taking the new one back out would turn a login this machine may still be able to
        use into no login at all, and it cannot restore the old one either way."""
        vault = secret_vault_factory(blob=_blob(key="sk-in-use"))
        path = _token_file(isolated_home)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()

        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-new"), vault=vault)

        assert vault.blob is not None

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


class TestScrubFailure:
    """A keychain that took the secret while the file kept it is the worst of both stores: the
    credential is live, it is in cleartext on disk, and every command reports success."""

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    def test_a_file_that_will_not_give_its_copy_up_rolls_the_vault_write_back(
        self, isolated_home, secret_vault_factory
    ):
        """Handing the keychain a copy without taking the file's away leaves the credential live in
        two stores instead of one. A directory that permits neither the rewrite nor the delete, a
        root-owned ~/.litellm left behind by a `sudo lite login`, must widen nothing."""
        path = _write_legacy_file(isolated_home)
        vault = secret_vault_factory()
        path.parent.chmod(0o500)
        try:
            record = load_cli_token(vault=vault)
        finally:
            path.parent.chmod(0o700)

        assert record.key == "sk-legacy"
        assert json.loads(path.read_text())["key"] == "sk-legacy"
        assert vault.blob is None

    def test_a_full_disk_stops_the_migration_before_the_keychain_is_handed_anything(
        self, isolated_home, secret_vault_factory, monkeypatch
    ):
        """The scrubbed file is staged first precisely so this is knowable in advance. A disk that
        cannot take the rewrite leaves the credential where it already was, in one store."""
        path = _write_legacy_file(isolated_home)
        vault = secret_vault_factory()

        def _explode(*args, **kwargs):
            raise OSError("no space left on device")

        monkeypatch.setattr("litellm.litellm_core_utils.private_json.json.dump", _explode)

        record = load_cli_token(vault=vault)

        assert record.key == "sk-legacy"
        assert vault.blob is None
        assert json.loads(path.read_text())["key"] == "sk-legacy"
        assert list(path.parent.glob(".tmp-*")) == []


class TestClearCliToken:
    def test_removes_the_credential_from_both_stores(self, isolated_home, secret_vault_factory):
        vault = secret_vault_factory()
        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-new"), vault=vault)

        assert clear_cli_token(vault=vault) == SecretErased()
        assert vault.blob is None
        assert not _token_file(isolated_home).exists()
        assert load_cli_token(vault=vault) is None

    def test_reports_a_keychain_that_will_not_release_the_secret(self, isolated_home, secret_vault_factory):
        _write_legacy_file(isolated_home)
        vault = secret_vault_factory(blob=_blob(), erasable=False)

        assert clear_cli_token(vault=vault) == SecretStranded()
        assert not _token_file(isolated_home).exists()

    def test_a_keychain_that_will_not_release_the_secret_still_ends_the_local_login(
        self, isolated_home, secret_vault_factory
    ):
        """The warning this returns says the machine is logged out locally and the keychain entry is
        what is left over. Keeping the file that names that entry makes the first half untrue: every
        later command reads the credential straight back out of the keychain and keeps working."""
        _write_metadata_only_file(isolated_home)
        vault = secret_vault_factory(blob=_blob(), erasable=False)

        assert clear_cli_token(vault=vault) == SecretStranded()
        assert load_cli_token(vault=vault) is None

    @pytest.mark.parametrize(
        "failure", [KeyringDisabled(), KeyringUnreachable(), KeyringDiscardsWrites()]
    )
    def test_a_secret_in_the_file_is_no_evidence_about_a_keychain_that_exists(
        self, isolated_home, secret_vault_factory, failure
    ):
        """Store a secret in the keychain, sign in again while the keychain is unusable so the new
        secret lands in the file, then log out while it is still unusable. The file now carries its
        own secret and the first login's entry is still there, so reading the file as proof of a
        clean keychain reports a logout that did not happen."""
        _write_legacy_file(isolated_home)
        vault = secret_vault_factory(available=False, failure=failure)

        assert clear_cli_token(vault=vault) == failure
        assert json.loads(_token_file(isolated_home).read_text()).get("key") is None

    def test_a_second_logout_still_reports_the_keychain_it_could_not_clear(
        self, isolated_home, secret_vault_factory
    ):
        """The first logout deletes the file and tells the user to run it again once the keychain is
        reachable. If the second run reads that missing file as proof of a clean keychain, the advice
        turns into the very false all-clear it was issued to prevent."""
        _write_metadata_only_file(isolated_home)
        vault = secret_vault_factory(available=False, failure=KeyringUnreachable())

        assert clear_cli_token(vault=vault) == KeyringUnreachable()
        assert clear_cli_token(vault=vault) == KeyringUnreachable()

    def test_logout_from_an_install_without_keyring_does_not_claim_the_keychain_is_clear(
        self, isolated_home, secret_vault_factory
    ):
        """A file holding only metadata put its secret in a keychain by definition. Losing the
        package that reaches it does not take the entry with it, so this cannot report success."""
        _write_metadata_only_file(isolated_home)
        vault = secret_vault_factory(available=False, failure=KeyringNotInstalled())

        assert clear_cli_token(vault=vault) == KeyringNotInstalled()
        assert json.loads(_token_file(isolated_home).read_text()).get("key") is None

    def test_a_logout_that_cannot_clear_the_keychain_keeps_the_record_that_it_has_to(
        self, isolated_home, secret_vault_factory
    ):
        """The file left behind holds no secret. It is what a later run reads to tell a machine with
        a credential it cannot reach apart from one that never had a login, which is the difference
        between warning the user and inventing a credential for them to worry about."""
        _write_metadata_only_file(isolated_home)
        vault = secret_vault_factory(available=False, failure=KeyringUnreachable())

        clear_cli_token(vault=vault)

        assert json.loads(_token_file(isolated_home).read_text()).get("key") is None

    def test_a_logout_that_cannot_clear_the_keychain_still_takes_the_file_secret_away(
        self, isolated_home, secret_vault_factory
    ):
        """Keeping a record of the unreachable keychain must never mean keeping the cleartext copy
        the user just asked to be rid of."""
        _write_legacy_file(isolated_home)
        vault = secret_vault_factory(available=False, failure=KeyringUnreachable())

        clear_cli_token(vault=vault)

        assert "sk-legacy" not in _token_file(isolated_home).read_text()

    def test_a_repeat_logout_never_answers_its_own_warning_with_an_all_clear(
        self, isolated_home, secret_vault_factory
    ):
        """Sign in while the keychain works, sign in again once it has gone out of reach so the
        second secret lands in the file, then log out twice. The first logout cannot say the first
        login's entry is gone, and says so. If the second one reads the file the first one took
        away as proof of a clean keychain, it retracts that warning while the credential behind it
        is still live."""
        vault = secret_vault_factory()
        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-first"), vault=vault)
        vault.available = False
        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-second"), vault=vault)

        assert clear_cli_token(vault=vault) == KeyringUnreachable()
        assert clear_cli_token(vault=vault) == KeyringUnreachable()
        assert vault.blob is not None
        assert "sk-second" not in _token_file(isolated_home).read_text()

    @pytest.mark.parametrize("failure", [KeyringNotInstalled(), KeyringDisabled(), KeyringUnreachable()])
    def test_logging_out_of_a_machine_that_never_logged_in_invents_nothing_to_warn_about(
        self, isolated_home, secret_vault_factory, failure
    ):
        """`lite logout` with no token file has nothing to end. Warning that a credential may be
        stranded in a keychain it cannot check sends the user after something that was never there,
        and `pip install keyring` will not make it appear."""
        vault = secret_vault_factory(available=False, failure=failure)

        assert clear_cli_token(vault=vault) == SecretErased()

    def test_a_file_backed_login_logs_out_quietly_without_keyring(self, isolated_home, secret_vault_factory):
        """The complement, and the one inference the file does support: nothing here can reach a
        keychain without the package, so an install that lacks it and a file that still holds its
        own secret between them account for the whole credential."""
        _write_legacy_file(isolated_home)
        vault = secret_vault_factory(available=False, failure=KeyringNotInstalled())

        assert clear_cli_token(vault=vault) == SecretErased()
        assert not _token_file(isolated_home).exists()

    def test_is_safe_when_nothing_was_ever_stored(self, isolated_home, secret_vault_factory):
        assert clear_cli_token(vault=secret_vault_factory()) == SecretErased()


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
    def __init__(self, stored=None, *, get_error=None, set_error=None, delete_error=None, discard=False):
        self.stored = stored
        self.get_error = get_error
        self.set_error = set_error
        self.delete_error = delete_error
        self.discard = discard
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
        if self.discard or username != KEYRING_ACCOUNT:
            return
        self.stored = password

    def delete_password(self, service_name, username):
        self.calls.append(("delete", service_name, username))
        if self.delete_error is not None:
            raise self.delete_error
        if username == KEYRING_ACCOUNT:
            self.stored = None


class _NeverAnsweringKeyringModule(_FakeKeyringModule):
    """A keychain whose writes block instead of returning, the way macOS does under a HOME that
    has no usable login keychain."""

    def __init__(self):
        super().__init__()
        self.blocked = threading.Event()

    def set_password(self, service_name, username, password):
        self.calls.append(("set", service_name, username))
        self.blocked.set()
        threading.Event().wait()


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
        assert vault.erase() == SecretErased()
        assert vault.read() == SecretMissing()
        assert {call[1] for call in fake.calls} == {KEYRING_SERVICE}
        assert {call[2] for call in fake.calls} == {KEYRING_ACCOUNT, KEYRING_PREFLIGHT_ACCOUNT}

    def test_the_kill_switch_reports_no_keychain(self, monkeypatch):
        """`LITELLM_CLI_DISABLE_KEYRING` has to work without importing keyring, because keyring
        caches its backend on first use and cannot be reconfigured later. Erase still fails: a
        credential stored before the switch was set may be in the keychain, and with reads
        disabled `lite logout` cannot verify it is gone, so it must say so instead."""
        monkeypatch.setenv(DISABLE_KEYRING_ENV_VAR, "1")
        vault = KeyringVault()

        assert vault.read() == KeyringDisabled()
        assert vault.write("blob-1") == KeyringDisabled()
        assert vault.erase() == KeyringDisabled()

    def test_an_uninstalled_keyring_library_degrades_to_the_file(self, monkeypatch):
        """keyring is an optional extra, so the SDK must survive its absence rather than raise on
        the hot path. Erase cannot succeed: the entry belongs to the OS and outlives the package,
        so an install without it is not evidence that the keychain is empty."""
        monkeypatch.delenv(DISABLE_KEYRING_ENV_VAR, raising=False)
        monkeypatch.setitem(sys.modules, "keyring", None)
        vault = KeyringVault()

        assert vault.read() == KeyringNotInstalled()
        assert vault.write("blob-1") == KeyringNotInstalled()
        assert vault.erase() == KeyringNotInstalled()

    def test_a_locked_keychain_is_reported_not_raised(self, install_fake_keyring):
        install_fake_keyring(_FakeKeyringModule(get_error=RuntimeError("keyring is locked")))

        assert KeyringVault().read() == KeyringUnreachable()

    def test_a_refused_write_is_reported_not_raised(self, install_fake_keyring):
        install_fake_keyring(_FakeKeyringModule(set_error=RuntimeError("no backend")))

        assert KeyringVault().write("blob-1") == KeyringUnreachable()

    def test_a_refused_delete_is_reported_so_logout_can_warn(self, install_fake_keyring):
        install_fake_keyring(_FakeKeyringModule(stored="blob-1", delete_error=RuntimeError("locked")))

        assert KeyringVault().erase() == SecretStranded()

    def test_a_backend_that_keeps_nothing_is_not_a_successful_write(self, install_fake_keyring):
        """keyring's null backend accepts every write, stores nothing, and raises nothing to say so.
        Taking its silence for success is how a credential gets deleted: the caller drops its own
        copy on our word. Only reading the value back tells the two apart."""
        fake = install_fake_keyring(_FakeKeyringModule(discard=True))

        assert KeyringVault().write("blob-1") == KeyringDiscardsWrites()
        assert fake.stored is None

    def test_a_keychain_that_never_answers_does_not_hang_the_login(self, install_fake_keyring):
        """macOS derives the login keychain from `$HOME`, and `set_password` under a HOME with no
        usable one blocks forever with no timeout of its own. Containers, CI images, `sudo -H`, and
        service accounts all run there, and `lite login` never touched a keychain before this, so a
        sign-in that simply never returns would be a new way for it to fail."""
        fake = install_fake_keyring(_NeverAnsweringKeyringModule())
        vault = KeyringVault(preflight_timeout_seconds=0.2)

        started = time.monotonic()
        outcome = vault.write("blob-1")

        assert outcome == KeyringUnreachable()
        assert time.monotonic() - started < 5
        assert fake.blocked.is_set()

    def test_a_keychain_that_never_answers_is_never_handed_the_credential(self, install_fake_keyring):
        """Giving up on the write is only safe if the secret was never the thing being written. A
        blocked call can still land later, and a keychain copy nobody waited for would sit beside
        the file copy the user was told about."""
        fake = install_fake_keyring(_NeverAnsweringKeyringModule())

        KeyringVault(preflight_timeout_seconds=0.2).write("blob-1")

        assert [call[2] for call in fake.calls] == [KEYRING_PREFLIGHT_ACCOUNT]

    def test_a_login_survives_a_keychain_that_never_answers(self, isolated_home, install_fake_keyring):
        """The end of the same story: the credential still has to be usable afterwards."""
        install_fake_keyring(_NeverAnsweringKeyringModule())

        outcome = save_cli_token(
            CliTokenRecord(base_url=SERVER, key="sk-only-copy"),
            vault=KeyringVault(preflight_timeout_seconds=0.2),
        )

        assert outcome == KeyringUnreachable()
        assert json.loads(_token_file(isolated_home).read_text())["key"] == "sk-only-copy"

    def test_the_real_null_backend_is_rejected(self, monkeypatch):
        """Pinned against the actual library rather than the double above, because the whole risk is
        that upstream's no-op write looks exactly like a successful one."""
        keyring = pytest.importorskip("keyring")
        null_backend = pytest.importorskip("keyring.backends.null")
        monkeypatch.delenv(DISABLE_KEYRING_ENV_VAR, raising=False)
        previous = keyring.get_keyring()
        keyring.set_keyring(null_backend.Keyring())
        try:
            assert KeyringVault().write("blob-1") == KeyringDiscardsWrites()
        finally:
            keyring.set_keyring(previous)

    def test_a_credential_survives_a_backend_that_keeps_nothing(
        self, isolated_home, install_fake_keyring
    ):
        """The end of the same story: the credential must still be usable afterwards. Reporting the
        discard is only worth anything if the token file then keeps the copy the keychain refused."""
        install_fake_keyring(_FakeKeyringModule(discard=True))

        outcome = save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-only-copy"))

        assert outcome == KeyringDiscardsWrites()
        assert json.loads(_token_file(isolated_home).read_text())["key"] == "sk-only-copy"
        assert load_cli_token().key == "sk-only-copy"

    def test_erasing_a_locked_keychain_is_a_failure(self, install_fake_keyring):
        install_fake_keyring(_FakeKeyringModule(get_error=RuntimeError("locked")))

        assert KeyringVault().erase() == KeyringUnreachable()
