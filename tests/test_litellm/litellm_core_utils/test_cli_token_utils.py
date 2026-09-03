import errno
import json
import os
import stat
import sys
import tempfile
import threading
import time

import pytest

from litellm.constants import CLI_JWT_EXPIRATION_HOURS
from litellm.litellm_core_utils.cli_keyring import (
    DISABLE_KEYRING_ENV_VAR,
    KEYRING_ACCOUNT,
    KEYRING_PREFLIGHT_ACCOUNT,
    KEYRING_SERVICE,
    KeyringDisabled,
    KeyringDiscardsWrites,
    KeyringNotInstalled,
    KeyringUnreachable,
    KeyringVault,
    SecretErased,
    SecretFound,
    SecretMissing,
    SecretStored,
    SecretStranded,
)
from litellm.litellm_core_utils.cli_token_utils import (
    CliTokenRecord,
    CredentialNotCleared,
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


def _blob(base_url=SERVER, key="sk-vault", jwt_token="", timestamp=0.0, refresh_token=None):
    return json.dumps(
        {
            "base_url": base_url,
            "key": key,
            "jwt_token": jwt_token,
            "refresh_token": refresh_token,
            "timestamp": timestamp,
        }
    )


def _write_key_only_keychain_file(home, *, refresh_token="rt-live", timestamp=2000.0):
    """What the release that kept only the key in the keychain left on disk: metadata, plus the
    refresh token in the clear."""
    path = _token_file(home)
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps({"base_url": SERVER, "user_id": "u-1", "refresh_token": refresh_token, "timestamp": timestamp})
    )
    path.chmod(0o600)
    return path


_REAL_MKSTEMP = tempfile.mkstemp


class _MkstempThatNeedsTheOldFileGone:
    """A disk with exactly one token file's worth of room left on it.

    Staging a replacement needs room for a second file, which is what a full disk refuses. Removing
    the file already there is what gives that room back.
    """

    def __init__(self, path):
        self.path = path

    def __call__(self, *args, **kwargs):
        if self.path.exists():
            raise OSError(errno.ENOSPC, "No space left on device")
        return _REAL_MKSTEMP(*args, **kwargs)


_REAL_REPLACE = os.replace


def _refuse_replace(*args, **kwargs):
    raise OSError("device or resource busy")


class _ReplaceThatStartsRefusing:
    """`os.replace` standing in for a path that cannot be replaced yet: a file another process holds
    open on Windows, a directory that went read-only between staging and the rewrite."""

    def __init__(self):
        self.allowed = False

    def __call__(self, src, dst):
        if not self.allowed:
            raise OSError("device or resource busy")
        _REAL_REPLACE(src, dst)


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

    def test_the_refresh_token_round_trips_through_the_vault(self, isolated_home, secret_vault_factory):
        """A refresh token mints a fresh key from the proxy on demand, so it is the credential just
        as much as the key is, and it has to come back out of the keychain to be usable."""
        _write_metadata_only_file(isolated_home)
        vault = secret_vault_factory(blob=_blob(key="sk-a", refresh_token="rt-a"))

        record = load_cli_token(vault=vault)

        assert (record.key, record.refresh_token) == ("sk-a", "rt-a")

    def test_a_plaintext_refresh_token_is_moved_off_disk(self, isolated_home, secret_vault_factory):
        path = _write_legacy_file(isolated_home, refresh_token="rt-legacy")
        vault = secret_vault_factory()

        record = load_cli_token(vault=vault)

        assert record.refresh_token == "rt-legacy"
        assert "rt-legacy" not in path.read_text()
        assert json.loads(vault.blob)["refresh_token"] == "rt-legacy"

    def test_an_upgrade_that_left_the_refresh_token_on_disk_rejoins_it_with_the_key(
        self, isolated_home, secret_vault_factory
    ):
        """The release before this one took the key into the keychain and left the refresh token
        behind, so upgrading finds one sign-in split across both stores. The read has to end with
        the whole credential in the keychain, not with whichever half it happened to prefer."""
        path = _write_key_only_keychain_file(isolated_home)
        vault = secret_vault_factory(blob=_blob(key="sk-live", timestamp=2000.0))

        record = load_cli_token(vault=vault)

        assert (record.key, record.refresh_token) == ("sk-live", "rt-live")
        assert "rt-live" not in path.read_text()
        assert json.loads(vault.blob)["key"] == "sk-live"
        assert json.loads(vault.blob)["refresh_token"] == "rt-live"

    def test_a_superseded_refresh_token_on_disk_never_outlives_the_keychain(
        self, isolated_home, secret_vault_factory
    ):
        """Two stores, two sign-ins, and the newer one is in the keychain. Handing back its key with
        the older one's refresh token would build a credential neither store ever held, and would
        renew the login the user already replaced."""
        path = _write_legacy_file(isolated_home, key="sk-old", refresh_token="rt-old", timestamp=1000.0)
        vault = secret_vault_factory(blob=_blob(key="sk-new", refresh_token="rt-new", timestamp=2000.0))

        record = load_cli_token(vault=vault)

        assert (record.key, record.refresh_token) == ("sk-new", "rt-new")
        assert "rt-old" not in path.read_text()

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

    def test_a_login_the_file_could_not_record_is_the_one_that_gets_used(
        self, isolated_home, secret_vault_factory
    ):
        """A login the keychain took and the file could not be pointed at afterwards leaves the
        superseded secret sitting on disk in front of the fresh one. Serving the file's copy would
        put a credential the user just replaced, and may well have just revoked, back into every
        request, and would overwrite the keychain with it on the way past."""
        path = _write_legacy_file(isolated_home, key="sk-superseded", timestamp=1000.0)
        vault = secret_vault_factory(blob=_blob(key="sk-fresh", timestamp=2000.0))

        record = load_cli_token(vault=vault)

        assert record.key == "sk-fresh"
        assert record.timestamp == 2000.0
        assert json.loads(vault.blob)["key"] == "sk-fresh"
        assert "key" not in json.loads(path.read_text())

    def test_a_secret_written_to_disk_after_the_keychain_entry_still_wins(
        self, isolated_home, secret_vault_factory
    ):
        """The other direction of the same rule, which is the common one: a login that fell back to
        the file because the keychain refused it is newer than whatever the keychain kept."""
        path = _write_legacy_file(isolated_home, key="sk-fresh", timestamp=2000.0)
        vault = secret_vault_factory(blob=_blob(key="sk-stale", timestamp=1000.0))

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

    def test_the_refresh_token_goes_to_the_keychain_and_never_to_the_file(
        self, isolated_home, secret_vault_factory
    ):
        vault = secret_vault_factory()

        stored = save_cli_token(
            CliTokenRecord(base_url=SERVER, key="sk-new", refresh_token="rt-new", timestamp=time.time()),
            vault=vault,
        )

        assert stored == SecretStored()
        assert "rt-new" not in _token_file(isolated_home).read_text()
        assert json.loads(vault.blob)["refresh_token"] == "rt-new"
        assert load_cli_token(vault=vault).refresh_token == "rt-new"

    def test_the_refresh_token_falls_back_to_the_owner_only_file_with_the_key(
        self, isolated_home, secret_vault_factory
    ):
        """A machine with no keychain keeps the whole credential in the 0600 file, refresh token
        included, because a renewal that cannot be stored logs the user out on the next command."""
        vault = secret_vault_factory(available=False, failure=KeyringNotInstalled())

        save_cli_token(
            CliTokenRecord(base_url=SERVER, key="sk-new", refresh_token="rt-new", timestamp=time.time()),
            vault=vault,
        )

        assert json.loads(_token_file(isolated_home).read_text())["refresh_token"] == "rt-new"
        assert load_cli_token(vault=vault).refresh_token == "rt-new"

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

    def test_a_login_is_stamped_past_the_one_it_replaces_even_on_a_clock_that_went_back(
        self, isolated_home, secret_vault_factory
    ):
        """The stamp is what decides the keychain secret against the one on disk, so a login that
        carries an earlier wall clock than the login before it must not be filed as the older of
        the two."""
        _write_legacy_file(isolated_home, key="sk-old", timestamp=2000.0)
        vault = secret_vault_factory()

        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-new", timestamp=1000.0), vault=vault)

        assert json.loads(vault.blob)["timestamp"] > 2000.0

    def test_a_clock_that_went_back_does_not_hand_the_win_to_the_superseded_login(
        self, isolated_home, secret_vault_factory
    ):
        """The disk state a login reports as CredentialNotRecorded: the keychain took the new
        secret and the file still holds the previous one. Reading it back has to produce the login
        that was just made, and an earlier wall clock is no reason to serve the one it replaced."""
        _write_legacy_file(isolated_home, key="sk-superseded", timestamp=2000.0)
        vault = secret_vault_factory()

        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-fresh", timestamp=1000.0), vault=vault)
        _write_legacy_file(isolated_home, key="sk-superseded", timestamp=2000.0)

        assert load_cli_token(vault=vault).key == "sk-fresh"

    def test_a_login_on_a_clock_that_moved_forwards_keeps_its_own_time(
        self, isolated_home, secret_vault_factory
    ):
        """Pinning the stamp above the previous login is only ever a floor. The ordinary case has
        to record when the user actually signed in, because that is what decides expiry."""
        _write_legacy_file(isolated_home, key="sk-old", timestamp=1000.0)
        vault = secret_vault_factory()

        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-new", timestamp=2000.0), vault=vault)

        assert json.loads(vault.blob)["timestamp"] == 2000.0
        assert json.loads(_token_file(isolated_home).read_text())["timestamp"] == 2000.0

    def test_a_login_is_stamped_past_the_keychain_the_file_could_not_keep_up_with(
        self, isolated_home, secret_vault_factory
    ):
        """A login reported as CredentialNotRecorded leaves the keychain holding a later sign-in
        than the file names, so the file alone is no longer the floor. A later login on a clock
        that went back past that keychain entry still has to be the one served."""
        _write_legacy_file(isolated_home, key="sk-superseded", timestamp=1000.0)
        vault = secret_vault_factory(blob=_blob(key="sk-recorded", timestamp=2000.0), writable=False)

        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-fresh", timestamp=1500.0), vault=vault)

        assert load_cli_token(vault=vault).key == "sk-fresh"


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

    def test_a_rewrite_the_directory_refuses_is_finished_in_place(
        self, isolated_home, secret_vault_factory, monkeypatch
    ):
        """Staging can succeed and the rewrite still fail afterwards, which is the one window where
        both stores hold the credential. Shortening the file already there needs neither a second
        file nor a cooperative directory, so the move finishes rather than handing the keychain copy
        back and leaving the cleartext where it was."""
        path = _write_legacy_file(isolated_home)
        vault = secret_vault_factory()
        monkeypatch.setattr("litellm.litellm_core_utils.private_json.os.replace", _refuse_replace)

        record = load_cli_token(vault=vault)

        assert record.key == "sk-legacy"
        assert vault.blob is not None
        assert json.loads(path.read_text()).get("key") is None
        assert list(path.parent.glob(".tmp-*")) == []

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_a_rollback_the_keychain_refuses_is_finished_by_the_next_read(
        self, isolated_home, secret_vault_factory, monkeypatch
    ):
        """A file that will take neither a replacement nor an overwrite, and a keychain that will not
        give back what it just took, leave the credential in both stores. Nothing is lost by that,
        and nothing is abandoned either: the next read carries the move the rest of the way, so the
        duplicate outlives only the conditions that caused it."""
        path = _write_legacy_file(isolated_home)
        vault = secret_vault_factory(erasable=False)
        replace = _ReplaceThatStartsRefusing()
        monkeypatch.setattr("litellm.litellm_core_utils.private_json.os.replace", replace)
        path.chmod(0o400)

        assert load_cli_token(vault=vault).key == "sk-legacy"
        assert vault.blob is not None
        assert json.loads(path.read_text())["key"] == "sk-legacy"

        replace.allowed = True
        path.chmod(0o600)

        assert load_cli_token(vault=vault).key == "sk-legacy"
        assert json.loads(path.read_text()).get("key") is None


    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_a_rejoin_the_file_refuses_never_takes_the_key_with_it(
        self, isolated_home, secret_vault_factory, monkeypatch
    ):
        """Rolling the rejoined entry back would erase a key that was safely in the keychain before
        this read began, and the file it would fall back to is the one that has just refused to be
        rewritten. The duplicate refresh token stays until a later read can finish the move."""
        path = _write_key_only_keychain_file(isolated_home)
        vault = secret_vault_factory(blob=_blob(key="sk-live", timestamp=2000.0))
        monkeypatch.setattr("litellm.litellm_core_utils.private_json.os.replace", _refuse_replace)
        path.chmod(0o400)

        record = load_cli_token(vault=vault)

        assert (record.key, record.refresh_token) == ("sk-live", "rt-live")
        assert json.loads(vault.blob)["key"] == "sk-live"
        assert json.loads(vault.blob)["refresh_token"] == "rt-live"
        assert vault.erases == 0


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
        "failure", [KeyringDisabled(), KeyringUnreachable(), KeyringNotInstalled()]
    )
    def test_a_secret_in_the_file_is_no_evidence_about_a_keychain_that_exists(
        self, isolated_home, secret_vault_factory, failure
    ):
        """Store a secret in the keychain, sign in again while the keychain is unusable so the new
        secret lands in the file, then log out while it is still unusable. The file now carries its
        own secret and the first login's entry is still there, so reading the file as proof of a
        clean keychain reports a logout that did not happen.

        The three unusable states are the whole of what an erase can answer besides erased and
        stranded; a backend that keeps nothing it is given is something only a write finds out."""
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
        _write_legacy_file(isolated_home, refresh_token="rt-legacy")
        vault = secret_vault_factory(available=False, failure=KeyringUnreachable())

        clear_cli_token(vault=vault)

        left_on_disk = _token_file(isolated_home).read_text()
        assert "sk-legacy" not in left_on_disk
        assert "rt-legacy" not in left_on_disk

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

    def test_a_file_backed_login_cannot_vouch_for_a_keychain_no_package_can_reach(
        self, isolated_home, secret_vault_factory
    ):
        """Sign in with the keyring package installed, lose the package, then sign in again so the
        second secret lands in the file. The first login's entry outlives both, and the file that
        replaced it holds a secret of its own, which is the shape a logout must not read as proof
        that no keychain was ever involved."""
        vault = secret_vault_factory()
        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-keychain"), vault=vault)
        vault.available = False
        vault.failure = KeyringNotInstalled()
        save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-in-file"), vault=vault)

        assert clear_cli_token(vault=vault) == KeyringNotInstalled()
        assert vault.blob is not None
        assert "sk-in-file" not in _token_file(isolated_home).read_text()

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_a_file_that_gives_up_neither_its_secret_nor_itself_is_reported_not_raised(
        self, isolated_home, secret_vault_factory
    ):
        """A `~/.litellm` gone read-only refuses the staged rewrite and the removal, and a token file
        left read-only with it, as a `sudo lite login` leaves both, refuses the overwrite too. The
        credential is still readable on disk, which is the one thing logging out is for, so it has to
        come back as an answer rather than as a traceback the user has to read the code to
        understand."""
        path = _write_legacy_file(isolated_home)
        path.chmod(0o400)
        path.parent.chmod(0o500)
        try:
            outcome = clear_cli_token(vault=secret_vault_factory())
        finally:
            path.parent.chmod(0o700)
            path.chmod(0o600)

        assert isinstance(outcome, CredentialNotCleared)
        assert json.loads(path.read_text())["key"] == "sk-legacy"

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    def test_a_directory_that_takes_no_new_file_still_gives_up_the_secret_in_the_old_one(
        self, isolated_home, secret_vault_factory
    ):
        """A read-only `~/.litellm` accepts no replacement token file and no removal of the one it
        has, and still lets that one be shortened. The secret goes, the file stays as the note that
        the keychain went unchecked, and the logout after it warns again instead of reading the gap
        the removal would have left as a clean keychain.

        The key is a realistic length so the file genuinely shrinks: a rewrite in place that leaves
        the tail of the old contents behind hands the next run a file it cannot parse."""
        path = _write_legacy_file(isolated_home, key="sk-" + "a" * 700)
        vault = secret_vault_factory(available=False, failure=KeyringUnreachable())
        path.parent.chmod(0o500)
        try:
            assert clear_cli_token(vault=vault) == KeyringUnreachable()
            assert clear_cli_token(vault=vault) == KeyringUnreachable()
        finally:
            path.parent.chmod(0o700)

        assert json.loads(path.read_text()).get("key") is None

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_a_note_the_logout_had_to_remove_is_written_again_for_the_next_one(
        self, isolated_home, secret_vault_factory, monkeypatch
    ):
        """A full disk refuses the replacement file and a read-only token file refuses the rewrite
        in place, so the only way left to get the secret off disk is to remove the file carrying it.
        That file was also the note saying the keychain went unchecked, and its absence is what the
        next logout would read as a keychain already known to be clean.

        Removing it is what frees the room the replacement was refused for, so the note is written
        again on the way out and the logout after this one still warns."""
        path = _write_legacy_file(isolated_home)
        path.chmod(0o400)
        monkeypatch.setattr(
            "litellm.litellm_core_utils.private_json.tempfile.mkstemp",
            _MkstempThatNeedsTheOldFileGone(path),
        )
        vault = secret_vault_factory(available=False, failure=KeyringUnreachable())

        assert clear_cli_token(vault=vault) == KeyringUnreachable()
        assert clear_cli_token(vault=vault) == KeyringUnreachable()

        assert json.loads(path.read_text()).get("key") is None

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    def test_a_metadata_file_that_will_not_go_is_not_worth_alarming_the_user_over(
        self, isolated_home, secret_vault_factory
    ):
        """The secret was in the keychain and the keychain gave it up. What is stuck on disk names a
        credential that no longer exists, so the logout it describes really did happen."""
        path = _write_metadata_only_file(isolated_home)
        path.parent.chmod(0o500)
        try:
            outcome = clear_cli_token(vault=secret_vault_factory(blob=_blob()))
        finally:
            path.parent.chmod(0o700)

        assert outcome == SecretErased()

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

    def test_a_stamp_left_in_the_future_keeps_reporting_fresh_until_the_clock_catches_up(self):
        """The stamp both orders the two stores and drives this shortcut, so a store left stamped
        ahead of the clock hands that stamp to the next sign-in and keeps it looking fresh past the
        expiry the gateway will actually enforce. Pinning that here so the shared stamp cannot stop
        being a deliberate trade without this failing first."""
        ahead = CliTokenRecord(timestamp=time.time() + CLI_JWT_EXPIRATION_HOURS * 3600)

        assert is_cli_token_fresh(ahead) is True


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


class _KeychainHeldByABlockedWrite(_NeverAnsweringKeyringModule):
    """The same keychain, plus what the blocked write does to everything after it: the stuck call
    holds the keychain, so every later read blocks behind it too."""

    def get_password(self, service_name, username):
        self.calls.append(("get", service_name, username))
        if self.blocked.is_set():
            threading.Event().wait()
        return self.stored


def _answered_within(seconds, call):
    answers = []
    worker = threading.Thread(target=lambda: answers.append(call()), daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), f"{call.__qualname__} never returned"
    return answers[0]


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

    def test_a_keychain_that_stopped_answering_is_not_asked_again(self, install_fake_keyring):
        """The write that timed out is still holding the keychain when we give up on it, so the
        call after it is the one that hangs, and read has nothing to time out against. Anything
        resolving the credential more than once in a process hits that: an SDK client built twice
        pays the pre-flight timeout on the first build and never returns from the second."""
        install_fake_keyring(_KeychainHeldByABlockedWrite())
        vault = KeyringVault(preflight_timeout_seconds=0.05)

        assert vault.write("blob-1") == KeyringUnreachable()

        assert _answered_within(5, vault.read) == KeyringUnreachable()
        assert _answered_within(5, vault.erase) == KeyringUnreachable()
        assert _answered_within(5, lambda: vault.write("blob-2")) == KeyringUnreachable()

    def test_a_keychain_that_stopped_answering_leaves_the_credential_in_the_file(
        self, isolated_home, install_fake_keyring
    ):
        """The end of the same story: giving up on the keychain has to leave a login that still
        works, and loading it back must not go asking the keychain that already stopped answering."""
        install_fake_keyring(_KeychainHeldByABlockedWrite())
        vault = KeyringVault(preflight_timeout_seconds=0.05)

        outcome = save_cli_token(CliTokenRecord(base_url=SERVER, key="sk-only-copy"), vault=vault)

        assert outcome == KeyringUnreachable()
        assert _answered_within(5, lambda: load_cli_token(vault=vault)).key == "sk-only-copy"

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


class TestIsCliTokenFreshWithExpiresAt:
    """A ``lite login --pkce`` record carries the proxy's own ``expires_at``, which wins
    over the age-based guess made from ``timestamp``."""

    def test_future_expiry_is_fresh(self):
        from litellm.litellm_core_utils.cli_token_utils import is_cli_token_fresh

        assert is_cli_token_fresh({"expires_at": time.time() + 3600, "timestamp": 0}) is True

    def test_expiry_inside_the_buffer_is_stale(self):
        from litellm.litellm_core_utils.cli_token_utils import is_cli_token_fresh

        assert is_cli_token_fresh({"expires_at": time.time() + 100}) is False
        assert is_cli_token_fresh({"expires_at": time.time() + 100}, buffer_hours=0) is True

    def test_past_expiry_is_stale_even_with_a_fresh_timestamp(self):
        from litellm.litellm_core_utils.cli_token_utils import is_cli_token_fresh

        assert is_cli_token_fresh({"expires_at": time.time() - 1, "timestamp": time.time()}) is False

    def test_non_numeric_expiry_falls_back_to_the_timestamp(self):
        from litellm.litellm_core_utils.cli_token_utils import is_cli_token_fresh

        assert is_cli_token_fresh({"expires_at": "soon", "timestamp": time.time()}) is True
