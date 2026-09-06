"""Two engines standing in for two uvicorn workers that read the same assertion and share one
``FileTokenStore``: an issuer that accepts each assertion once must see one exchange per assertion."""

import errno
import json
import os
import stat
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest

from pydantic import SecretStr

from litellm.llms.base_llm.auth.shared_token_store import (
    CACHE_DIR_ENV,
    FileTokenStore,
    StoredToken,
    default_shared_token_store,
)
from litellm.llms.base_llm.auth.token_exchange import JwtBearerTokenExchangeEngine
from litellm.llms.base_llm.auth.types import MintedToken, TokenEndpointError
from tests.test_litellm.llms.base_llm.auth.test_token_exchange import (
    DEFAULT_ASSERTION,
    DEFAULT_REF,
    FakeClock,
    ManualExecutor,
    RecordingMetricsSink,
    ScriptedPoster,
    make_spec,
    token_response,
)

real_write_bytes: Final = Path.write_bytes


class SingleUsePoster:
    """Mints for an assertion it has never seen and answers 401 to any assertion sent a second time,
    which is how an issuer enforcing single-use ``jti`` behaves."""

    def __init__(self, token: str = "sk-ant-oat01-minted", expires_in: int = 3600) -> None:
        self.requests: list[dict] = []
        self._token = token
        self._expires_in = expires_in

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        body = json.loads(content)
        seen_before = any(prior["assertion"] == body["assertion"] for prior in self.requests)
        self.requests.append(body)
        if seen_before:
            return httpx.Response(401, json={"error": "invalid_grant"})
        return token_response(f"{self._token}-{len(self.requests)}", expires_in=self._expires_in)


def store_engine(
    poster,
    store: FileTokenStore,
    *,
    reader: Mapping[str, str] | None = None,
    clock: FakeClock | None = None,
    wall_clock: Callable[[], float] | None = None,
) -> JwtBearerTokenExchangeEngine:
    return JwtBearerTokenExchangeEngine(
        poster=poster,
        assertion_reader=(reader if reader is not None else {DEFAULT_REF: DEFAULT_ASSERTION}).get,
        clock=clock if clock is not None else FakeClock(),
        refresh_executor=ManualExecutor(),
        metrics_sink=RecordingMetricsSink(),
        shared_store=store,
        wall_clock=wall_clock if wall_clock is not None else FakeClock(1_700_000_000.0),
    )


def minted(result: object) -> MintedToken:
    assert isinstance(result, MintedToken), result
    return result


def stored_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.json"))


def test_second_worker_reuses_the_first_workers_token_without_a_post(tmp_path: Path):
    poster = SingleUsePoster()
    store = FileTokenStore(tmp_path)
    first = minted(store_engine(poster, store).get_token(make_spec()))

    second = minted(store_engine(poster, store).get_token(make_spec()))

    assert second.access_token.get_secret_value() == first.access_token.get_secret_value()
    assert len(poster.requests) == 1


def test_a_minted_assertion_never_reaches_the_shared_store(tmp_path: Path):
    """internal_issuer and keycloak mint a fresh assertion per exchange, so no other worker ever holds
    the same one and a stored token could never be matched back. Writing a live token to disk for a
    lookup that cannot succeed is exposure that buys nothing."""
    poster = SingleUsePoster()
    store = FileTokenStore(tmp_path)
    assertions = iter(("minted-jwt-1", "minted-jwt-2"))
    spec = make_spec(assertion_source=lambda: next(assertions))

    first = minted(store_engine(poster, store).get_token(spec))
    second = minted(store_engine(poster, store).get_token(spec))

    assert stored_files(tmp_path) == [], "a per-exchange assertion must keep its token off disk"
    assert [request["assertion"] for request in poster.requests] == ["minted-jwt-1", "minted-jwt-2"]
    assert second.access_token.get_secret_value() != first.access_token.get_secret_value()


def test_a_failed_write_leaves_no_staging_file_holding_a_live_token(tmp_path: Path):
    """Nothing ever sweeps this directory, so a staging file a failed write leaves behind would keep a
    working token readable on disk for as long as the pod lives."""
    store = FileTokenStore(tmp_path)
    (tmp_path / "occupied.json").mkdir()

    store.save(
        "occupied",
        StoredToken(access_token=SecretStr("sk-ant-oat01-live"), expires_at_epoch=None, assertion_sha256="sha"),
    )

    leaked = [path for path in tmp_path.rglob("*") if path.is_file() and "sk-ant-oat01-live" in path.read_text()]
    assert leaked == [], "a staged token file survived the failed write"
    assert store.load("occupied") is None


def test_a_write_that_only_fails_on_close_leaves_no_staging_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A token is small enough to sit in the handle's buffer until it closes, so a full disk surfaces
    at close rather than at ``write()``, and the staging file left behind would still hold the token."""

    def write_bytes_then_run_out_of_space(path: Path, data: bytes) -> int:
        real_write_bytes(path, data)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "write_bytes", write_bytes_then_run_out_of_space)
    store = FileTokenStore(tmp_path)

    store.save(
        "closing",
        StoredToken(access_token=SecretStr("sk-ant-oat01-live"), expires_at_epoch=None, assertion_sha256="sha"),
    )

    leaked = [path for path in tmp_path.rglob("*") if path.is_file() and "sk-ant-oat01-live" in path.read_text()]
    assert leaked == [], "a staged token file survived the close that failed"
    assert store.load("closing") is None


def test_a_rotated_assertion_buys_a_fresh_token_that_other_workers_pick_up(tmp_path: Path):
    poster = SingleUsePoster()
    store = FileTokenStore(tmp_path)
    assertions = {DEFAULT_REF: "jwt-v1"}
    first = minted(store_engine(poster, store, reader=assertions).get_token(make_spec()))

    assertions[DEFAULT_REF] = "jwt-v2"
    rotated = minted(store_engine(poster, store, reader=assertions).get_token(make_spec()))
    follower = minted(store_engine(poster, store, reader=assertions).get_token(make_spec()))

    assert rotated.access_token.get_secret_value() != first.access_token.get_secret_value()
    assert follower.access_token.get_secret_value() == rotated.access_token.get_secret_value()
    assert [request["assertion"] for request in poster.requests] == ["jwt-v1", "jwt-v2"]


def test_an_expired_shared_token_is_not_reused(tmp_path: Path):
    poster = ScriptedPoster([token_response("first", expires_in=60), token_response("second", expires_in=60)])
    store = FileTokenStore(tmp_path)
    wall = FakeClock(1_700_000_000.0)
    minted(store_engine(poster, store, wall_clock=wall).get_token(make_spec()))

    wall.advance(61)
    later = minted(store_engine(poster, store, wall_clock=wall).get_token(make_spec()))

    assert later.access_token.get_secret_value() == "second"
    assert len(poster.requests) == 2


def test_the_remaining_lifetime_survives_different_monotonic_origins(tmp_path: Path):
    poster = ScriptedPoster([token_response(expires_in=3600)])
    store = FileTokenStore(tmp_path)
    wall = FakeClock(1_700_000_000.0)
    minted(store_engine(poster, store, clock=FakeClock(1_000.0), wall_clock=wall).get_token(make_spec()))

    wall.advance(600)
    later_clock = FakeClock(50_000.0)
    later = minted(store_engine(poster, store, clock=later_clock, wall_clock=wall).get_token(make_spec()))

    assert later.expires_at == pytest.approx(50_000.0 + 3000.0)
    assert len(poster.requests) == 1


def test_mandatory_refresh_serves_the_shared_token_until_it_expires_then_fails_once(tmp_path: Path):
    """With an unrotated assertion there is nothing new to exchange: refreshes inside the mandatory
    window keep serving the shared token, and once it has expired the one allowed POST is denied
    without a second identical POST behind it."""
    poster = SingleUsePoster(expires_in=3600)
    store = FileTokenStore(tmp_path)
    clock = FakeClock(1_000.0)
    engine = store_engine(poster, store, clock=clock, wall_clock=clock)
    first = minted(engine.get_token(make_spec()))

    clock.advance(3600 - 20)
    refreshed = minted(engine.get_token(make_spec()))
    assert refreshed.access_token.get_secret_value() == first.access_token.get_secret_value()
    assert len(poster.requests) == 1

    clock.advance(25)
    failed = engine.get_token(make_spec())

    assert isinstance(failed, TokenEndpointError)
    assert failed.status_code == 401
    assert len(poster.requests) == 2


def test_a_corrupt_cache_entry_is_treated_as_absent(tmp_path: Path):
    poster = ScriptedPoster([token_response("first"), token_response("second")])
    store = FileTokenStore(tmp_path)
    minted(store_engine(poster, store).get_token(make_spec()))
    (entry,) = stored_files(tmp_path)
    entry.write_text("{not json")

    later = minted(store_engine(poster, store).get_token(make_spec()))

    assert later.access_token.get_secret_value() == "second"
    assert json.loads(entry.read_text())["access_token"] == "second"


def test_cache_entries_are_private_to_the_owner(tmp_path: Path):
    store = FileTokenStore(tmp_path / "cache")
    minted(store_engine(ScriptedPoster([token_response()]), store).get_token(make_spec()))

    (entry,) = stored_files(tmp_path / "cache")
    assert stat.S_IMODE((tmp_path / "cache").stat().st_mode) == 0o700
    assert stat.S_IMODE(entry.stat().st_mode) == 0o600


def test_a_group_readable_cache_directory_is_refused_and_the_engine_still_mints(tmp_path: Path):
    loose = tmp_path / "loose"
    loose.mkdir(mode=0o750)
    os.chmod(loose, 0o750)
    poster = ScriptedPoster([token_response("first"), token_response("second")])
    store = FileTokenStore(loose)

    minted(store_engine(poster, store).get_token(make_spec()))
    later = minted(store_engine(poster, store).get_token(make_spec()))

    assert later.access_token.get_secret_value() == "second"
    assert stored_files(loose) == []


def test_default_store_follows_the_cache_dir_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(CACHE_DIR_ENV, "")
    assert default_shared_token_store() is None

    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path / "configured"))
    configured = default_shared_token_store()
    assert isinstance(configured, FileTokenStore)
    assert configured.directory == tmp_path / "configured"

    monkeypatch.delenv(CACHE_DIR_ENV)
    default = default_shared_token_store()
    assert isinstance(default, FileTokenStore)
    assert default.directory.name == f"litellm-token-exchange-{os.getuid()}"


class GatedSingleUsePoster(SingleUsePoster):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def post(self, url: str, *, content: bytes, headers: Mapping[str, str], timeout: float) -> httpx.Response:
        self.entered.set()
        assert self.release.wait(timeout=10)
        return super().post(url, content=content, headers=headers, timeout=timeout)


def test_a_worker_arriving_mid_exchange_waits_for_the_leader_instead_of_posting(tmp_path: Path):
    poster = GatedSingleUsePoster()
    store = FileTokenStore(tmp_path)
    leader = store_engine(poster, store)
    follower = store_engine(poster, store)
    results: dict[str, object] = {}

    def lead() -> None:
        results["leader"] = leader.get_token(make_spec())

    def follow() -> None:
        results["follower"] = follower.get_token(make_spec())

    leader_thread = threading.Thread(target=lead, daemon=True)
    leader_thread.start()
    assert poster.entered.wait(timeout=10)
    follower_thread = threading.Thread(target=follow, daemon=True)
    follower_thread.start()
    follower_thread.join(timeout=0.5)
    assert follower_thread.is_alive()
    poster.release.set()
    leader_thread.join(timeout=10)
    follower_thread.join(timeout=10)

    assert not follower_thread.is_alive()
    assert (
        minted(results["follower"]).access_token.get_secret_value()
        == minted(results["leader"]).access_token.get_secret_value()
    )
    assert len(poster.requests) == 1


def test_invalidate_drops_the_shared_entry(tmp_path: Path):
    poster = ScriptedPoster([token_response("first"), token_response("second")])
    store = FileTokenStore(tmp_path)
    engine = store_engine(poster, store)
    spec: Final = make_spec()
    minted(engine.get_token(spec))

    engine.invalidate(spec)

    assert stored_files(tmp_path) == []
    assert minted(store_engine(poster, store).get_token(spec)).access_token.get_secret_value() == "second"
