"""Harness coverage for the barriers that gate on every replica.

No proxy needed and no ``e2e`` marker: this pins that a model registered through
the control plane only counts as servable once every configured replica lists it
on /v1/models, and that a management write only counts as read back once every
replica's read satisfies the caller's predicate, which is what keeps a two-gateway
stack from handing a test a model or a key that one gateway has not caught up on
yet. The fakes are plain pollers standing in for each replica's transport plus an
injected clock, so nothing here monkeypatches anything.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import chain, repeat
from types import MappingProxyType
from typing import Final

import pytest

from e2e_config import parse_replica_urls
from e2e_http import Result, Success
from models import KeyInfo, KeyInfoResponse, ModelListEntry, ModelsListResponse
from proxy_client import (
    Poller,
    ConvergeOutcome,
    Converged,
    ModelsPoller,
    NotConverged,
    NotServableOn,
    Servable,
    await_converged_everywhere,
    await_servable_everywhere,
    first_lagging_replica,
    converge_timeout_message,
)

MODEL: Final = "gpt-under-test"
TIMEOUT: Final = 10.0
INTERVAL: Final = 2.0
RPM_BEFORE_UPDATE: Final = 100
RPM_AFTER_UPDATE: Final = 200


@dataclass
class FakeClock:
    elapsed: float = 0.0

    def now(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


def _listing(*model_ids: str) -> Success[ModelsListResponse]:
    entries: Final = tuple(ModelListEntry(id=model_id) for model_id in model_ids)
    return Success(status_code=200, data=ModelsListResponse(data=entries))


def _poller(results: Iterable[Success[ModelsListResponse]]) -> ModelsPoller:
    it: Final = iter(results)
    return lambda _timeout: next(it)


def _await(pollers: Mapping[str, ModelsPoller]) -> Servable | NotServableOn:
    clock: Final = FakeClock()
    return await_servable_everywhere(
        pollers,
        model_name=MODEL,
        timeout=TIMEOUT,
        interval=INTERVAL,
        request_timeout=5.0,
        db_sync_seconds=0.0,
        now=clock.now,
        sleep=clock.sleep,
    )


class TestAwaitServableEverywhere:
    @pytest.mark.parametrize("missing", ["gateway-1", "gateway-2"])
    def test_fails_on_the_replica_that_never_lists_the_model(self, missing: str) -> None:
        pollers: Final = {
            "gateway-1": _poller(repeat(_listing(MODEL))),
            "gateway-2": _poller(repeat(_listing(MODEL))),
        } | {missing: _poller(repeat(_listing()))}
        assert _await(pollers) == NotServableOn(replica=missing, last_result=_listing())

    def test_passes_once_every_replica_lists_the_model(self) -> None:
        pollers: Final = {
            "gateway-1": _poller(repeat(_listing(MODEL))),
            "gateway-2": _poller(chain(repeat(_listing(), 2), repeat(_listing(MODEL)))),
        }
        assert _await(pollers) == Servable()


def _key_info(rpm_limit: int) -> Success[KeyInfoResponse]:
    return Success(status_code=200, data=KeyInfoResponse(info=KeyInfo(rpm_limit=rpm_limit)))


def _reads(results: Iterable[Result[KeyInfoResponse]]) -> Poller[Result[KeyInfoResponse]]:
    it: Final = iter(results)
    return lambda: next(it)


def _updated(result: Result[KeyInfoResponse]) -> bool:
    return isinstance(result, Success) and result.data.info.rpm_limit == RPM_AFTER_UPDATE


def _converge(
    pollers: Mapping[str, Poller[Result[KeyInfoResponse]]], clock: FakeClock
) -> Mapping[str, ConvergeOutcome[Result[KeyInfoResponse]]]:
    return await_converged_everywhere(
        pollers,
        converged=_updated,
        timeout=TIMEOUT,
        interval=INTERVAL,
        now=clock.now,
        sleep=clock.sleep,
    )


class TestAwaitConvergedEverywhere:
    def test_waits_for_the_replica_that_lags_behind_the_write(self) -> None:
        clock: Final = FakeClock()
        pollers: Final = MappingProxyType(
            {
                "gateway-1": _reads(repeat(_key_info(RPM_AFTER_UPDATE))),
                "gateway-2": _reads(
                    chain(repeat(_key_info(RPM_BEFORE_UPDATE), 2), repeat(_key_info(RPM_AFTER_UPDATE)))
                ),
            }
        )
        outcomes: Final = _converge(pollers, clock)
        assert outcomes == {
            "gateway-1": Converged(result=_key_info(RPM_AFTER_UPDATE)),
            "gateway-2": Converged(result=_key_info(RPM_AFTER_UPDATE)),
        }
        assert first_lagging_replica(outcomes) is None
        assert clock.elapsed == 2 * INTERVAL

    def test_names_the_replica_that_never_converges_with_its_last_read(self) -> None:
        clock: Final = FakeClock()
        pollers: Final = MappingProxyType(
            {
                "gateway-1": _reads(repeat(_key_info(RPM_AFTER_UPDATE))),
                "gateway-2": _reads(repeat(_key_info(RPM_BEFORE_UPDATE))),
            }
        )
        outcomes: Final = _converge(pollers, clock)
        assert first_lagging_replica(outcomes) == (
            "gateway-2",
            NotConverged(last_result=_key_info(RPM_BEFORE_UPDATE)),
        )
        assert clock.elapsed == TIMEOUT
        message: Final = converge_timeout_message(
            what="GET /key/info",
            replica="gateway-2",
            timeout=TIMEOUT,
            last_result=_key_info(RPM_BEFORE_UPDATE),
        )
        assert "gateway-2" in message and "/key/info" in message and str(RPM_BEFORE_UPDATE) in message

    def test_each_replica_gets_its_own_full_budget(self) -> None:
        """A replica that converges late must not eat into the next replica's budget: both
        need most of the timeout here, so one shared deadline would starve the second."""
        clock: Final = FakeClock()
        slow: Final = chain(repeat(_key_info(RPM_BEFORE_UPDATE), 3), repeat(_key_info(RPM_AFTER_UPDATE)))
        pollers: Final = MappingProxyType(
            {
                "gateway-1": _reads(slow),
                "gateway-2": _reads(
                    chain(repeat(_key_info(RPM_BEFORE_UPDATE), 3), repeat(_key_info(RPM_AFTER_UPDATE)))
                ),
            }
        )
        outcomes: Final = _converge(pollers, clock)
        assert first_lagging_replica(outcomes) is None
        assert clock.elapsed == 2 * 3 * INTERVAL


class TestParseReplicaUrls:
    def test_splits_and_trims_the_gateway_addresses(self) -> None:
        raw: Final = " http://127.0.0.1:4010/, http://127.0.0.1:4011 "
        assert parse_replica_urls(raw, "http://lb") == ("http://127.0.0.1:4010", "http://127.0.0.1:4011")

    def test_falls_back_to_the_data_plane_address_when_unset(self) -> None:
        assert parse_replica_urls("", "http://lb") == ("http://lb",)
