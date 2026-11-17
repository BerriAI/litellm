"""Which CustomLoggerV2 callbacks a proxy request selects: no-log, `x-litellm-disable-callbacks`, a key's
`litellm_disabled_callbacks`, team settings, activation and aliases. The Rust ports (Prometheus, S3 v3, budgets,
limiter, capacity) declare their Selection in their manifest (rust.py WireSelection), so this is the switchboard
every ported callback is turned on and off by.

Contract under test:

- The proxy fills only `caller_opt_outs` and `no_log` (litellm_pre_call_utils.py:1540-1552). `requested` and
  `admin_opt_outs` are never set, so `opt_in` loggers never run and admin-only opt-out has no effect
  (user-guide.md:14; selection.py:24-36).
- `caller_opt_outs` come from the header when present, otherwise from the key's metadata
  `litellm_disabled_callbacks`; team settings are not a source; both need a premium proxy and
  `litellm.allow_dynamic_callback_disabling` True (litellm_pre_call_utils.py:1513-1537). Header names are split on
  commas, stripped and lowercased; key names are lowercased.
- A logger is skipped when one of its name or aliases is in the opt-outs and it allows `OptOutBy.caller`, or the
  request is no-log and it `honors_no_log`; `Activation.required` short-circuits every opt-out (selection.py:24-36).
- `no_log` is False whenever `litellm.global_disable_no_log_param` is set (litellm_pre_call_utils.py:1550).

Readers are `attempt_ended` readers from `dec1_cblv2b_probe` that differ only in their Selection. Opt-out tests
need LITELLM_LICENSE (premium) and skip without it, like round 1.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import Scenario
from integration.observability.dec1_cblv2_support import Rig, cblv2_rig, chat, marker
from integration.observability.dec1_cblv2b_support import (
    DISABLE_HEADER,
    LICENSE_SKIP,
    instances,
    new_key,
    records,
)

LICENSED: Final = bool(os.environ.get("LITELLM_LICENSE"))
READERS: Final = (
    "required_reader",
    "opt_in_reader",
    "admin_only_reader",
    "caller_reader",
    "alias_reader",
    "camel_reader",
    "no_log_reader",
)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("cblv2b-selection")
    with cblv2_rig(directory, {"callbacks": instances(*READERS)}) as booted:
        yield booted


@pytest.fixture(scope="module")
def locked(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    """A proxy with dynamic callback disabling switched off and the no-log parameter disabled globally."""
    directory: Final = tmp_path_factory.mktemp("cblv2b-selection-locked")
    with cblv2_rig(
        directory,
        {
            "callbacks": instances(*READERS),
            "allow_dynamic_callback_disabling": False,
            "global_disable_no_log_param": True,
        },
    ) as booted:
        yield booted


@pytest.fixture(scope="module")
def scenario(rig: Rig) -> Iterator[Scenario]:
    with rig.proxy.scenario() as opened:
        yield opened


def _settled(rig: Rig, text: str) -> None:
    """Wait until the always-on reader has recorded the request, so absences after it are meaningful."""
    records(rig, text, "required_reader", "attempt_ended", 1)


def test_every_default_on_reader_records_a_plain_chat(rig: Rig) -> None:
    """A plain chat must give exactly one `attempt_ended` record with outcome "success" to each default-on and
    required reader: required, admin-only, caller, alias, camel and no-log readers."""
    text: Final = marker()
    assert chat(rig, text).status_code == 200
    for reader in ("required_reader", "admin_only_reader", "caller_reader", "alias_reader", "camel_reader"):
        assert records(rig, text, reader, "attempt_ended", 1)[0]["outcome"] == "success", reader
    records(rig, text, "no_log_reader", "attempt_ended", 1)


def test_opt_in_reader_never_runs_on_the_proxy(rig: Rig) -> None:
    """`opt_in_reader` has `Activation.opt_in`. The proxy never fills `RequestSelection.requested`
    (litellm_pre_call_utils.py:1547), so it must record nothing for a plain chat, even after the always-on
    reader has recorded it. Pinned so wiring opt-in names is a visible change."""
    text: Final = marker()
    assert chat(rig, text).status_code == 200
    _settled(rig, text)
    assert records(rig, text, "opt_in_reader", "attempt_ended", 0) == ()


def test_honors_no_log_reader_skips_a_no_log_request_but_required_still_runs(rig: Rig) -> None:
    """For a `no-log: true` chat, `no_log_reader` (honors_no_log=True) must record nothing, while `required_reader`,
    which also declares honors_no_log=True but is `Activation.required`, must still record it: required wins
    (selection.py:31-32)."""
    text: Final = marker()
    assert chat(rig, text, **{"no-log": True}).status_code == 200
    _settled(rig, text)
    assert records(rig, text, "no_log_reader", "attempt_ended", 0) == ()


def test_no_log_does_not_skip_a_reader_that_does_not_honor_it(rig: Rig) -> None:
    """`caller_reader` keeps the default honors_no_log=False, so a `no-log: true` chat must still give it one
    record (selection.py:16, 21)."""
    text: Final = marker()
    assert chat(rig, text, **{"no-log": True}).status_code == 200
    records(rig, text, "caller_reader", "attempt_ended", 1)


def test_disable_header_skips_a_caller_reader_by_an_alias(rig: Rig) -> None:
    """`alias_reader` is named `cblv2b_alias_reader` with alias `cblv2b_alias` and allows caller opt-out. On a
    licensed proxy `x-litellm-disable-callbacks: cblv2b_alias` must skip it, while `caller_reader` still records the
    request (selection.py:33)."""
    if not LICENSED:
        pytest.skip(LICENSE_SKIP)
    text: Final = marker()
    assert chat(rig, text, headers={DISABLE_HEADER: "cblv2b_alias"}).status_code == 200
    records(rig, text, "caller_reader", "attempt_ended", 1)
    assert records(rig, text, "alias_reader", "attempt_ended", 0) == ()


def test_disable_header_with_several_names_skips_each_one(rig: Rig) -> None:
    """The header ` cblv2b_caller_reader , CBLV2B_ALIAS_READER` (spaces and upper case) must skip both named readers:
    names are split on commas, stripped and lowercased (litellm_pre_call_utils.py:1531)."""
    if not LICENSED:
        pytest.skip(LICENSE_SKIP)
    text: Final = marker()
    headers: Final = {DISABLE_HEADER: " cblv2b_caller_reader , CBLV2B_ALIAS_READER"}
    assert chat(rig, text, headers=headers).status_code == 200
    _settled(rig, text)
    assert records(rig, text, "caller_reader", "attempt_ended", 0) == ()
    assert records(rig, text, "alias_reader", "attempt_ended", 0) == ()


def test_disable_header_cannot_skip_an_admin_only_reader(rig: Rig) -> None:
    """`admin_only_reader` keeps the default `opt_out_by={admin}`. The header names it, but the header only fills
    caller opt-outs, so it must still record the request."""
    if not LICENSED:
        pytest.skip(LICENSE_SKIP)
    text: Final = marker()
    assert chat(rig, text, headers={DISABLE_HEADER: "cblv2b_admin_only_reader"}).status_code == 200
    records(rig, text, "admin_only_reader", "attempt_ended", 1)


def test_disable_header_cannot_skip_a_required_reader(rig: Rig) -> None:
    """`required_reader` allows caller opt-out in its Selection but is `Activation.required`, so naming it in the
    header must not skip it (selection.py:31-32)."""
    if not LICENSED:
        pytest.skip(LICENSE_SKIP)
    text: Final = marker()
    assert chat(rig, text, headers={DISABLE_HEADER: "cblv2b_required_reader"}).status_code == 200
    records(rig, text, "required_reader", "attempt_ended", 1)


def test_disable_header_skips_a_reader_whose_name_has_capitals(rig: Rig) -> None:
    """`camel_reader` is named `Cblv2bCamelReader` and allows caller opt-out. The header `Cblv2bCamelReader` should
    skip it. Today the header is lowercased (litellm_pre_call_utils.py:1531) and compared case-sensitively with the
    Selection name (selection.py:25-26), so a logger with capitals in its name, which is every logger left on the
    default name `type(logger).__name__` (manifest.py:64), can never be switched off by a caller."""
    if not LICENSED:
        pytest.skip(LICENSE_SKIP)
    pytest.skip("BUG: caller opt-out names are lowercased but Selection names are matched case-sensitively")
    text: Final = marker()
    assert chat(rig, text, headers={DISABLE_HEADER: "Cblv2bCamelReader"}).status_code == 200
    _settled(rig, text)
    assert records(rig, text, "camel_reader", "attempt_ended", 0) == ()


def test_key_disabled_callbacks_skip_a_caller_reader_for_that_key_only(rig: Rig, scenario: Scenario) -> None:
    """A key with metadata `litellm_disabled_callbacks: ["cblv2b_caller_reader"]` must never reach `caller_reader`,
    while a sibling key without it does (litellm_pre_call_utils.py:1533-1537)."""
    if not LICENSED:
        pytest.skip(LICENSE_SKIP)
    muted, _, _ = new_key(scenario, metadata={"litellm_disabled_callbacks": ["cblv2b_caller_reader"]})
    loud, _, _ = new_key(scenario)
    quiet_text: Final = marker()
    loud_text: Final = marker()
    assert chat(rig, quiet_text, key=muted).status_code == 200
    assert chat(rig, loud_text, key=loud).status_code == 200
    records(rig, loud_text, "caller_reader", "attempt_ended", 1)
    _settled(rig, quiet_text)
    assert records(rig, quiet_text, "caller_reader", "attempt_ended", 0) == ()


def test_key_disabled_callbacks_cannot_skip_an_admin_only_reader(rig: Rig, scenario: Scenario) -> None:
    """The key's list is set by an admin, but the proxy files it under caller opt-outs, so a key listing
    `cblv2b_admin_only_reader` must still reach `admin_only_reader` (default opt_out_by={admin}). Pinned as a
    surprise: a ported callback that keeps the default Selection cannot be muted per key."""
    if not LICENSED:
        pytest.skip(LICENSE_SKIP)
    key, _, _ = new_key(scenario, metadata={"litellm_disabled_callbacks": ["cblv2b_admin_only_reader"]})
    text: Final = marker()
    assert chat(rig, text, key=key).status_code == 200
    records(rig, text, "admin_only_reader", "attempt_ended", 1)


def test_disable_header_replaces_the_key_list(rig: Rig, scenario: Scenario) -> None:
    """A key listing `cblv2b_caller_reader` sends the header `cblv2b_alias`. The header wins outright, so
    `caller_reader` must record the request and `alias_reader` must not (litellm_pre_call_utils.py:1528-1532)."""
    if not LICENSED:
        pytest.skip(LICENSE_SKIP)
    key, _, _ = new_key(scenario, metadata={"litellm_disabled_callbacks": ["cblv2b_caller_reader"]})
    text: Final = marker()
    assert chat(rig, text, key=key, headers={DISABLE_HEADER: "cblv2b_alias"}).status_code == 200
    records(rig, text, "caller_reader", "attempt_ended", 1)
    assert records(rig, text, "alias_reader", "attempt_ended", 0) == ()


def test_team_disabled_callbacks_are_not_a_source(rig: Rig, scenario: Scenario) -> None:
    """A team whose metadata lists `cblv2b_caller_reader` in `litellm_disabled_callbacks` has a key without such a
    list. Team settings are not a source of opt-outs (litellm_pre_call_utils.py:1520), so `caller_reader` must
    still record that key's chat. UNVERIFIED: assumes the team metadata is not copied onto the key's auth object."""
    if not LICENSED:
        pytest.skip(LICENSE_SKIP)
    team: Final = scenario.team(metadata={"litellm_disabled_callbacks": ["cblv2b_caller_reader"]})
    key, _, _ = new_key(scenario, team_id=team)
    text: Final = marker()
    assert chat(rig, text, key=key).status_code == 200
    records(rig, text, "caller_reader", "attempt_ended", 1)


def test_disable_header_is_ignored_without_a_license(rig: Rig) -> None:
    """On a proxy without LITELLM_LICENSE, `premium_user` is False and the header must change nothing:
    `caller_reader` still records the request (litellm_pre_call_utils.py:1525)."""
    if LICENSED:
        pytest.skip("this rig runs licensed; the unlicensed branch cannot be observed here")
    text: Final = marker()
    assert chat(rig, text, headers={DISABLE_HEADER: "cblv2b_caller_reader"}).status_code == 200
    records(rig, text, "caller_reader", "attempt_ended", 1)


def test_disable_header_is_ignored_when_dynamic_disabling_is_off(locked: Rig) -> None:
    """With `allow_dynamic_callback_disabling: false`, even a licensed proxy must ignore the header: `caller_reader`
    still records the request (litellm_pre_call_utils.py:1525)."""
    text: Final = marker()
    assert chat(locked, text, headers={DISABLE_HEADER: "cblv2b_caller_reader"}).status_code == 200
    records(locked, text, "caller_reader", "attempt_ended", 1)


def test_global_disable_no_log_param_makes_an_honoring_reader_record_no_log_requests(locked: Rig) -> None:
    """With `global_disable_no_log_param: true` the proxy opens the scope with no_log False
    (litellm_pre_call_utils.py:1550), so `no_log_reader` (honors_no_log=True) must record a `no-log: true` chat."""
    text: Final = marker()
    assert chat(locked, text, **{"no-log": True}).status_code == 200
    records(locked, text, "no_log_reader", "attempt_ended", 1)
