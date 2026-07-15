import argparse
import importlib.util
import os
from datetime import date
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "ptu_reservation_backfill.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("_ptu_backfill", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dates_from_args_single_date():
    mod = _load_script()
    args = argparse.Namespace(date=date(2026, 7, 12), date_range=None)
    assert mod._dates_from_args(args) == [date(2026, 7, 12)]


def test_dates_from_args_inclusive_range():
    mod = _load_script()
    args = argparse.Namespace(date=None, date_range="2026-07-10:2026-07-12")
    assert mod._dates_from_args(args) == [
        date(2026, 7, 10),
        date(2026, 7, 11),
        date(2026, 7, 12),
    ]


def test_dates_from_args_single_day_range():
    mod = _load_script()
    args = argparse.Namespace(date=None, date_range="2026-07-12:2026-07-12")
    assert mod._dates_from_args(args) == [date(2026, 7, 12)]


def test_dates_from_args_rejects_reversed_range():
    mod = _load_script()
    args = argparse.Namespace(date=None, date_range="2026-07-15:2026-07-01")
    with pytest.raises(ValueError):
        mod._dates_from_args(args)


@pytest.mark.asyncio
async def test_run_exits_when_database_url_missing(monkeypatch, capsys):
    mod = _load_script()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    code = await mod._run([date(2026, 7, 12)])
    assert code == 2
    err = capsys.readouterr().err
    assert "DATABASE_URL" in err


@pytest.mark.asyncio
async def test_run_calls_rollup_with_force_true(monkeypatch, capsys):
    mod = _load_script()

    class _FakePrismaClient:
        def __init__(self, *_args, **_kwargs):
            self.connected = False
            self.disconnected = False
            self.db = self

        async def connect(self):
            self.connected = True

        async def disconnect(self):
            self.disconnected = True

    fake_client_holder = {}

    def _fake_ctor(*args, **kwargs):
        client = _FakePrismaClient(*args, **kwargs)
        fake_client_holder["client"] = client
        return client

    from litellm.proxy import utils as proxy_utils
    monkeypatch.setattr(proxy_utils, "PrismaClient", _fake_ctor)

    from litellm.proxy.spend_tracking import ptu_reservation_rollup as rollup_mod
    from litellm.proxy.spend_tracking.ptu_reservation_rollup import RollupResult

    calls: list[dict] = []

    async def _fake_rollup(prisma, *, target_date, force=False):
        calls.append({"target_date": target_date, "force": force})
        return RollupResult(day=target_date, reservations_processed=1, rows_written=1)

    monkeypatch.setattr(rollup_mod, "run_ptu_reservation_rollup", _fake_rollup)

    monkeypatch.setenv("DATABASE_URL", "postgresql://fake@localhost/test")

    code = await mod._run([date(2026, 7, 10), date(2026, 7, 11)])
    assert code == 0
    assert [c["target_date"] for c in calls] == [date(2026, 7, 10), date(2026, 7, 11)]
    assert all(c["force"] is True for c in calls)
    out = capsys.readouterr().out
    assert "total rows written: 2" in out
    assert fake_client_holder["client"].connected is True
    assert fake_client_holder["client"].disconnected is True
