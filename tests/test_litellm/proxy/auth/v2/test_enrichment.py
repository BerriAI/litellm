from types import SimpleNamespace

import pytest

from litellm.proxy.auth.v2.enrichment import enrich_identity


def _identity(**overrides):
    base = dict(
        user_id="u1",
        team_id="t1",
        user_max_budget=None,
        user_tpm_limit=None,
        user_rpm_limit=None,
        user_spend=None,
        team_max_budget=None,
        team_tpm_limit=None,
        team_rpm_limit=None,
        team_spend=None,
        team_models=None,
        team_blocked=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _loader(obj):
    async def load(_id):
        return obj

    return load


@pytest.mark.asyncio
async def test_user_limits_are_copied_from_the_user_row():
    identity = _identity()
    user = SimpleNamespace(max_budget=12.5, tpm_limit=100, rpm_limit=10, spend=3.0)
    await enrich_identity(identity, load_user=_loader(user), load_team=_loader(None))
    assert identity.user_max_budget == 12.5
    assert identity.user_tpm_limit == 100
    assert identity.user_rpm_limit == 10
    assert identity.user_spend == 3.0


@pytest.mark.asyncio
async def test_team_limits_and_models_are_copied_from_the_team_row():
    identity = _identity()
    team = SimpleNamespace(
        max_budget=50.0,
        tpm_limit=1000,
        rpm_limit=100,
        spend=9.0,
        models=["gpt-4o"],
        blocked=True,
    )
    await enrich_identity(identity, load_user=_loader(None), load_team=_loader(team))
    assert identity.team_max_budget == 50.0
    assert identity.team_rpm_limit == 100
    assert identity.team_models == ["gpt-4o"]
    assert identity.team_blocked is True


@pytest.mark.asyncio
async def test_already_set_fields_are_not_overwritten():
    # A value resolved earlier (e.g. a key's own user limit) must win over the row.
    identity = _identity(user_rpm_limit=7)
    user = SimpleNamespace(max_budget=None, tpm_limit=None, rpm_limit=999, spend=None)
    await enrich_identity(identity, load_user=_loader(user), load_team=_loader(None))
    assert identity.user_rpm_limit == 7


@pytest.mark.asyncio
async def test_missing_ids_and_loaders_are_a_noop():
    identity = _identity(user_id=None, team_id=None)
    await enrich_identity(identity)  # no loaders, no ids
    assert identity.user_max_budget is None
    assert identity.team_max_budget is None


@pytest.mark.asyncio
async def test_loader_returning_none_leaves_identity_untouched():
    identity = _identity()
    await enrich_identity(identity, load_user=_loader(None), load_team=_loader(None))
    assert identity.user_max_budget is None
    assert identity.team_rpm_limit is None
