import datetime
import json

"""Tests for PTU config on the model deployment (v1 model-settings design)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy.management_endpoints.model_management_endpoints import (
    _merged_ptu_model_info,
    _validate_ptu_model_info,
)
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo, updateDeployment


def test_model_info_accepts_valid_ptu_fields():
    info = ModelInfo(id="x", team_id="t", ptu_count=5, cost_per_ptu_per_hour=2.0)
    assert info.ptu_count == 5
    assert info.cost_per_ptu_per_hour == 2.0


def test_model_info_rejects_non_positive_count():
    with pytest.raises(ValueError):
        ModelInfo(id="x", team_id="t", ptu_count=0, cost_per_ptu_per_hour=2.0)


def test_model_info_rejects_negative_rate():
    with pytest.raises(ValueError):
        ModelInfo(id="x", team_id="t", ptu_count=5, cost_per_ptu_per_hour=-1.0)


def test_model_info_allows_partial_delta_for_patch():
    # A PATCH delta may carry only one field; bounds-only validation must not reject it.
    info = ModelInfo(id="x", ptu_count=5)
    assert info.ptu_count == 5
    assert info.cost_per_ptu_per_hour is None


def test_validate_helper_no_ptu_is_noop():
    _validate_ptu_model_info({"team_id": "t"})


def test_validate_helper_requires_both_fields():
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info({"team_id": "t", "ptu_count": 5})
    assert exc.value.status_code == 400
    assert "set together" in exc.value.detail


def test_validate_helper_requires_team_id():
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info(
            {"ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "ptu_effective_from": "2026-08-01T00:00:00Z"}
        )
    assert exc.value.status_code == 400
    assert "team_id" in exc.value.detail


def test_validate_helper_requires_an_effective_start():
    """Flat cost accrues from the start, so it cannot be inferred."""
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info({"team_id": "t", "ptu_count": 5, "cost_per_ptu_per_hour": 2.0})
    assert exc.value.status_code == 400
    assert "ptu_effective_from is required" in exc.value.detail


def test_validate_helper_passes_full_config():
    _validate_ptu_model_info(
        {"team_id": "t", "ptu_count": 5, "cost_per_ptu_per_hour": 2.0, "ptu_effective_from": "2026-08-01T00:00:00Z"}
    )


def test_model_info_rejects_effective_to_before_from():
    import datetime

    with pytest.raises(ValueError):
        ModelInfo(
            id="x",
            team_id="t",
            ptu_count=5,
            cost_per_ptu_per_hour=2.0,
            ptu_effective_from=datetime.datetime(2026, 7, 30, tzinfo=datetime.timezone.utc),
            ptu_effective_to=datetime.datetime(2026, 7, 29, tzinfo=datetime.timezone.utc),
        )


def test_model_info_accepts_valid_effective_window():
    import datetime

    info = ModelInfo(
        id="x",
        team_id="t",
        ptu_count=5,
        cost_per_ptu_per_hour=2.0,
        ptu_effective_from=datetime.datetime(2026, 7, 30, tzinfo=datetime.timezone.utc),
        ptu_effective_to=datetime.datetime(2026, 8, 30, tzinfo=datetime.timezone.utc),
    )
    assert info.ptu_effective_from is not None


def test_model_info_compares_mixed_naive_and_aware_timestamps():
    import datetime

    info = ModelInfo(
        id="x",
        team_id="t",
        ptu_count=5,
        cost_per_ptu_per_hour=2.0,
        ptu_effective_from=datetime.datetime(2026, 7, 30, 23, 0),
        ptu_effective_to=datetime.datetime(2026, 7, 31, 0, 0, tzinfo=datetime.timezone.utc),
    )
    assert info.ptu_effective_to is not None

    with pytest.raises(ValueError):
        ModelInfo(
            id="x",
            team_id="t",
            ptu_count=5,
            cost_per_ptu_per_hour=2.0,
            ptu_effective_from=datetime.datetime(2026, 7, 31, 2, 0),
            ptu_effective_to=datetime.datetime(2026, 7, 31, 0, 0, tzinfo=datetime.timezone.utc),
        )


def test_validate_helper_rejects_effective_to_before_from():
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info(
            {
                "team_id": "t",
                "ptu_count": 5,
                "cost_per_ptu_per_hour": 2.0,
                "ptu_effective_from": "2026-07-30T00:00:00Z",
                "ptu_effective_to": "2026-07-29T00:00:00Z",
            }
        )
    assert exc.value.status_code == 400
    assert "ptu_effective_to" in exc.value.detail


def test_validate_helper_accepts_valid_window_on_merged_info():
    _validate_ptu_model_info(
        {
            "team_id": "t",
            "ptu_count": 5,
            "cost_per_ptu_per_hour": 2.0,
            "ptu_effective_from": "2026-07-30T00:00:00Z",
            "ptu_effective_to": "2026-08-30T00:00:00Z",
        }
    )


def test_validate_helper_rejects_inverted_window_without_count_or_rate():
    """A patch that touches only one end of the window merges to a model_info with no count
    or rate. Returning early on that shape let an inverted window reach the row, and the next
    load then failed to parse it and dropped the deployment out of the router."""
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info(
            {
                "team_id": "t",
                "ptu_effective_from": "2026-08-02T00:00:00Z",
                "ptu_effective_to": "2026-08-01T00:00:00Z",
            }
        )
    assert exc.value.status_code == 400
    assert "ptu_effective_to" in exc.value.detail


def test_validate_helper_rejects_equal_window_bounds_without_count_or_rate():
    with pytest.raises(HTTPException) as exc:
        _validate_ptu_model_info(
            {
                "ptu_effective_from": "2026-08-01T00:00:00Z",
                "ptu_effective_to": "2026-08-01T00:00:00Z",
            }
        )
    assert exc.value.status_code == 400


def test_validate_helper_accepts_ordered_window_without_count_or_rate():
    """Window-only edits stay legal; only the ordering is enforced, and no team_id is
    demanded while the deployment carries no priced PTU config."""
    _validate_ptu_model_info(
        {
            "ptu_effective_from": "2026-08-01T00:00:00Z",
            "ptu_effective_to": "2026-08-02T00:00:00Z",
        }
    )


def test_validate_helper_accepts_a_single_open_ended_bound():
    _validate_ptu_model_info({"ptu_effective_from": "2026-08-01T00:00:00Z"})
    _validate_ptu_model_info({"ptu_effective_to": "2026-08-02T00:00:00Z"})


class TestPartialPtuEditsUseTheMergedView:
    """A PTU invariant holds over the deployment as it will exist, not over whichever
    subset of fields a caller sent. Validating the patch alone rejected an ordinary edit."""

    @staticmethod
    def _configured():
        return Deployment(
            model_name="gpt-4o",
            litellm_params=LiteLLM_Params(model="openai/gpt-4o"),
            model_info=ModelInfo(
                id="dep-0",
                team_id="t",
                ptu_count=10,
                cost_per_ptu_per_hour=2.0,
                ptu_effective_from=datetime.datetime(2026, 7, 1, tzinfo=datetime.timezone.utc),
            ),
        )

    def test_raising_the_rate_on_a_configured_model_is_allowed(self):
        """The patch carries no start; the stored row supplies it."""
        merged = _merged_ptu_model_info(
            db_model=self._configured(),
            patch_data=updateDeployment(model_info=ModelInfo(id="dep-0", ptu_count=10, cost_per_ptu_per_hour=3.0)),
        )
        _validate_ptu_model_info(merged)
        assert merged["cost_per_ptu_per_hour"] == 3.0
        assert merged["ptu_effective_from"] is not None

    def test_a_genuinely_startless_configuration_is_still_rejected(self):
        """Merging must not become a way to smuggle PTU config in without a start."""
        bare = Deployment(model_name="gpt-4o", litellm_params=LiteLLM_Params(model="openai/gpt-4o"))
        merged = _merged_ptu_model_info(
            db_model=bare,
            patch_data=updateDeployment(
                model_info=ModelInfo(id="dep-0", team_id="t", ptu_count=10, cost_per_ptu_per_hour=2.0)
            ),
        )
        with pytest.raises(HTTPException) as exc:
            _validate_ptu_model_info(merged)
        assert "ptu_effective_from is required" in exc.value.detail

    def test_the_patch_still_wins_over_the_stored_value(self):
        merged = _merged_ptu_model_info(
            db_model=self._configured(),
            patch_data=updateDeployment(model_info=ModelInfo(id="dep-0", ptu_count=25)),
        )
        assert merged["ptu_count"] == 25

    def test_an_explicit_null_clears_the_stored_field(self):
        """update_db_model drops a PTU field a patch sends as null, so the merged view has to
        drop it too. Carrying the stored value forward validated a deployment that never
        existed."""
        merged = _merged_ptu_model_info(
            db_model=self._configured(),
            patch_data=updateDeployment(model_info=ModelInfo(id="dep-0", ptu_count=None)),
        )
        assert "ptu_count" not in merged

    def test_clearing_one_half_of_the_pair_is_rejected(self):
        """The write leaves a rate with no count. Merging on the stored count hid that."""
        merged = _merged_ptu_model_info(
            db_model=self._configured(),
            patch_data=updateDeployment(model_info=ModelInfo(id="dep-0", ptu_count=None)),
        )
        with pytest.raises(HTTPException) as exc:
            _validate_ptu_model_info(merged)
        assert "must be set together" in exc.value.detail

    def test_clearing_the_whole_pair_is_allowed(self):
        """Turning PTU off on a deployment is a legitimate edit."""
        merged = _merged_ptu_model_info(
            db_model=self._configured(),
            patch_data=updateDeployment(model_info=ModelInfo(id="dep-0", ptu_count=None, cost_per_ptu_per_hour=None)),
        )
        _validate_ptu_model_info(merged)
        assert "ptu_count" not in merged
        assert "cost_per_ptu_per_hour" not in merged

    def test_an_omitted_field_is_not_a_clear(self):
        """A partial edit that never mentions the count keeps it. Only an explicit null clears."""
        merged = _merged_ptu_model_info(
            db_model=self._configured(),
            patch_data=updateDeployment(model_info=ModelInfo(id="dep-0", cost_per_ptu_per_hour=3.0)),
        )
        assert merged["ptu_count"] == 10


class TestTeamModelUpdateValidatesBeforeWriting:
    """Drives the endpoint path itself, not the helpers. The validator sits above the team
    ACL write, which autocommits, so what it validates has to be right at that call site."""

    @staticmethod
    async def _run(db_model, patch_data, monkeypatch, touched=None):
        import litellm.proxy.management_endpoints.model_management_endpoints as mme

        touched = [] if touched is None else touched

        async def _never(*args, **kwargs):
            touched.append("team_write")

        monkeypatch.setattr(mme, "_setup_new_team_model_assignment", _never)
        monkeypatch.setattr(mme, "_update_existing_team_model_assignment", _never)
        monkeypatch.setattr(mme.ModelManagementAuthChecks, "allow_team_model_action", AsyncMock(return_value=True))
        result = await mme._update_team_model_in_db(
            db_model=db_model,
            patch_data=patch_data,
            user_api_key_dict=MagicMock(),
            prisma_client=MagicMock(),
        )
        return result, touched

    @pytest.mark.asyncio
    async def test_raising_the_rate_on_a_configured_model_reaches_the_write(self, monkeypatch):
        """The patch carries no start. Validating it alone rejected this ordinary edit."""
        db_model = TestPartialPtuEditsUseTheMergedView._configured()
        patch = updateDeployment(model_info=ModelInfo(id="dep-0", team_id="t", ptu_count=10, cost_per_ptu_per_hour=3.0))

        result, touched = await self._run(db_model, patch, monkeypatch)

        assert touched == ["team_write"]
        assert json.loads(result["model_info"])["cost_per_ptu_per_hour"] == 3.0

    @pytest.mark.asyncio
    async def test_a_startless_configuration_is_refused_before_the_team_write(self, monkeypatch):
        """And the refusal still lands before anything is committed."""
        bare = Deployment(model_name="gpt-4o", litellm_params=LiteLLM_Params(model="openai/gpt-4o"))
        patch = updateDeployment(model_info=ModelInfo(id="dep-0", team_id="t", ptu_count=10, cost_per_ptu_per_hour=2.0))

        with pytest.raises(HTTPException) as exc:
            await self._run(bare, patch, monkeypatch)

        assert "ptu_effective_from is required" in exc.value.detail

    @pytest.mark.asyncio
    async def test_clearing_half_the_pair_is_refused_before_the_team_write(self, monkeypatch):
        """The write drops the nulled field, so validating against the stored one let a
        deployment with a rate and no count commit."""
        db_model = TestPartialPtuEditsUseTheMergedView._configured()
        patch = updateDeployment(model_info=ModelInfo(id="dep-0", team_id="t", ptu_count=None))
        touched = []

        with pytest.raises(HTTPException) as exc:
            await self._run(db_model, patch, monkeypatch, touched)

        assert "must be set together" in exc.value.detail
        assert touched == []

    @pytest.mark.asyncio
    async def test_clearing_the_whole_pair_reaches_the_write_and_stores_neither_field(self, monkeypatch):
        """What the validator approved is what the write persists."""
        db_model = TestPartialPtuEditsUseTheMergedView._configured()
        patch = updateDeployment(
            model_info=ModelInfo(id="dep-0", team_id="t", ptu_count=None, cost_per_ptu_per_hour=None)
        )

        result, touched = await self._run(db_model, patch, monkeypatch)

        assert touched == ["team_write"]
        stored = json.loads(result["model_info"])
        assert "ptu_count" not in stored
        assert "cost_per_ptu_per_hour" not in stored
