"""Tests for PTU config on the model deployment (v1 model-settings design)."""

import datetime
import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy._types import (
    LiteLLM_ProxyModelTable,
    LitellmUserRoles,
    ReconcileOutcome,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints.model_management_endpoints import (
    _merged_ptu_model_info,
    _raise_if_ptu_cost_attribution_disabled,
    _validate_ptu_model_info,
    add_new_model,
    update_db_model,
)
from litellm.proxy.spend_tracking.ptu_feature_flag import PTU_COST_ATTRIBUTION_ENV_VAR
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo, updateDeployment


def test_model_info_accepts_valid_ptu_fields():
    info = ModelInfo(
        id="x",
        team_id="t",
        ptu_count=5,
        cost_per_ptu_per_hour=2.0,
        ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
    )
    assert info.ptu_count == 5
    assert info.cost_per_ptu_per_hour == 2.0


def test_model_info_rejects_non_positive_count():
    with pytest.raises(ValueError):
        ModelInfo(
            id="x",
            team_id="t",
            ptu_count=0,
            cost_per_ptu_per_hour=2.0,
            ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
        )


def test_model_info_rejects_negative_rate():
    with pytest.raises(ValueError):
        ModelInfo(
            id="x",
            team_id="t",
            ptu_count=5,
            cost_per_ptu_per_hour=-1.0,
            ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
        )


def test_model_info_rejects_a_count_beyond_the_cap():
    """flat cost multiplies the count by a float, and an unbounded int overflows that
    conversion, which aborted the rollup for every team rather than skipping one model."""
    with pytest.raises(ValueError):
        ModelInfo(id="x", team_id="t", ptu_count=10**400, cost_per_ptu_per_hour=2.0)


def test_model_info_accepts_a_count_at_the_cap():
    info = ModelInfo(id="x", team_id="t", ptu_count=ModelInfo.MAX_PTU_COUNT, cost_per_ptu_per_hour=2.0)
    assert info.ptu_count == ModelInfo.MAX_PTU_COUNT


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), float("-inf")])
def test_model_info_rejects_a_non_finite_rate(rate):
    """NaN compares False against every bound, so a bare `< 0` check let it through and the
    deployment then accrued a flat cost of nan."""
    with pytest.raises(ValueError):
        ModelInfo(id="x", team_id="t", ptu_count=5, cost_per_ptu_per_hour=rate)


def test_model_info_rejects_a_rate_beyond_the_cap():
    with pytest.raises(ValueError):
        ModelInfo(id="x", team_id="t", ptu_count=5, cost_per_ptu_per_hour=ModelInfo.MAX_COST_PER_PTU_PER_HOUR * 2)


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

    @pytest.fixture(autouse=True)
    def _enabled(self, monkeypatch):
        """PTU writes are gated off by default; these are about the validator, not the gate."""
        monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")

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

    @pytest.fixture(autouse=True)
    def _enabled(self, monkeypatch):
        """PTU writes are gated off by default; these are about the validator, not the gate."""
        monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")

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
    async def test_the_gate_refuses_before_the_team_write(self, monkeypatch):
        """The gate lived inside update_db_model, which runs after the team ACL write, so a
        rejected edit still moved the model between teams."""
        monkeypatch.delenv(PTU_COST_ATTRIBUTION_ENV_VAR, raising=False)
        db_model = Deployment(
            model_name="gpt-4o",
            litellm_params=LiteLLM_Params(model="openai/gpt-4o"),
            model_info=ModelInfo(id="dep-0", team_id="team-A"),
        )
        patch = updateDeployment(
            model_info=ModelInfo(
                id="dep-0",
                team_id="team-B",
                ptu_count=15,
                cost_per_ptu_per_hour=2.0,
                ptu_effective_from=datetime.datetime(2026, 8, 1, tzinfo=datetime.timezone.utc),
            )
        )
        touched = []

        with pytest.raises(HTTPException) as exc:
            await self._run(db_model, patch, monkeypatch, touched)

        assert PTU_COST_ATTRIBUTION_ENV_VAR in exc.value.detail
        assert touched == []

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


class TestPtuCostAttributionGate:
    """PTU config is only writable once an operator sets LITELLM_ENABLE_PTU_COST_ATTRIBUTION.

    The fields are rejected rather than dropped: a silent accept-and-drop would let a
    caller believe a flat cost was configured while the rollup that prices it is not
    even scheduled.
    """

    @pytest.fixture(autouse=True)
    def _flag_off(self, monkeypatch):
        monkeypatch.delenv(PTU_COST_ATTRIBUTION_ENV_VAR, raising=False)

    @pytest.fixture
    def flag_on(self, monkeypatch):
        monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")

    @pytest.mark.parametrize(
        "model_info",
        [
            {"team_id": "t", "ptu_count": 5, "cost_per_ptu_per_hour": 2.0},
            {"ptu_count": 5},
            {"cost_per_ptu_per_hour": 2.0},
            {"ptu_effective_from": "2026-08-01T00:00:00Z"},
            {"ptu_effective_to": "2026-08-02T00:00:00Z"},
        ],
    )
    def test_rejects_any_ptu_field_while_disabled(self, model_info):
        with pytest.raises(HTTPException) as exc:
            _raise_if_ptu_cost_attribution_disabled(model_info)
        assert exc.value.status_code == 400
        assert PTU_COST_ATTRIBUTION_ENV_VAR in exc.value.detail

    def test_names_every_offending_field(self):
        with pytest.raises(HTTPException) as exc:
            _raise_if_ptu_cost_attribution_disabled({"ptu_count": 5, "cost_per_ptu_per_hour": 2.0})
        assert "ptu_count" in exc.value.detail
        assert "cost_per_ptu_per_hour" in exc.value.detail

    def test_allows_a_request_without_ptu_fields_while_disabled(self):
        _raise_if_ptu_cost_attribution_disabled({"team_id": "t", "access_groups": ["a"]})

    def test_allows_every_ptu_field_once_enabled(self, flag_on):
        _raise_if_ptu_cost_attribution_disabled(
            {
                "team_id": "t",
                "ptu_count": 5,
                "cost_per_ptu_per_hour": 2.0,
                "ptu_effective_from": "2026-08-01T00:00:00Z",
                "ptu_effective_to": "2026-08-02T00:00:00Z",
            }
        )


def _deployment_without_ptu() -> Deployment:
    return Deployment(
        model_name="gpt-4o",
        litellm_params=LiteLLM_Params(model="openai/gpt-4o"),
        model_info=ModelInfo(id="dep-0", team_id="t"),
    )


def _deployment_with_stored_ptu() -> Deployment:
    return Deployment(
        model_name="gpt-4o",
        litellm_params=LiteLLM_Params(model="openai/gpt-4o"),
        model_info=ModelInfo(
            id="dep-0",
            team_id="t",
            ptu_count=15,
            cost_per_ptu_per_hour=2.0,
            ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
        ),
    )


class TestUpdateDbModelPtuGate:
    @pytest.fixture(autouse=True)
    def _flag_off(self, monkeypatch):
        monkeypatch.delenv(PTU_COST_ATTRIBUTION_ENV_VAR, raising=False)

    def test_patch_carrying_ptu_config_is_rejected(self):
        with pytest.raises(HTTPException) as exc:
            update_db_model(
                db_model=_deployment_without_ptu(),
                updated_patch=updateDeployment(model_info=ModelInfo(id="dep-0", team_id="t", ptu_count=15)),
            )
        assert exc.value.status_code == 400

    def test_patch_that_touches_nothing_ptu_still_succeeds(self):
        result = update_db_model(
            db_model=_deployment_without_ptu(),
            updated_patch=updateDeployment(model_info=ModelInfo(id="dep-0", access_groups=["a"])),
        )
        assert json.loads(result["model_info"])["access_groups"] == ["a"]

    def test_unrelated_patch_of_a_model_that_stores_ptu_config_is_not_blocked(self):
        """A deployment configured during an earlier opt-in stays editable: the gate reads the
        incoming patch, not the merged deployment, so the stored config is left in place."""
        result = update_db_model(
            db_model=_deployment_with_stored_ptu(),
            updated_patch=updateDeployment(model_name="gpt-4o-renamed"),
        )
        assert result["model_name"] == "gpt-4o-renamed"

    def test_explicit_nulls_do_not_erase_stored_ptu_config_while_disabled(self):
        """A client round-tripping a model_info blob sends the PTU keys as nulls. While the
        feature is disabled those nulls must not reach the clear loop: disabling pauses PTU,
        it does not silently discard a billing configuration the operator set up earlier."""
        result = update_db_model(
            db_model=_deployment_with_stored_ptu(),
            updated_patch=updateDeployment(
                model_info=ModelInfo(id="dep-0", ptu_count=None, cost_per_ptu_per_hour=None)
            ),
        )
        stored = json.loads(result["model_info"])
        assert stored["ptu_count"] == 15
        assert stored["cost_per_ptu_per_hour"] == 2.0

    def test_the_merged_view_agrees_with_the_write_while_disabled(self):
        """The validator sees what the write will store. If the merged view honoured a null the
        clear loop ignores, a round-tripped blob would 400 on a half-set pair that never forms."""
        merged = _merged_ptu_model_info(
            db_model=_deployment_with_stored_ptu(),
            patch_data=updateDeployment(model_info=ModelInfo(id="dep-0", ptu_count=None)),
        )
        assert merged["ptu_count"] == 15
        _validate_ptu_model_info(merged)

    def test_explicit_nulls_still_clear_once_enabled(self, monkeypatch):
        """Clearing remains available to an operator who opted in, which is how PTU config is
        removed from a deployment."""
        monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")
        result = update_db_model(
            db_model=_deployment_with_stored_ptu(),
            updated_patch=updateDeployment(
                model_info=ModelInfo(id="dep-0", ptu_count=None, cost_per_ptu_per_hour=None)
            ),
        )
        stored = json.loads(result["model_info"])
        assert "ptu_count" not in stored
        assert "cost_per_ptu_per_hour" not in stored

    def test_patch_carrying_ptu_config_is_accepted_once_enabled(self, monkeypatch):
        monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")
        result = update_db_model(
            db_model=_deployment_without_ptu(),
            updated_patch=updateDeployment(
                model_info=ModelInfo(
                    id="dep-0",
                    team_id="t",
                    ptu_count=15,
                    cost_per_ptu_per_hour=2.0,
                    ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                )
            ),
        )
        stored = json.loads(result["model_info"])
        assert stored["ptu_count"] == 15
        assert stored["cost_per_ptu_per_hour"] == 2.0


class TestAddNewModelPtuGate:
    @pytest.fixture(autouse=True)
    def _flag_off(self, monkeypatch):
        monkeypatch.delenv(PTU_COST_ATTRIBUTION_ENV_VAR, raising=False)

    @staticmethod
    def _patched_proxy(model_id: str):
        """Patch everything /model/new touches except the PTU gate, and hand back the DB writers."""
        db_row = LiteLLM_ProxyModelTable(
            model_id=model_id,
            model_name="ptu-model",
            litellm_params={"model": "openai/gpt-4.1-nano"},
            model_info={"id": model_id},
            created_by="test-admin",
            updated_by="test-admin",
        )
        add_model_to_db = AsyncMock(return_value=db_row)
        add_team_model_to_db = AsyncMock(return_value=db_row)

        mock_proxy_config = MagicMock()
        # Both fields None: no reconcile state was captured, so the serving verdict
        # falls back to reading the router live -- which is what mock_router below
        # drives. These tests are about the PTU gate, not the reload verdict.
        mock_proxy_config.add_deployment = AsyncMock(
            return_value=ReconcileOutcome(still_desired=None, live_after=None)
        )

        mock_router = MagicMock()
        mock_router.get_model_ids.return_value = [model_id]

        proxy_server = "litellm.proxy.proxy_server"
        endpoints = "litellm.proxy.management_endpoints.model_management_endpoints"
        return (add_model_to_db, add_team_model_to_db), [
            patch(f"{proxy_server}.prisma_client", MagicMock()),
            patch(f"{proxy_server}.store_model_in_db", True),
            patch(f"{proxy_server}.proxy_config", mock_proxy_config),
            patch(f"{proxy_server}.proxy_logging_obj", MagicMock()),
            patch(f"{proxy_server}.general_settings", {}),
            patch(f"{proxy_server}.premium_user", True),
            patch(f"{proxy_server}.llm_router", mock_router),
            patch(
                f"{endpoints}.ModelManagementAuthChecks.can_user_make_model_call",
                AsyncMock(return_value=True),
            ),
            patch(f"{endpoints}._add_model_to_db", add_model_to_db),
            patch(f"{endpoints}._add_team_model_to_db", add_team_model_to_db),
        ]

    @staticmethod
    def _ptu_deployment(model_id: str) -> Deployment:
        return Deployment(
            model_name="ptu-model",
            litellm_params=LiteLLM_Params(model="openai/gpt-4.1-nano", api_key="fake-key"),
            model_info=ModelInfo(
                id=model_id,
                team_id="team-1",
                ptu_count=15,
                cost_per_ptu_per_hour=2.0,
                ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
            ),
        )

    @pytest.mark.asyncio
    async def test_model_new_rejects_ptu_config_while_disabled(self):
        (add_model_to_db, add_team_model_to_db), patches = self._patched_proxy("ptu-gate-model")
        admin = UserAPIKeyAuth(user_id="test-admin", user_role=LitellmUserRoles.PROXY_ADMIN)

        with ExitStack() as stack:
            for active_patch in patches:
                stack.enter_context(active_patch)
            with pytest.raises(Exception) as exc:
                await add_new_model(model_params=self._ptu_deployment("ptu-gate-model"), user_api_key_dict=admin)

        assert PTU_COST_ATTRIBUTION_ENV_VAR in str(exc.value)
        add_model_to_db.assert_not_called()
        add_team_model_to_db.assert_not_called()

    @pytest.mark.asyncio
    async def test_model_new_accepts_a_deployment_without_ptu_config_while_disabled(self):
        _, patches = self._patched_proxy("plain-model")
        admin = UserAPIKeyAuth(user_id="test-admin", user_role=LitellmUserRoles.PROXY_ADMIN)

        with ExitStack() as stack:
            for active_patch in patches:
                stack.enter_context(active_patch)
            result = await add_new_model(
                model_params=Deployment(
                    model_name="ptu-model",
                    litellm_params=LiteLLM_Params(model="openai/gpt-4.1-nano", api_key="fake-key"),
                    model_info=ModelInfo(id="plain-model"),
                ),
                user_api_key_dict=admin,
            )

        assert result.model_id == "plain-model"

    @pytest.mark.asyncio
    async def test_model_new_accepts_ptu_config_once_enabled(self, monkeypatch):
        monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")
        (_, add_team_model_to_db), patches = self._patched_proxy("ptu-gate-model")
        admin = UserAPIKeyAuth(user_id="test-admin", user_role=LitellmUserRoles.PROXY_ADMIN)

        with ExitStack() as stack:
            for active_patch in patches:
                stack.enter_context(active_patch)
            result = await add_new_model(model_params=self._ptu_deployment("ptu-gate-model"), user_api_key_dict=admin)

        assert result.model_id == "ptu-gate-model"
        add_team_model_to_db.assert_called_once()
