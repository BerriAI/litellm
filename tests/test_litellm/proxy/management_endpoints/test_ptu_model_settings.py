"""Tests for PTU config on the model deployment (v1 model-settings design)."""

import datetime
import json
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch
from unittest.mock import patch as patch_ctx

import pytest
from fastapi import HTTPException

from litellm.proxy._types import (
    LiteLLM_ProxyModelTable,
    LitellmUserRoles,
    ReconcileOutcome,
    UserAPIKeyAuth,
)
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.proxy.auth.auth_checks import _is_model_cost_zero
from litellm.llms.gemini.cost_calculator import cost_per_web_search_request
from litellm.proxy.management_endpoints.model_management_endpoints import (
    _PTU_ZEROED_PRICING_FIELDS,
    _SEARCH_CONTEXT_SIZES,
    _is_nonzero_price,
    _merged_ptu_model_info,
    _update_team_model_in_db,
    _ptu_priced_deployment,
    _ptu_zeroed_pricing,
    _raise_if_ptu_cost_attribution_disabled,
    _validate_ptu_model_info,
    add_new_model,
    update_db_model,
)
from litellm.proxy.spend_tracking.ptu_feature_flag import PTU_COST_ATTRIBUTION_ENV_VAR
from litellm.types.utils import PromptTokensDetailsWrapper
from litellm.router import Router
from litellm.types.router import (
    SPECIAL_MODEL_INFO_PARAMS,
    Deployment,
    LiteLLM_Params,
    ModelInfo,
    updateDeployment,
    updateLiteLLMParams,
)
from litellm.types.utils import Usage


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
    with pytest.raises(ValueError, match='value_error, input_value'):
        ModelInfo(
            id="x",
            team_id="t",
            ptu_count=0,
            cost_per_ptu_per_hour=2.0,
            ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
        )


def test_model_info_rejects_negative_rate():
    with pytest.raises(ValueError, match='value_error, input_value'):
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
    with pytest.raises(ValueError, match='validation error for ModelInfo'):
        ModelInfo(id="x", team_id="t", ptu_count=10**400, cost_per_ptu_per_hour=2.0)


def test_model_info_accepts_a_count_at_the_cap():
    info = ModelInfo(id="x", team_id="t", ptu_count=ModelInfo.MAX_PTU_COUNT, cost_per_ptu_per_hour=2.0)
    assert info.ptu_count == ModelInfo.MAX_PTU_COUNT


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), float("-inf")])
def test_model_info_rejects_a_non_finite_rate(rate):
    """NaN compares False against every bound, so a bare `< 0` check let it through and the
    deployment then accrued a flat cost of nan."""
    with pytest.raises(ValueError, match='value_error, input_value'):
        ModelInfo(id="x", team_id="t", ptu_count=5, cost_per_ptu_per_hour=rate)


def test_model_info_rejects_a_rate_beyond_the_cap():
    with pytest.raises(ValueError, match='validation error for ModelInfo'):
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

    with pytest.raises(ValueError, match='validation error for ModelInfo'):
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

    with pytest.raises(ValueError, match='validation error for ModelInfo'):
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
            with pytest.raises(Exception, match='PTU cost attribution is disabled, so ptu_count') as exc:
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



class TestPtuDeploymentsAreNotBilledPerToken:
    """Reserved capacity is billed by the flat cost the rollup writes, so a PTU deployment must
    not also bill the traffic that capacity serves."""

    PTU = {"ptu_count": 15, "cost_per_ptu_per_hour": 2.0}

    @pytest.fixture(autouse=True)
    def _flag_on(self, monkeypatch):
        monkeypatch.setenv(PTU_COST_ATTRIBUTION_ENV_VAR, "true")
        # update_db_model encrypts every litellm_params value it is handed, and the salt falls
        # back to the master key the proxy sets at boot, which no unit test has.
        monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key")

    @staticmethod
    def _zeroed(model_info=None, litellm_params=None, supplied=None):
        return _ptu_zeroed_pricing(
            model_info=model_info if model_info is not None else {},
            litellm_params=litellm_params if litellm_params is not None else {},
            supplied=supplied if supplied is not None else {},
        )

    def test_a_deployment_without_ptu_config_keeps_its_pricing(self):
        assert self._zeroed(model_info={"team_id": "t"}, litellm_params={"input_cost_per_token": 5e-07}) == {}

    def test_a_half_set_pair_is_not_treated_as_ptu(self):
        assert self._zeroed(model_info={"ptu_count": 15}) == {}

    def test_every_field_the_cost_map_could_fill_is_zeroed(self):
        assert self._zeroed(model_info=self.PTU) == {
            **dict.fromkeys(_PTU_ZEROED_PRICING_FIELDS, 0.0),
            "tiered_pricing": (),
            "search_context_cost_per_query": dict.fromkeys(_SEARCH_CONTEXT_SIZES, 0.0),
        }

    def test_nothing_is_zeroed_while_the_feature_is_disabled(self, monkeypatch):
        monkeypatch.delenv(PTU_COST_ATTRIBUTION_ENV_VAR, raising=False)
        assert self._zeroed(model_info=self.PTU) == {}

    @pytest.mark.parametrize("field", ["input_cost_per_token", "cache_read_input_token_cost", "input_cost_per_second"])
    def test_a_price_the_caller_supplies_is_refused(self, field):
        """Every custom-pricing field, not only the mirrored ones: per-second pricing bills a
        PTU deployment just as surely as per-token pricing does."""
        with pytest.raises(HTTPException) as exc:
            self._zeroed(model_info=self.PTU, supplied={field: 5e-07})
        assert exc.value.status_code == 400
        assert field in str(exc.value.detail)

    def test_a_tiered_price_the_caller_supplies_is_refused(self):
        """Tier rates bill the traffic per token just as surely as a flat rate does."""
        with pytest.raises(HTTPException) as exc:
            self._zeroed(model_info=self.PTU, supplied={"tiered_pricing": [{"range": [0, 100], "input_cost_per_token": 1e-06}]})
        assert exc.value.status_code == 400
        assert "tiered_pricing" in str(exc.value.detail)

    def test_a_search_context_price_the_caller_supplies_is_refused(self):
        """The rates sit in a table keyed by context size, so a guard that only reads numbers
        lets a per-request charge onto a deployment its reserved capacity already pays for."""
        with pytest.raises(HTTPException) as exc:
            self._zeroed(model_info=self.PTU, supplied={"search_context_cost_per_query": {"search_context_size_medium": 0.05}})
        assert exc.value.status_code == 400
        assert "search_context_cost_per_query" in str(exc.value.detail)

    def test_search_context_already_on_the_row_is_zeroed_in_place(self):
        """An absent table means the provider's own default rate rather than free, so emptying or
        dropping this one would start a charge instead of stopping it."""
        stored = {"search_context_cost_per_query": {"search_context_size_medium": 0.05}}
        override = self._zeroed(model_info=self.PTU, litellm_params=stored)
        assert override["search_context_cost_per_query"] == dict.fromkeys(_SEARCH_CONTEXT_SIZES, 0.0)
        assert cost_per_web_search_request(usage=self._grounded_usage(), model_info={**stored, **override}) == 0
        assert cost_per_web_search_request(usage=self._grounded_usage(), model_info={}) > 0

    def test_an_all_zero_search_context_table_is_not_a_price(self):
        """An all-zero table is how an operator expresses free, so refusing it would block the save
        and replacing it would restore the provider default."""
        free = dict.fromkeys(_SEARCH_CONTEXT_SIZES, 0.0)
        assert self._zeroed(model_info=self.PTU, supplied={"search_context_cost_per_query": free})
        assert cost_per_web_search_request(usage=self._grounded_usage(), model_info={"search_context_cost_per_query": free}) == 0

    @staticmethod
    def _grounded_usage():
        return Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=1),
        )

    def test_tiered_pricing_already_on_the_row_is_emptied_not_zeroed(self):
        """tiered_pricing is a table of ranges, so the zero the other fields store would not even
        validate. Dropping it instead would fall back to the cost map's tiers, whose rates outrank
        the zeros written beside them, so it is stored empty."""
        tiers = [{"range": [0, 128000], "input_cost_per_token": 3e-06}]
        priced = _ptu_priced_deployment(
            Deployment(
                model_name="tiered",
                litellm_params=LiteLLM_Params(model="openai/gpt-4o"),
                model_info=ModelInfo(
                    id="dep-tiered",
                    team_id="t",
                    tiered_pricing=tiers,
                    ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                    **self.PTU,
                ),
            )
        )
        assert priced.litellm_params.tiered_pricing == []
        assert priced.model_info.tiered_pricing == []

        written = update_db_model(
            db_model=Deployment(
                model_name="tiered",
                litellm_params=LiteLLM_Params(model="openai/gpt-4o", tiered_pricing=tiers),
                model_info=ModelInfo(id="dep-tiered", team_id="t"),
            ),
            updated_patch=updateDeployment(
                model_info=ModelInfo(
                    id="dep-tiered",
                    team_id="t",
                    ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                    **self.PTU,
                )
            ),
        )
        for blob in ("model_info", "litellm_params"):
            stored = json.loads(written[blob])
            assert stored["tiered_pricing"] == [], blob
            assert stored["input_cost_per_token"] == 0, blob

    def test_a_price_the_caller_supplies_as_zero_is_accepted(self):
        assert self._zeroed(model_info={**self.PTU, "input_cost_per_token": 0}, supplied={"input_cost_per_token": 0})[
            "input_cost_per_token"
        ] == 0

    def test_a_price_already_on_the_row_is_zeroed_rather_than_refused(self):
        """A row priced through a path this rule does not cover must heal on its next save. The
        alternative refuses every later edit of a field that has nothing to do with pricing."""
        zeroed = self._zeroed(model_info={**self.PTU, "input_cost_per_second": 3.0}, litellm_params={})
        assert zeroed["input_cost_per_second"] == 0
        assert zeroed["input_cost_per_token"] == 0

    @pytest.mark.asyncio
    async def test_a_refused_price_does_not_leave_the_team_changed(self):
        """The team ACL write autocommits, so the refusal has to run before it. Otherwise a
        rejected edit grants the team a model whose settings were never saved."""
        db_model = Deployment(
            model_name="gpt-4o",
            litellm_params=LiteLLM_Params(model="openai/gpt-4o"),
            model_info=ModelInfo(
                id="dep-0",
                team_id="team-1",
                ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                **self.PTU,
            ),
        )
        patch = updateDeployment(
            litellm_params=updateLiteLLMParams(model="openai/gpt-4o", input_cost_per_token=5e-07),
            model_info=ModelInfo(id="dep-0", team_id="team-2"),
        )
        endpoints = "litellm.proxy.management_endpoints.model_management_endpoints"
        setup_new = AsyncMock()
        update_existing = AsyncMock()
        with ExitStack() as stack:
            stack.enter_context(
                patch_ctx(f"{endpoints}.ModelManagementAuthChecks.allow_team_model_action", AsyncMock(return_value=True))
            )
            stack.enter_context(patch_ctx(f"{endpoints}._setup_new_team_model_assignment", setup_new))
            stack.enter_context(patch_ctx(f"{endpoints}._update_existing_team_model_assignment", update_existing))
            stack.enter_context(patch_ctx("litellm.proxy.proxy_server.premium_user", True))
            with pytest.raises(HTTPException) as exc:
                await _update_team_model_in_db(
                    db_model=db_model,
                    patch_data=patch,
                    user_api_key_dict=UserAPIKeyAuth(user_id="a", user_role=LitellmUserRoles.PROXY_ADMIN),
                    prisma_client=MagicMock(),
                )

        assert exc.value.status_code == 400
        setup_new.assert_not_called()
        update_existing.assert_not_called()

    def test_a_setting_that_is_not_a_charge_is_left_alone(self):
        """CustomPricingLiteLLMParams also carries an embedding's output vector size and the
        regional uplift multipliers. Zeroing one of those destroys the deployment's config, and
        refusing it answers with a message calling a setting a charge."""
        priced = _ptu_priced_deployment(
            Deployment(
                model_name="embeddings",
                litellm_params=LiteLLM_Params(
                    model="azure/text-embedding-3-large",
                    output_vector_size=1536,
                    regional_processing_uplift_multiplier_eu=1.15,
                ),
                model_info=ModelInfo(
                    id="dep-emb",
                    team_id="team-1",
                    ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                    **self.PTU,
                ),
            )
        )
        assert priced.litellm_params.get("output_vector_size") == 1536
        assert priced.litellm_params.get("regional_processing_uplift_multiplier_eu") == 1.15
        assert priced.litellm_params.get("input_cost_per_token") == 0

    def test_removing_ptu_config_releases_every_rate_it_zeroed(self):
        """The zeroing covers any stored rate, so a release that only spans the mirrored fields
        leaves a per-second deployment billing nothing for that dimension forever."""
        on = update_db_model(
            db_model=Deployment(
                model_name="audio",
                litellm_params=LiteLLM_Params(model="azure/whisper", input_cost_per_second=0.006),
                model_info=ModelInfo(id="dep-audio", team_id="t"),
            ),
            updated_patch=updateDeployment(
                model_info=ModelInfo(
                    id="dep-audio",
                    team_id="t",
                    ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                    **self.PTU,
                )
            ),
        )
        assert json.loads(on["litellm_params"])["input_cost_per_second"] == 0

        off = update_db_model(
            db_model=Deployment(
                model_name="audio",
                litellm_params=LiteLLM_Params(**json.loads(on["litellm_params"])),
                model_info=ModelInfo(**json.loads(on["model_info"])),
            ),
            updated_patch=updateDeployment(
                model_info=ModelInfo(id="dep-audio", ptu_count=None, cost_per_ptu_per_hour=None)
            ),
        )
        assert "input_cost_per_second" not in json.loads(off["litellm_params"])

    def test_removing_ptu_config_releases_a_zeroed_search_context_table(self):
        """The all-zero table exists only to stop the double charge, so a deployment taken off PTU
        has to give it up or it keeps serving grounded requests for free forever."""
        on = update_db_model(
            db_model=Deployment(
                model_name="grounded",
                litellm_params=LiteLLM_Params(
                    model="gemini/gemini-2.5-pro",
                    search_context_cost_per_query={"search_context_size_medium": 0.05},
                ),
                model_info=ModelInfo(id="dep-ground", team_id="t"),
            ),
            updated_patch=updateDeployment(
                model_info=ModelInfo(
                    id="dep-ground",
                    team_id="t",
                    ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                    **self.PTU,
                )
            ),
        )
        assert json.loads(on["litellm_params"])["search_context_cost_per_query"] == dict.fromkeys(
            _SEARCH_CONTEXT_SIZES, 0.0
        )

        off = update_db_model(
            db_model=Deployment(
                model_name="grounded",
                litellm_params=LiteLLM_Params(**json.loads(on["litellm_params"])),
                model_info=ModelInfo(**json.loads(on["model_info"])),
            ),
            updated_patch=updateDeployment(
                model_info=ModelInfo(id="dep-ground", ptu_count=None, cost_per_ptu_per_hour=None)
            ),
        )
        assert "search_context_cost_per_query" not in json.loads(off["litellm_params"])

    @pytest.mark.parametrize(
        "backend", ["azure/gpt-4o", "anthropic/claude-sonnet-4-5", "bedrock/anthropic.claude-sonnet-4-20250514-v1:0"]
    )
    def test_the_cost_map_contributes_no_price_to_a_priced_ptu_deployment(self, backend):
        """The acceptance criterion, read off the entry the router registers for the deployment.

        Zeroing only the per-token pair leaves the cache-tier fields unset, which is exactly what
        Router._inherit_builtin_cache_pricing back-fills from the public cost map, so a cached
        prompt would still be billed at the public rate."""
        priced = _ptu_priced_deployment(
            Deployment(
                model_name="ptu-deployment",
                litellm_params=LiteLLM_Params(model=backend, api_key="fake-key"),
                model_info=ModelInfo(
                    id="dep-ptu",
                    team_id="team-1",
                    ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                    **self.PTU,
                ),
            )
        )
        registered = Router._deployment_model_cost_payload(priced)
        charged = {
            k: v
            for k, v in registered.items()
            if "cost" in k and k != "cost_per_ptu_per_hour" and _is_nonzero_price(v)
        }
        assert charged == {}

    def test_the_cost_map_tiers_contribute_no_price_to_a_priced_ptu_deployment(self):
        """A tier table outranks the zeroed flat rates wherever cost is read, so leaving the
        deployment's own table unset bills the reserved capacity's traffic at the map's tiers."""
        priced = _ptu_priced_deployment(
            Deployment(
                model_name="ptu-deployment",
                litellm_params=LiteLLM_Params(model="dashscope/qwen-flash", api_key="fake-key"),
                model_info=ModelInfo(
                    id="dep-ptu",
                    team_id="team-1",
                    ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                    **self.PTU,
                ),
            )
        )
        router = Router(model_list=[priced.to_json(exclude_none=True)])
        registered = router.get_deployment_model_info(model_id="dep-ptu", model_name="dashscope/qwen-flash")
        assert registered is not None
        assert registered["tiered_pricing"] == []
        assert generic_cost_per_token(
            model="dashscope/qwen-flash",
            usage=Usage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100),
            custom_llm_provider="dashscope",
            model_info=registered,
        ) == (0.0, 0.0)

    def test_the_zeroed_pricing_does_not_waive_budget_enforcement(self):
        """A zero price otherwise tells auth the model is free and skips every budget check."""
        priced = _ptu_priced_deployment(
            Deployment(
                model_name="model_name_team-1_dep-ptu",
                litellm_params=LiteLLM_Params(model="gemini/gemini-2.5-flash", api_key="fake-key"),
                model_info=ModelInfo(
                    id="dep-ptu",
                    team_id="team-1",
                    team_public_model_name="ptu-model",
                    ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                    **self.PTU,
                ),
            )
        )
        router = Router(model_list=[priced.to_json(exclude_none=True)])
        assert _is_model_cost_zero(model="model_name_team-1_dep-ptu", llm_router=router) is False
        assert _is_model_cost_zero(model="ptu-model", llm_router=router) is False

    def test_an_unrelated_patch_heals_a_deployment_stored_before_this_rule(self):
        """Both blobs, because litellm_params wins over model_info wherever the two are merged."""
        written = update_db_model(
            db_model=_deployment_with_stored_ptu(),
            updated_patch=updateDeployment(model_name="gpt-4o-renamed"),
        )
        for blob in ("model_info", "litellm_params"):
            stored = json.loads(written[blob])
            assert all(stored[field] == 0 for field in _PTU_ZEROED_PRICING_FIELDS), blob

    def test_an_unrelated_patch_of_a_ptu_row_that_carries_a_price_is_not_refused(self):
        """The pause toggle and the credential-rotation modal send no pricing at all. Refusing
        them because the stored row is mispriced blocks flows that cannot fix it."""
        priced_ptu = Deployment(
            model_name="gpt-4o",
            litellm_params=LiteLLM_Params(model="openai/gpt-4o", input_cost_per_token=5e-07),
            model_info=ModelInfo(
                id="dep-0",
                team_id="t",
                ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                **self.PTU,
            ),
        )
        written = update_db_model(db_model=priced_ptu, updated_patch=updateDeployment(model_name="renamed"))
        assert written["model_name"] == "renamed"
        assert json.loads(written["litellm_params"])["input_cost_per_token"] == 0

    def test_removing_ptu_config_hands_per_token_billing_back(self):
        """Left behind, the zeros this rule wrote would serve the deployment for free forever."""
        zeros = dict.fromkeys(_PTU_ZEROED_PRICING_FIELDS, 0.0)
        written = update_db_model(
            db_model=Deployment(
                model_name="gpt-4o",
                litellm_params=LiteLLM_Params(model="openai/gpt-4o", **zeros),
                model_info=ModelInfo(
                    id="dep-0",
                    team_id="t",
                    ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                    **self.PTU,
                    **zeros,
                ),
            ),
            updated_patch=updateDeployment(
                model_info=ModelInfo(id="dep-0", ptu_count=None, cost_per_ptu_per_hour=None)
            ),
        )
        for blob in ("model_info", "litellm_params"):
            stored = json.loads(written[blob])
            assert not any(field in stored for field in _PTU_ZEROED_PRICING_FIELDS), blob

    def test_the_dashboard_clear_releases_the_zeros_it_echoes_back(self):
        """The edit form re-sends the whole stored model_info on every save, so the clearing
        patch carries the zeros this rule wrote. Treating those as a rate the operator chose
        left the deployment serving free and reading as a free model to the budget checks."""
        zeros = dict.fromkeys(_PTU_ZEROED_PRICING_FIELDS, 0.0)
        written = update_db_model(
            db_model=Deployment(
                model_name="gpt-4o",
                litellm_params=LiteLLM_Params(model="openai/gpt-4o", **zeros),
                model_info=ModelInfo(
                    id="dep-0",
                    team_id="t",
                    ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                    **self.PTU,
                    **zeros,
                ),
            ),
            updated_patch=updateDeployment(
                model_info=ModelInfo(id="dep-0", ptu_count=None, cost_per_ptu_per_hour=None, **zeros)
            ),
        )
        stored = json.loads(written["model_info"])
        assert not any(field in stored for field in _PTU_ZEROED_PRICING_FIELDS)

    def test_a_deployment_that_never_had_ptu_keeps_a_price_its_operator_set_to_zero(self):
        """The dashboard sends both PTU keys as null on every save while the feature is on, so a
        release keyed on the patch alone would strip a deliberate zero rate from any model."""
        free = Deployment(
            model_name="free-model",
            litellm_params=LiteLLM_Params(model="openai/gpt-4o", input_cost_per_token=0.0),
            model_info=ModelInfo(id="dep-free", team_id="t", input_cost_per_token=0.0),
        )
        written = update_db_model(
            db_model=free,
            updated_patch=updateDeployment(
                model_info=ModelInfo(id="dep-free", ptu_count=None, cost_per_ptu_per_hour=None)
            ),
        )
        for blob in ("model_info", "litellm_params"):
            assert json.loads(written[blob])["input_cost_per_token"] == 0, blob

    def test_a_patch_pricing_a_ptu_deployment_is_refused(self):
        with pytest.raises(HTTPException) as exc:
            update_db_model(
                db_model=_deployment_with_stored_ptu(),
                updated_patch=updateDeployment(
                    litellm_params=updateLiteLLMParams(model="openai/gpt-4o", input_cost_per_token=5e-07)
                ),
            )
        assert exc.value.status_code == 400

    def test_a_price_the_client_only_echoes_back_is_not_read_as_an_attempt_to_charge(self):
        """/model/info fills missing rates from the public cost map and the edit form re-sends the
        whole blob, so a model_info price is one the server wrote. Reading it as the operator's
        refused every attempt to put an existing deployment on PTU from the dashboard."""
        written = update_db_model(
            db_model=_deployment_without_ptu(),
            updated_patch=updateDeployment(
                model_info=ModelInfo(
                    id="dep-0",
                    team_id="t",
                    input_cost_per_token=3e-07,
                    output_cost_per_token=2.5e-06,
                    ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                    **self.PTU,
                )
            ),
        )
        stored = json.loads(written["model_info"])
        assert stored["ptu_count"] == 15
        assert stored["input_cost_per_token"] == 0
        assert stored["output_cost_per_token"] == 0

    def test_adding_ptu_config_to_an_already_priced_deployment_is_refused(self):
        priced = Deployment(
            model_name="gpt-4o",
            litellm_params=LiteLLM_Params(model="openai/gpt-4o"),
            model_info=ModelInfo(id="dep-0", team_id="t"),
        )
        with pytest.raises(HTTPException) as exc:
            update_db_model(
                db_model=priced,
                updated_patch=updateDeployment(
                    litellm_params=updateLiteLLMParams(model="openai/gpt-4o", input_cost_per_token=5e-07),
                    model_info=ModelInfo(
                        id="dep-0",
                        team_id="t",
                        ptu_effective_from=datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc),
                        **self.PTU,
                    ),
                ),
            )
        assert exc.value.status_code == 400

    def test_a_deployment_without_ptu_config_keeps_its_pricing_through_a_patch(self):
        priced = Deployment(
            model_name="gpt-4o",
            litellm_params=LiteLLM_Params(model="openai/gpt-4o"),
            model_info=ModelInfo(id="dep-0", team_id="t", input_cost_per_token=5e-07),
        )
        stored = json.loads(
            update_db_model(db_model=priced, updated_patch=updateDeployment(model_name="renamed"))["model_info"]
        )
        assert stored["input_cost_per_token"] == 5e-07

    @pytest.mark.asyncio
    async def test_model_new_stores_zero_pricing_on_both_blobs(self):
        (_, add_team_model_to_db), patches = TestAddNewModelPtuGate._patched_proxy("ptu-priced-model")
        admin = UserAPIKeyAuth(user_id="test-admin", user_role=LitellmUserRoles.PROXY_ADMIN)

        with ExitStack() as stack:
            for active_patch in patches:
                stack.enter_context(active_patch)
            await add_new_model(
                model_params=TestAddNewModelPtuGate._ptu_deployment("ptu-priced-model"),
                user_api_key_dict=admin,
            )

        written = add_team_model_to_db.call_args.kwargs["model_params"]
        assert all(getattr(written.model_info, field, None) == 0 for field in SPECIAL_MODEL_INFO_PARAMS if field != "tiered_pricing")
        assert written.model_info.tiered_pricing == []
        assert all(written.litellm_params.get(field) == 0 for field in _PTU_ZEROED_PRICING_FIELDS)
        assert written.litellm_params.tiered_pricing == []

    @pytest.mark.asyncio
    async def test_model_new_refuses_a_priced_ptu_deployment(self):
        (_, add_team_model_to_db), patches = TestAddNewModelPtuGate._patched_proxy("ptu-priced-model")
        admin = UserAPIKeyAuth(user_id="test-admin", user_role=LitellmUserRoles.PROXY_ADMIN)
        base = TestAddNewModelPtuGate._ptu_deployment("ptu-priced-model")
        deployment = base.model_copy(
            update={"litellm_params": base.litellm_params.model_copy(update={"input_cost_per_token": 5e-07})}
        )

        with ExitStack() as stack:
            for active_patch in patches:
                stack.enter_context(active_patch)
            with pytest.raises(Exception, match='A PTU deployment bills by reserved capacity, so') as exc:
                await add_new_model(model_params=deployment, user_api_key_dict=admin)

        assert "input_cost_per_token" in str(exc.value)
        add_team_model_to_db.assert_not_called()
