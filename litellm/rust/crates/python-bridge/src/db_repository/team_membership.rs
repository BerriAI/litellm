use std::sync::Arc;

use litellm_repositories::{
    MemberAdd, MemberBudget, NewMemberBudget, Side, TeamMembershipRepository,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// A member's own budget as Python passes it: `(allowed_models, budget_duration, budget_reset_at, created_by,
/// updated_by)`, with the reset time in epoch seconds.
type OwnBudget = (Vec<String>, Option<String>, Option<f64>, String, String);

/// The team membership repository. A membership is answered with its member budget as JSON text.
#[pyclass(frozen)]
pub(crate) struct RustTeamMembershipRepository(Arc<TeamMembershipRepository>);

#[pymethods]
impl RustTeamMembershipRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(TeamMembershipRepository::new(
            database.repository_database()?,
        ))))
    }

    #[pyo3(signature = (user_id, team_id, *, use_writer=false))]
    fn find_with_budget<'py>(
        &self,
        py: Python<'py>,
        user_id: String,
        team_id: String,
        use_writer: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        let side = if use_writer { Side::Writer } else { Side::Read };
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let membership = repository
                    .find_with_budget(&user_id, &team_id, side)
                    .await
                    .map_err(map_error)?;
                super::json(membership.as_ref())
            })?,
        )
    }

    /// `(outcome, membership, roster)`. The outcome is `added`, `already_member`, `team_missing` or
    /// `user_missing`, and only `added` carries the membership and the team's roster as now stored.
    /// `own_budget` is inserted under `budget_id` in the same transaction.
    #[pyo3(signature = (user_id, team_id, *, budget_id=None, own_budget=None))]
    fn add_member<'py>(
        &self,
        py: Python<'py>,
        user_id: String,
        team_id: String,
        budget_id: Option<String>,
        own_budget: Option<OwnBudget>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        let new_budget = own_budget
            .map(
                |(allowed_models, budget_duration, budget_reset_at, created_by, updated_by)| {
                    Ok::<_, PyErr>(NewMemberBudget {
                        budget_id: budget_id.clone().ok_or_else(|| {
                            PyValueError::new_err(
                                "own_budget needs the budget_id to insert it under",
                            )
                        })?,
                        allowed_models,
                        budget_duration,
                        budget_reset_at: budget_reset_at.map(crate::auth::convert::timestamp),
                        created_by,
                        updated_by,
                    })
                },
            )
            .transpose()?;
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let budget = match (&new_budget, &budget_id) {
                    (Some(new_budget), _) => MemberBudget::New(new_budget),
                    (None, Some(budget_id)) => MemberBudget::Existing(budget_id),
                    (None, None) => MemberBudget::None,
                };
                let added = repository
                    .add_member(&user_id, &team_id, budget)
                    .await
                    .map_err(map_error)?;
                Ok(match added {
                    MemberAdd::Added(added) => (
                        "added",
                        super::json(Some(&added.membership))?,
                        Some(added.roster.to_string()),
                    ),
                    MemberAdd::AlreadyMember => ("already_member", None, None),
                    MemberAdd::TeamMissing => ("team_missing", None, None),
                    MemberAdd::UserMissing => ("user_missing", None, None),
                })
            })?,
        )
    }
}
