use std::sync::Arc;

use litellm_repositories::{NewUserDefaults, Side, SignIn, UserRepository};
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The user repository. A user is answered with its organization memberships, as JSON text, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustUserRepository(Arc<UserRepository>);

#[pymethods]
impl RustUserRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(UserRepository::new(
            database.repository_database()?,
        ))))
    }

    #[pyo3(signature = (
        user_id,
        *,
        sso_user_id=None,
        user_email=None,
        email_verified=None,
        links_any_user,
        use_writer=false
    ))]
    #[expect(
        clippy::too_many_arguments,
        reason = "Each argument is one keyword of the Python call"
    )]
    fn resolve<'py>(
        &self,
        py: Python<'py>,
        user_id: String,
        sso_user_id: Option<String>,
        user_email: Option<String>,
        email_verified: Option<bool>,
        links_any_user: bool,
        use_writer: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        let sign_in = SignIn {
            user_id,
            sso_user_id,
            user_email,
            email_verified,
            links_any_user,
        };
        let side = if use_writer { Side::Writer } else { Side::Read };
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                super::json(
                    repository
                        .resolve(&sign_in, side)
                        .await
                        .map_err(map_error)?
                        .as_ref(),
                )
            })?,
        )
    }

    fn email_belongs_to_another_user<'py>(
        &self,
        py: Python<'py>,
        user_email: String,
        user_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                repository
                    .email_belongs_to_another_user(&user_email, &user_id)
                    .await
                    .map_err(map_error)
            })?,
        )
    }

    #[pyo3(signature = (user_id, user_email, *, links_any_user))]
    fn backfill_email<'py>(
        &self,
        py: Python<'py>,
        user_id: String,
        user_email: String,
        links_any_user: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                repository
                    .backfill_email(&user_id, &user_email, links_any_user)
                    .await
                    .map_err(map_error)
            })?,
        )
    }

    /// The user with its role set to `user_role`, or `None` when no row has that id.
    fn set_role<'py>(
        &self,
        py: Python<'py>,
        user_id: String,
        user_role: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                super::json(
                    repository
                        .set_role(&user_id, &user_role)
                        .await
                        .map_err(map_error)?
                        .as_ref(),
                )
            })?,
        )
    }

    /// `(created, user)`: whether this call created the user, then the user now stored. `budget_reset_at` is
    /// epoch seconds and `budget_limits` JSON text.
    #[pyo3(signature = (
        user_id,
        *,
        sso_user_id=None,
        user_email=None,
        links_any_user,
        user_role=None,
        max_budget=None,
        budget_duration=None,
        budget_reset_at=None,
        models=None,
        budget_limits=None,
        tpm_limit=None,
        rpm_limit=None,
        max_parallel_requests=None,
        user_alias=None,
        team_id=None,
        organization_id=None,
        object_permission_id=None,
        password_reset_required=None,
        metadata=None,
        allowed_cache_controls=None,
        policies=None,
        model_max_budget=None
    ))]
    #[expect(
        clippy::too_many_arguments,
        reason = "Each argument is one keyword of the Python call"
    )]
    fn insert<'py>(
        &self,
        py: Python<'py>,
        user_id: String,
        sso_user_id: Option<String>,
        user_email: Option<String>,
        links_any_user: bool,
        user_role: Option<String>,
        max_budget: Option<f64>,
        budget_duration: Option<String>,
        budget_reset_at: Option<f64>,
        models: Option<Vec<String>>,
        budget_limits: Option<String>,
        tpm_limit: Option<i64>,
        rpm_limit: Option<i64>,
        max_parallel_requests: Option<i32>,
        user_alias: Option<String>,
        team_id: Option<String>,
        organization_id: Option<String>,
        object_permission_id: Option<String>,
        password_reset_required: Option<bool>,
        metadata: Option<String>,
        allowed_cache_controls: Option<Vec<String>>,
        policies: Option<Vec<String>>,
        model_max_budget: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        let sign_in = SignIn {
            user_id,
            sso_user_id,
            user_email,
            email_verified: None,
            links_any_user,
        };
        let defaults = NewUserDefaults {
            user_role,
            max_budget,
            budget_duration,
            budget_reset_at: budget_reset_at.map(crate::auth::convert::timestamp),
            models,
            budget_limits: json_argument("budget_limits", budget_limits.as_deref())?,
            tpm_limit,
            rpm_limit,
            max_parallel_requests,
            user_alias,
            team_id,
            organization_id,
            object_permission_id,
            password_reset_required,
            metadata: json_argument("metadata", metadata.as_deref())?,
            allowed_cache_controls,
            policies,
            model_max_budget: json_argument("model_max_budget", model_max_budget.as_deref())?,
        };
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                super::inserted_json(
                    repository
                        .insert(&sign_in, &defaults)
                        .await
                        .map_err(map_error)?,
                )
            })?,
        )
    }
}

fn json_argument(name: &str, text: Option<&str>) -> PyResult<Option<serde_json::Value>> {
    text.map(|text| litellm_host_python::json_parse(name, text))
        .transpose()
}
