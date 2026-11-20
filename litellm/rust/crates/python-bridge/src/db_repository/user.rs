use std::sync::Arc;

use litellm_repositories::{Side, SignIn, UserRepository};
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The user repository. `resolve` answers the user a sign-in resolves to, with its organization memberships,
/// as JSON text, or `None`.
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
}
