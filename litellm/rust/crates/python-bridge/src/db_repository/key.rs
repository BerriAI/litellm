//! Python's handle on the key's database read in `auth-store`: the `combined_view` row shaped into the
//! entry the auth cache holds. The proxy wraps this one query in its lookup gate and deadline, nests the object
//! permission from a read it bounds separately, and caches the result itself.

use std::sync::Arc;

use litellm_auth_store::{Error, KeyDatabase};
use litellm_gateway_auth_core::KeyHash;
use litellm_redis_store::IntoValue;
use litellm_repositories::{KeyRepository, ObjectPermissionRepository};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use crate::{
    db::{NativeDatabase, map_error},
    redis::store::value::PyValue,
};

/// A database failure is `NativeDatabaseFailure`, as it is from every Rust repository, so the caller's
/// fail-open check reads it the same way. A row the auth model would refuse fails the read.
fn raised(error: Error) -> PyErr {
    match error {
        Error::Database(error) => map_error(error),
        Error::Row { .. } => {
            PyRuntimeError::new_err(litellm_error::Report::new(&error).to_string())
        }
    }
}

/// The virtual key repository. `find` answers the `combined_view` read for a hashed key as the plain data the
/// auth cache holds for it, with no object permission nested, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustKeyRepository(Arc<KeyDatabase>);

#[pymethods]
impl RustKeyRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(KeyDatabase::new(
            Arc::new(KeyRepository::new(database.repository_database()?)),
            Arc::new(ObjectPermissionRepository::new(
                database.repository_database()?,
            )),
        ))))
    }

    #[pyo3(signature = (token, *, use_writer=false))]
    fn find<'py>(
        &self,
        py: Python<'py>,
        token: String,
        use_writer: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let keys = Arc::clone(&self.0);
        let side = super::side(use_writer);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(
                py,
                Box::pin(async move {
                    let key = keys.key(&KeyHash::new(token), side).await.map_err(raised)?;
                    Ok(key.map(|key| PyValue(key.into_value())))
                }),
            )?,
        )
    }
}
