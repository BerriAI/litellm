use std::sync::Arc;

use litellm_repositories::KeyRepository;
use pyo3::prelude::*;

use crate::db::{NativeDatabase, map_error};

/// The virtual key repository. `find` answers the `combined_view` read for a hashed key as JSON text, or `None`.
#[pyclass(frozen)]
pub(crate) struct RustKeyRepository(Arc<KeyRepository>);

#[pymethods]
impl RustKeyRepository {
    #[new]
    fn new(database: &NativeDatabase) -> PyResult<Self> {
        Ok(Self(Arc::new(KeyRepository::new(
            database.repository_database()?,
        ))))
    }

    #[pyo3(signature = (token, *, use_writer=false))]
    fn find<'py>(
        &self,
        py: Python<'py>,
        token: String,
        use_writer: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let repository = Arc::clone(&self.0);
        let side = super::side(use_writer);
        crate::execution::run_async_value(
            py,
            crate::db::for_caller(py, async move {
                let view = repository.find(&token, side).await.map_err(map_error)?;
                super::json(view.as_ref())
            })?,
        )
    }
}
