use litellm_host_python::ready_future;
use pyo3::{prelude::*, types::PyNone};

#[pyclass(frozen)]
pub(crate) struct NativeDatabase {}

// TODO: take typed writer/reader settings and a token provider, then open a sqlx PgPool, once the first
// repository moves to native. The Prisma DATABASE_URL carries Prisma-only params and rotating IAM tokens
#[pymethods]
impl NativeDatabase {
    #[staticmethod]
    fn connect(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        ready_future(py, Bound::new(py, NativeDatabase {})?.as_any())
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        ready_future(py, PyNone::get(py).as_any())
    }
}
