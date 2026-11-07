mod fetch;
mod live;
mod rows;

use std::borrow::Cow;
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use litellm_host_python::{from_py, to_py};
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{CostMap, ErrorKind, IntegrityLimits, Provenance, Store};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub(crate) use fetch::{NativeCostMapSource, NativeFetchFailed, NativeFetchPolicy, NativeFetched};
use live::LiveDict;

/// The cost map's inputs, the installed document and the ordered `register_model` overlay, with the name index over
/// the dict Python serves as `litellm.model_cost`. Python owns that dict and passes it to every call that reads or
/// writes it, so this never keeps a second copy of the live rows. No lock is held while Python code runs, since that
/// code may release the GIL to a thread that then waits on the same lock
#[pyclass(
    frozen,
    name = "NativeModelCostMap",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeModelCostMap {
    inputs: RwLock<CostMap>,
    limits: IntegrityLimits,
    live: RwLock<Option<Arc<Store<LiveDict>>>>,
}

#[pyclass(
    frozen,
    name = "NativeInstalled",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeInstalled {
    /// The new live table, the document with every registration applied
    #[pyo3(get)]
    rows: Py<PyDict>,
    /// How many keys the document itself serves, aliases included and registrations left out
    #[pyo3(get)]
    model_count: usize,
    /// The document's `fallback_generalizations` block as written, `None` when it has none
    #[pyo3(get)]
    fallback_generalizations: Py<PyAny>,
}

#[pyclass(
    frozen,
    name = "NativeRejected",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeRejected {
    #[pyo3(get)]
    reason: String,
    /// The document parsed but has fewer models than the integrity limits allow
    #[pyo3(get)]
    integrity: bool,
}

/// The inputs at one moment, for a caller that has to put them back, such as test isolation
#[pyclass(
    frozen,
    name = "NativeModelCostMapState",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeModelCostMapState(CostMap);

#[pyclass(
    frozen,
    name = "NativeProvenance",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeProvenance {
    #[pyo3(get)]
    revision: String,
    #[pyo3(get)]
    etag: Option<String>,
}

#[pymethods]
impl NativeModelCostMap {
    /// Starts from the bundled document. A fetched document with fewer than `min_model_count` models, or fewer than
    /// `max_shrink_ratio` times the bundled count, is rejected
    #[staticmethod]
    fn bundled(min_model_count: usize, max_shrink_ratio: f64) -> PyResult<Self> {
        let limits = IntegrityLimits::new(min_model_count, max_shrink_ratio)
            .ok_or_else(|| PyValueError::new_err("max_shrink_ratio must be between 0 and 1"))?;
        Ok(Self {
            inputs: RwLock::new(CostMap::bundled()),
            limits,
            live: RwLock::new(None),
        })
    }

    #[pyo3(signature = (source, policy))]
    fn fetch<'py>(
        &self,
        py: Python<'py>,
        source: PyRef<'_, NativeCostMapSource>,
        policy: PyRef<'_, NativeFetchPolicy>,
    ) -> PyResult<Bound<'py, PyAny>> {
        fetch::fetch(py, &source, &policy)
    }

    #[pyo3(signature = (source, policy, resume_from = None))]
    fn fetch_blocking(
        &self,
        py: Python<'_>,
        source: PyRef<'_, NativeCostMapSource>,
        policy: PyRef<'_, NativeFetchPolicy>,
        resume_from: Option<PyRef<'_, NativeFetchFailed>>,
    ) -> PyResult<Py<PyAny>> {
        fetch::fetch_blocking(py, &source, &policy, resume_from.as_deref())
    }

    /// Accepts a fetched document and returns the live table built from it with every registration applied, or why
    /// it was rejected, in which case nothing changed
    #[pyo3(signature = (body, etag = None))]
    fn install(&self, py: Python<'_>, body: &[u8], etag: Option<String>) -> PyResult<Py<PyAny>> {
        let accepted = {
            let mut inputs = self.write_inputs();
            let accepted = inputs.accept(body, etag, self.limits).map(|_| ());
            accepted.map(|()| inputs.clone())
        };
        match accepted {
            Ok(inputs) => self.installed(py, &inputs),
            Err(error) => {
                let integrity = matches!(
                    error.kind(),
                    ErrorKind::Empty | ErrorKind::BelowMinimum | ErrorKind::Shrunk
                );
                let reason = litellm_error::Report::new(&error).to_string();
                Py::new(py, NativeRejected { reason, integrity }).map(Py::into_any)
            }
        }
    }

    /// Installs the bundled document, which is never size checked
    fn install_bundled(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inputs = {
            let mut inputs = self.write_inputs();
            inputs.accept_bundled();
            inputs.clone()
        };
        self.installed(py, &inputs)
    }

    /// Writes each entry into `rows` as main's `register_model` does and returns the row keys they landed on. With
    /// `persist`, the entries are also kept and written again over every later install
    #[pyo3(signature = (rows, entries, persist, custom_llm_provider = None, warning_display_name = None))]
    fn register(
        &self,
        rows: &Bound<'_, PyDict>,
        entries: &Bound<'_, PyDict>,
        persist: bool,
        custom_llm_provider: Option<&str>,
        warning_display_name: Option<&str>,
    ) -> PyResult<Vec<String>> {
        let entries: Vec<(String, Bound<'_, PyAny>)> = entries
            .iter()
            .map(|(key, value)| Ok((key.str()?.to_string(), value)))
            .collect::<PyResult<_>>()?;
        if persist {
            let parsed = entries
                .iter()
                .map(|(key, value)| Ok((key, from_py::<Fields>(value)?)))
                .collect::<PyResult<Vec<_>>>()?;
            let mut inputs = self.write_inputs();
            for (key, fields) in parsed {
                inputs.register(key, &fields);
            }
        }
        entries
            .iter()
            .map(|(key, value)| {
                self.write(
                    rows,
                    key,
                    value,
                    persist,
                    custom_llm_provider,
                    warning_display_name,
                )
            })
            .collect()
    }

    /// Writes every kept registration into `rows` again, oldest first, so it wins the fields it shares with rows the
    /// router replayed after an install
    fn reapply(&self, rows: &Bound<'_, PyDict>) -> PyResult<()> {
        let py = rows.py();
        let registrations: Vec<(String, Fields)> = self
            .read_inputs()
            .registrations()
            .entries()
            .map(|(key, fields)| (key.to_owned(), fields.clone()))
            .collect();
        for (key, fields) in registrations {
            self.write(
                rows,
                &key,
                &to_py(py, &fields)?.into_bound(py),
                true,
                None,
                None,
            )?;
        }
        Ok(())
    }

    /// The installed row under `key` with the kept registrations that apply to it, without anything the router wrote
    fn registered_row(&self, py: Python<'_>, key: &str) -> PyResult<Option<Py<PyAny>>> {
        let row = self.read_inputs().registered_row(key).map(Cow::into_owned);
        row.map(|row| to_py(py, &row)).transpose()
    }

    fn registered_keys(&self) -> Vec<String> {
        self.read_inputs()
            .registered_keys()
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    /// The installed document as a table, without any registration
    fn loaded<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let inputs = self.read_inputs().clone();
        rows::table(py, &inputs.loaded())
    }

    /// The key itself when `rows` has it, otherwise the key that differs from `name` only in case
    fn find_key(&self, rows: &Bound<'_, PyDict>, name: &str) -> Option<String> {
        self.with_store(rows, |store| store.find_key(name))
    }

    /// Keys whose row's `litellm_provider` is exactly `provider`, in row order
    fn keys_in(&self, rows: &Bound<'_, PyDict>, provider: &str) -> Vec<String> {
        self.with_store(rows, |store| store.keys_in(provider))
    }

    /// Drops the name index, for a caller that changed keys in a way the dict's length does not show
    fn invalidate(&self) {
        if let Some(store) = self
            .live
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
        {
            store.invalidate();
        }
    }

    fn snapshot(&self) -> NativeModelCostMapState {
        NativeModelCostMapState(self.read_inputs().clone())
    }

    /// Puts back the document, registrations and provenance a `snapshot` saw
    fn restore(&self, state: PyRef<'_, NativeModelCostMapState>) {
        *self.write_inputs() = state.0.clone();
        self.invalidate();
    }

    #[getter]
    fn provenance(&self) -> NativeProvenance {
        let Provenance { revision, etag } = self.read_inputs().provenance().clone();
        NativeProvenance { revision, etag }
    }
}

impl NativeModelCostMap {
    fn read_inputs(&self) -> RwLockReadGuard<'_, CostMap> {
        self.inputs.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write_inputs(&self) -> RwLockWriteGuard<'_, CostMap> {
        self.inputs.write().unwrap_or_else(PoisonError::into_inner)
    }

    fn installed(&self, py: Python<'_>, inputs: &CostMap) -> PyResult<Py<PyAny>> {
        let rows = rows::table(py, &inputs.materialize())?;
        let model_count = inputs.loaded().len();
        let fallback_generalizations = match inputs.fallback_generalizations() {
            Some(block) => to_py(py, block)?,
            None => py.None(),
        };
        *self.live.write().unwrap_or_else(PoisonError::into_inner) =
            Some(Arc::new(Store::new(LiveDict::new(&rows))));
        Py::new(
            py,
            NativeInstalled {
                rows: rows.unbind(),
                model_count,
                fallback_generalizations,
            },
        )
        .map(Py::into_any)
    }

    /// One registration through the host's write, with no lock held so the write may look names up through this map
    fn write(
        &self,
        rows: &Bound<'_, PyDict>,
        key: &str,
        value: &Bound<'_, PyAny>,
        persist: bool,
        custom_llm_provider: Option<&str>,
        warning_display_name: Option<&str>,
    ) -> PyResult<String> {
        let written = rows::write(rows, key, value, custom_llm_provider, warning_display_name)?;
        if persist {
            self.write_inputs().landed(&written, key);
        }
        self.invalidate();
        Ok(written)
    }

    /// Queries the index over `rows`, rebinding it when `rows` is not the dict it was built over
    fn with_store<T>(
        &self,
        rows: &Bound<'_, PyDict>,
        query: impl FnOnce(&Store<LiveDict>) -> T,
    ) -> T {
        let bound = self
            .live
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .filter(|store| store.rows().is(rows));
        let store = bound.unwrap_or_else(|| {
            let store = Arc::new(Store::new(LiveDict::new(rows)));
            *self.live.write().unwrap_or_else(PoisonError::into_inner) = Some(Arc::clone(&store));
            store
        });
        query(&store)
    }
}
