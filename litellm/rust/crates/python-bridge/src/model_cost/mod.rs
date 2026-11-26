mod fetch;
mod host;
mod live;
mod snapshot;
mod table;

use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use litellm_cost::adjust::CostConfig;
use litellm_host_python::{from_py_argument, to_py};
use litellm_model_catalog::{
    CapabilityRules, CostMap, FallbackRule, IntegrityLimits, Landed, Pass, Placed, PricingContext,
    Provenance, RegistrationTable, Registrations, Store, write_pass,
};
use litellm_model_catalog_refresh::{Refreshed, refresh};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMapping};

use crate::cost::{self, NativeRateCard};
use litellm_callbacks_legacy_python::NativeReceipt;

pub(crate) use fetch::{NativeCostMapSource, NativeFetchFailed, NativeFetchPolicy, NativeFetched};
use live::LiveDict;
pub(crate) use live::row_fields;
pub(crate) use snapshot::CostSnapshot;
use snapshot::Snapshots;

/// The installed document and the `register_model` calls kept above it, each as the dict Python merged
struct Inputs {
    cost_map: CostMap,
    registrations: Registrations<Py<PyDict>>,
}

impl Inputs {
    fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            cost_map: self.cost_map.clone(),
            registrations: self.registrations.map(|kept| kept.clone_ref(py)),
        }
    }
}

/// The capability rules of the host's `fallback_generalizations` list. A rule that does not read as one is skipped,
/// as the host skips it
fn capability_rules(rules: &Bound<'_, PyAny>) -> CapabilityRules {
    let read: Vec<FallbackRule> = rules
        .try_iter()
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|rule| from_py_argument(&rule).ok())
        .collect();
    CapabilityRules::new(&read)
}

/// The pricing a Messages route call reads, over the snapshot of the published table taken as the call is set up
pub(crate) fn route_pricing(py: Python<'_>) -> PyResult<cost::RoutePricing> {
    Ok(cost::RoutePricing::new(
        host::inputs(py)?.get().cost_snapshot(py)?,
    ))
}

/// The cost map's inputs, the name index over `litellm.model_cost` and its pricing snapshots
#[pyclass(
    frozen,
    name = "NativeModelCostMap",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeModelCostMap {
    inputs: RwLock<Inputs>,
    limits: IntegrityLimits,
    index: RwLock<Option<Arc<Store<LiveDict>>>>,
    snapshots: Snapshots,
}

#[pyclass(
    frozen,
    name = "NativeInstalled",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeInstalled {
    /// The new live table, the document without registrations
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

/// A fetched document that passed its checks and is installed nowhere yet
#[pyclass(
    frozen,
    name = "CostMapRefreshed",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct CostMapRefreshed(CostMap);

/// A refresh that installed nothing, with why in the words every host reports
#[pyclass(frozen, name = "CostMapKept", module = "litellm.rust_bridge._native")]
pub(crate) struct CostMapKept {
    /// The cause as the failure itself tells it, for a log line
    #[pyo3(get)]
    detail: String,
    /// What a reload answers its caller
    #[pyo3(get)]
    reason: String,
    /// What the source info shows while the bundled or last document stays
    #[pyo3(get)]
    fallback_reason: String,
    /// Another fetch could succeed. Only the attempt limit ended this one
    #[pyo3(get)]
    retryable: bool,
}

/// Ends a blocking refresh from another thread. A daemon thread that comes back from native code once the
/// interpreter is finalizing aborts the process, so interpreter exit calls `stop`, which returns only after the
/// refresh has left native code
#[pyclass(
    frozen,
    name = "CostMapRefreshStop",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct CostMapRefreshStop {
    stopped: tokio::sync::watch::Sender<bool>,
    /// Read by a refresh for as long as it is in native code, and written by `stop` to wait for it
    in_native: Arc<parking_lot::RwLock<()>>,
}

#[pymethods]
impl CostMapRefreshStop {
    #[new]
    fn new() -> Self {
        Self {
            stopped: tokio::sync::watch::Sender::new(false),
            in_native: Arc::default(),
        }
    }

    fn stop(&self, py: Python<'_>) {
        self.stopped.send_replace(true);
        let in_native = self.in_native.clone();
        litellm_host_python::release_gil(py, move || drop(in_native.write()));
    }
}

/// A finished registration pass, with the registrations no built-in row matched, by registered key
#[pyclass(
    frozen,
    name = "NativeRegistered",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeRegistered {
    #[pyo3(get)]
    unmatched: Py<PyDict>,
}

/// A registration pass that stopped because the table it was handed is no longer the published one
#[pyclass(
    frozen,
    name = "NativeTableSwapped",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeTableSwapped;

/// The inputs at one moment, for a caller that has to put them back, such as test isolation
#[pyclass(
    frozen,
    name = "NativeModelCostMapState",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeModelCostMapState(Inputs);

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
    /// `max_shrink_ratio` times the bundled count, is rejected. A limit left out is `IntegrityLimits::DEFAULT`'s
    #[staticmethod]
    #[pyo3(signature = (min_model_count = None, max_shrink_ratio = None))]
    fn bundled(min_model_count: Option<i64>, max_shrink_ratio: Option<f64>) -> PyResult<Self> {
        let min_model_count = min_model_count
            .map(|count| {
                usize::try_from(count).map_err(|_| {
                    PyValueError::new_err(format!("min_model_count must be 0 or more, got {count}"))
                })
            })
            .transpose()?;
        let limits =
            IntegrityLimits::over_default(min_model_count, max_shrink_ratio).map_err(|ratio| {
                PyValueError::new_err(format!(
                    "max_shrink_ratio must be between 0 and 1, got {ratio}"
                ))
            })?;
        Ok(Self {
            inputs: RwLock::new(Inputs {
                cost_map: CostMap::bundled(),
                registrations: Registrations::default(),
            }),
            limits,
            index: RwLock::new(None),
            snapshots: Snapshots::default(),
        })
    }

    /// The live table built from a fetched document, or why it was rejected, in which case nothing changed
    #[pyo3(signature = (body, etag = None))]
    fn install(&self, py: Python<'_>, body: &[u8], etag: Option<String>) -> PyResult<Py<PyAny>> {
        match crate::logger::capture(py).scope(|| CostMap::read(body, etag, self.limits)) {
            Ok(cost_map) => self.install_document(py, cost_map),
            Err(error) => rejected(py, &error),
        }
    }

    #[getter]
    fn min_model_count(&self) -> usize {
        self.limits.min_model_count()
    }

    #[getter]
    fn max_shrink_ratio(&self) -> f64 {
        self.limits.min_reference_ratio()
    }

    /// Fetches and reads the document on a throwaway runtime, off the GIL, and installs nothing. The result is a
    /// `CostMapRefreshed` or a `CostMapKept`, or `None` once `stop` was called
    fn refresh_blocking(
        &self,
        py: Python<'_>,
        source: PyRef<'_, NativeCostMapSource>,
        policy: PyRef<'_, NativeFetchPolicy>,
        stop: PyRef<'_, CostMapRefreshStop>,
    ) -> PyResult<Py<PyAny>> {
        let _in_native = stop.in_native.read();
        if *stop.stopped.borrow() {
            return Ok(py.None());
        }
        let mut stopped = stop.stopped.subscribe();
        let (connection, policy, limits) = (source.0.clone(), policy.0, self.limits);
        let url = connection.url().to_owned();
        let logger = crate::logger::capture(py);
        let refreshed = litellm_host_python::release_gil(py, move || {
            logger.scope(|| {
                let source = connection.unpooled(policy)?;
                litellm_host_python::run_isolated(async {
                    tokio::select! {
                        refreshed = refresh(&source, limits) => Some(refreshed),
                        _ = stopped.wait_for(|stopped| *stopped) => None,
                    }
                })
                .map_err(PyErr::from)
            })
        })?;
        match refreshed {
            Some(refreshed) => Outcome { refreshed, url }
                .into_pyobject(py)
                .map(Bound::unbind),
            None => Ok(py.None()),
        }
    }

    /// `refresh_blocking` on the shared runtime with a pooled client, for workers after the fork
    fn refresh<'py>(
        &self,
        py: Python<'py>,
        source: PyRef<'_, NativeCostMapSource>,
        policy: PyRef<'_, NativeFetchPolicy>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let url = source.0.url().to_owned();
        let (source, limits) = (source.0.pooled(policy.0)?, self.limits);
        crate::execution::run_async_value(py, async move {
            Ok(Outcome {
                refreshed: refresh(&source, limits).await,
                url,
            })
        })
    }

    fn install_refreshed(
        &self,
        py: Python<'_>,
        refreshed: PyRef<'_, CostMapRefreshed>,
    ) -> PyResult<Py<PyAny>> {
        self.install_document(py, refreshed.0.clone())
    }

    /// A caller's document as a table, read without size limits and installed nowhere
    #[staticmethod]
    fn finalize(py: Python<'_>, body: &[u8]) -> PyResult<Py<PyAny>> {
        match crate::logger::capture(py)
            .scope(|| CostMap::read(body, None, IntegrityLimits::UNCHECKED))
        {
            Ok(cost_map) => Ok(Py::new(py, built(py, &cost_map)?)?.into_any()),
            Err(error) => rejected(py, &error),
        }
    }

    /// The live table built from the bundled document, which is never size checked
    fn install_bundled(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.install_document(py, CostMap::bundled())
    }

    /// Writes each entry into `rows`, or into the published table when `rows` is `None`
    #[pyo3(signature = (rows, entries, persist, custom_llm_provider = None))]
    fn register(
        &self,
        py: Python<'_>,
        rows: Option<&Bound<'_, PyDict>>,
        entries: &Bound<'_, PyDict>,
        persist: bool,
        custom_llm_provider: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let entries: Vec<(String, Bound<'_, PyAny>)> = entries
            .iter()
            .map(|(key, registration)| Ok((key.str()?.to_string(), registration)))
            .collect::<PyResult<_>>()?;
        let kept = match persist {
            true => self.merged_with_kept(&entries)?,
            false => Vec::new(),
        };
        let pass = self.write_pass(rows, &entries, custom_llm_provider)?;
        if persist {
            self.keep(kept, pass.landed());
        }
        into_py(py, pass)
    }

    /// Writes every kept registration into `rows` again, oldest first
    fn reapply(&self, rows: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
        let py = rows.py();
        let entries: Vec<(String, Bound<'_, PyAny>)> = self
            .read_inputs()
            .registrations
            .entries()
            .map(|(key, kept)| (key.to_owned(), kept.bind(py).clone().into_any()))
            .collect();
        let pass = self.write_pass(Some(rows), &entries, None)?;
        self.write_inputs().registrations.record(pass.landed());
        into_py(py, pass)
    }

    /// The installed row under `key` with the kept registrations that apply to it, without anything the router wrote
    fn registered_row<'py>(
        &self,
        py: Python<'py>,
        key: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let (cost_map, applied) = {
            let inputs = self.read_inputs();
            let applied: Vec<Bound<'py, PyDict>> = inputs
                .registrations
                .over(key)
                .into_iter()
                .map(|kept| kept.bind(py).clone())
                .collect();
            (inputs.cost_map.clone(), applied)
        };
        table::registered_row(py, &cost_map, key, &applied)
    }

    fn registered_keys(&self) -> Vec<String> {
        let inputs = self.read_inputs();
        table::registered_keys(
            &inputs.cost_map,
            inputs.registrations.entries().map(|(key, _)| key),
        )
    }

    /// The installed document as a table, without any registration
    fn loaded<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let cost_map = self.read_inputs().cost_map.clone();
        table::build(py, &cost_map)
    }

    /// The key itself when `rows` has it, otherwise the key that differs from `name` only in case
    fn find_key(&self, rows: &Bound<'_, PyDict>, name: &str) -> Option<String> {
        self.index_over(rows).find_key(name)
    }

    /// Keys whose row's `litellm_provider` is exactly `provider`, in row order
    fn keys_in(&self, rows: &Bound<'_, PyDict>, provider: &str) -> Vec<String> {
        self.index_over(rows).keys_in(provider)
    }

    /// Drops the name index and the pricing snapshot, for a caller that changed keys in a way the dict's length does
    /// not show
    fn invalidate(&self) {
        self.snapshots.forget();
        if let Some(index) = self
            .index
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
        {
            index.invalidate();
        }
    }

    fn snapshot(&self, py: Python<'_>) -> NativeModelCostMapState {
        NativeModelCostMapState(self.read_inputs().clone_ref(py))
    }

    /// Puts back the document, registrations and provenance a `snapshot` saw
    fn restore(&self, py: Python<'_>, state: PyRef<'_, NativeModelCostMapState>) {
        *self.write_inputs() = state.0.clone_ref(py);
        self.invalidate();
    }

    #[getter]
    fn provenance(&self) -> NativeProvenance {
        let Provenance { revision, etag } = self.read_inputs().cost_map.provenance().clone();
        NativeProvenance { revision, etag }
    }

    /// Prices a token call from the facts in `document`, `None` when Python prices it the old way. `row` is a price
    /// row the caller passes for this call alone
    #[pyo3(signature = (document, row=None))]
    fn price(
        &self,
        py: Python<'_>,
        document: &Bound<'_, PyAny>,
        row: Option<&Bound<'_, PyMapping>>,
    ) -> PyResult<Option<NativeReceipt>> {
        self.cost_snapshot(py)?
            .lookup(|lookup| cost::price(py, lookup, document, row))
    }

    /// The legacy budget hold and the input cost floor of a request. `rows` are the model info mappings its shapes
    /// read by index
    fn bound(
        &self,
        py: Python<'_>,
        document: &Bound<'_, PyAny>,
        rows: Vec<Bound<'_, PyMapping>>,
    ) -> PyResult<(Option<f64>, Option<f64>)> {
        self.cost_snapshot(py)?
            .lookup(|lookup| cost::bound(lookup, document, &rows))
    }

    /// The rate card of the row under exactly `key`
    fn rates(&self, py: Python<'_>, key: &str) -> PyResult<Option<NativeRateCard>> {
        Ok(self
            .cost_snapshot(py)?
            .lookup(|lookup| cost::rates(lookup, key)))
    }

    /// `cost` after the configured discount and margin
    #[pyo3(signature = (cost, provider, model=None, region=None))]
    fn adjust(
        &self,
        py: Python<'_>,
        cost: f64,
        provider: Option<&str>,
        model: Option<&str>,
        region: Option<&str>,
    ) -> PyResult<(f64, f64, f64, f64, f64, f64)> {
        Ok(cost::adjust(
            self.cost_snapshot(py)?.config(),
            cost,
            provider,
            model,
            region,
        ))
    }
}

impl NativeModelCostMap {
    /// The pricing snapshot of the published table at its current generation, built only when the generation moved
    pub(crate) fn cost_snapshot(&self, py: Python<'_>) -> PyResult<Arc<CostSnapshot>> {
        let read = host::generation(py)?;
        let (discounts, margins) = cost::adjustment_config(&read.discounts, &read.margins)?;
        let config = CostConfig { discounts, margins };
        self.snapshots.at(read.generation, config, |config| {
            Ok((
                self.index_over(&read.rows),
                PricingContext {
                    rules: capability_rules(&read.rules),
                    names: host::known_names(py)?,
                    config: Arc::new(config),
                },
            ))
        })
    }

    fn read_inputs(&self) -> RwLockReadGuard<'_, Inputs> {
        self.inputs.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write_inputs(&self) -> RwLockWriteGuard<'_, Inputs> {
        self.inputs.write().unwrap_or_else(PoisonError::into_inner)
    }

    fn install_document(&self, py: Python<'_>, cost_map: CostMap) -> PyResult<Py<PyAny>> {
        let installed = built(py, &cost_map)?;
        self.write_inputs().cost_map = cost_map;
        *self.index.write().unwrap_or_else(PoisonError::into_inner) =
            Some(Arc::new(Store::new(LiveDict::new(installed.rows.bind(py)))));
        Ok(Py::new(py, installed)?.into_any())
    }

    /// Each entry merged into the registration kept for its key
    fn merged_with_kept<'py>(
        &self,
        entries: &[(String, Bound<'py, PyAny>)],
    ) -> PyResult<Vec<(String, Bound<'py, PyDict>)>> {
        entries
            .iter()
            .map(|(key, registration)| {
                let previous = self
                    .read_inputs()
                    .registrations
                    .get(key)
                    .map(|kept| kept.bind(registration.py()).clone());
                Ok((key.clone(), host::kept(previous.as_ref(), registration)?))
            })
            .collect()
    }

    fn keep(&self, kept: Vec<(String, Bound<'_, PyDict>)>, landed: &[Landed]) {
        let mut inputs = self.write_inputs();
        for (key, registration) in kept {
            inputs.registrations.register(&key, registration.unbind());
        }
        inputs.registrations.record(landed);
    }

    fn write_pass<'a, 'py>(
        &self,
        rows: Option<&Bound<'py, PyDict>>,
        entries: &'a [(String, Bound<'py, PyAny>)],
        custom_llm_provider: Option<&str>,
    ) -> PyResult<Pass<'a, Bound<'py, PyAny>>> {
        let mut table = PublishedTable {
            map: self,
            rows,
            custom_llm_provider,
        };
        write_pass(&mut table, entries)
    }

    /// The index over `rows`, rebuilt when `rows` is not the dict it was built over
    fn index_over(&self, rows: &Bound<'_, PyDict>) -> Arc<Store<LiveDict>> {
        let current = self
            .index
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        if let Some(index) = current.filter(|index| index.rows().is(rows)) {
            return index;
        }
        let index = Arc::new(Store::new(LiveDict::new(rows)));
        *self.index.write().unwrap_or_else(PoisonError::into_inner) = Some(Arc::clone(&index));
        index
    }
}

/// The dict Python publishes as `litellm.model_cost`, written through the host
struct PublishedTable<'a, 'py> {
    map: &'a NativeModelCostMap,
    rows: Option<&'a Bound<'py, PyDict>>,
    custom_llm_provider: Option<&'a str>,
}

impl<'py> RegistrationTable for PublishedTable<'_, 'py> {
    type Registration = Bound<'py, PyAny>;
    type Error = PyErr;

    fn place(&mut self, key: &str, registration: &Self::Registration) -> PyResult<Option<Placed>> {
        let Some(landing) = host::resolve(self.rows, key, registration, self.custom_llm_provider)?
        else {
            return Ok(None);
        };
        let rows = match self.rows {
            Some(rows) => rows.clone(),
            None => host::published(registration.py())?,
        };
        table::write(&rows, &landing, registration)?;
        self.map.invalidate();
        Ok(Some(Placed {
            key: landing.key,
            matched_builtin: landing.matched_builtin,
        }))
    }
}

fn built(py: Python<'_>, cost_map: &CostMap) -> PyResult<NativeInstalled> {
    let rows = table::build(py, cost_map)?;
    let fallback_generalizations = match cost_map.fallback_generalizations() {
        Some(block) => to_py(py, block)?,
        None => py.None(),
    };
    Ok(NativeInstalled {
        model_count: rows.len(),
        rows: rows.unbind(),
        fallback_generalizations,
    })
}

struct Outcome {
    refreshed: Refreshed,
    url: String,
}

impl<'py> IntoPyObject<'py> for Outcome {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        match self.refreshed.or_kept(&self.url) {
            Ok(cost_map) => Bound::new(py, CostMapRefreshed(cost_map)).map(Bound::into_any),
            Err(kept) => Bound::new(
                py,
                CostMapKept {
                    detail: kept.detail,
                    reason: kept.reason,
                    fallback_reason: kept.fallback_reason,
                    retryable: kept.retryable,
                },
            )
            .map(Bound::into_any),
        }
    }
}

fn rejected(py: Python<'_>, error: &litellm_model_catalog::Error) -> PyResult<Py<PyAny>> {
    let rejected = NativeRejected {
        reason: litellm_error::Report::new(error).to_string(),
        integrity: error.is_integrity(),
    };
    Ok(Py::new(py, rejected)?.into_any())
}

fn into_py<'py>(py: Python<'py>, pass: Pass<'_, Bound<'py, PyAny>>) -> PyResult<Py<PyAny>> {
    match pass {
        Pass::Swapped { .. } => Ok(Py::new(py, NativeTableSwapped)?.into_any()),
        Pass::Finished { unmatched, .. } => {
            let table = PyDict::new(py);
            for (key, registration) in unmatched {
                table.set_item(key, registration)?;
            }
            let registered = NativeRegistered {
                unmatched: table.unbind(),
            };
            Ok(Py::new(py, registered)?.into_any())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::time::Duration;

    use pyo3::ffi::c_str;
    use rstest::rstest;

    use super::*;

    /// Held by every test that installs a fake host module, since Python can switch threads while one runs
    static HOST_MODULE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const HOST: &std::ffi::CStr = c_str!(
        r#"
from types import SimpleNamespace

class Row(dict):
    names = ()

found = []

def resolve(rows, key, registration, custom_llm_provider):
    found.append(cost_map.find_key(rows, key))
    return SimpleNamespace(key=key, base={}, matched_builtin=True)

def merged(current, base, registration):
    return Row({**(current or {}), **base, **registration})

def write(rows, written):
    rows.update(written)
"#
    );

    #[rstest]
    fn a_host_lookup_that_reaches_find_key_on_the_same_map_finishes() {
        let _host = HOST_MODULE.lock().unwrap_or_else(PoisonError::into_inner);
        Python::initialize();
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            let outcome = Python::attach(|py| -> PyResult<(Vec<Option<String>>, f64)> {
                let cost_map = Py::new(py, NativeModelCostMap::bundled(Some(0), Some(0.0))?)?;
                let module_name = std::ffi::CString::new(host::MODULE).unwrap();
                let host =
                    PyModule::from_code(py, HOST, c_str!("host_model_cost.py"), &module_name)?;
                host.setattr("cost_map", &cost_map)?;
                let modules = py.import("sys")?.getattr("modules")?;
                let packages = ["litellm", "litellm.rust_bridge", "litellm.rust_bridge.host"];
                for package in packages {
                    modules.set_item(package, PyModule::new(py, package)?)?;
                }
                modules.set_item(host::MODULE, &host)?;
                let rows = PyDict::new(py);
                rows.set_item("Lit-Reentry", PyDict::new(py))?;
                let entry = PyDict::new(py);
                entry.set_item("input_cost_per_token", 1.5)?;
                let entries = PyDict::new(py);
                entries.set_item("lit-reentry", entry)?;
                let registered = cost_map
                    .bind(py)
                    .call_method1("register", (&rows, &entries, false));
                for name in [host::MODULE].into_iter().chain(packages) {
                    modules.del_item(name)?;
                }
                registered?;
                let price = rows
                    .get_item("lit-reentry")?
                    .expect("the registration landed")
                    .get_item("input_cost_per_token")?
                    .extract()?;
                Ok((host.getattr("found")?.extract()?, price))
            });
            sender
                .send(outcome.map_err(|error| error.to_string()))
                .unwrap();
        });

        let (found, price) = receiver
            .recv_timeout(Duration::from_secs(30))
            .expect("register finished instead of deadlocking on its own lock")
            .unwrap();
        assert_eq!(found, [Some("Lit-Reentry".to_owned())]);
        assert!((price - 1.5).abs() < f64::EPSILON);
    }

    const PRICING_HOST: &std::ffi::CStr = c_str!(
        r#"
class Row(dict):
    registered_fields = frozenset()

rows = {"claude-opus-5-5": Row(litellm_provider="anthropic", input_cost_per_token=1e-6)}
writes = [0]

def generation():
    return rows, (id(rows), len(rows), writes[0], 0), ({}, {}, [])

def known_names():
    return frozenset({"anthropic"}), frozenset()

def write(key, row):
    rows[key] = row
    writes[0] += 1
"#
    );

    /// The input rate of the first row a call to `claude-opus-5-5` on Anthropic tries, read in one call's view
    fn priced_input_rate(snapshot: &CostSnapshot) -> Option<f64> {
        let call = litellm_cost::candidates::CallIdentity {
            provider: "anthropic".to_owned(),
            request_model: Some("claude-opus-5-5".to_owned()),
            response_model: Some("claude-opus-5-5".to_owned()),
            ..Default::default()
        };
        snapshot.lookup(|lookup| {
            let candidates = litellm_cost::candidates::candidates(lookup, &call);
            let row = lookup.row(&candidates.rows.first()?.row)?;
            row.lowered.fields["input_cost_per_token"].as_f64()
        })
    }

    /// Runs `priced` with `PRICING_HOST` installed as the host module, and removes it after
    fn with_pricing_host<T>(
        priced: impl FnOnce(Python<'_>, &NativeModelCostMap, &Bound<'_, PyModule>) -> PyResult<T>,
    ) -> PyResult<T> {
        let _host = HOST_MODULE.lock().unwrap_or_else(PoisonError::into_inner);
        Python::initialize();
        Python::attach(|py| {
            let cost_map = NativeModelCostMap::bundled(Some(0), Some(0.0))?;
            let module_name = std::ffi::CString::new(host::MODULE).unwrap();
            let host =
                PyModule::from_code(py, PRICING_HOST, c_str!("host_model_cost.py"), &module_name)?;
            let modules = py.import("sys")?.getattr("modules")?;
            let packages = ["litellm", "litellm.rust_bridge", "litellm.rust_bridge.host"];
            for package in packages {
                modules.set_item(package, PyModule::new(py, package)?)?;
            }
            modules.set_item(host::MODULE, &host)?;
            let outcome = priced(py, &cost_map, &host);
            for name in [host::MODULE].into_iter().chain(packages) {
                modules.del_item(name)?;
            }
            outcome
        })
    }

    #[rstest]
    fn a_cost_snapshot_prices_the_row_python_holds_at_each_generation() {
        let (before, after, unchanged) = with_pricing_host(|py, cost_map, host| {
            let before = cost_map.cost_snapshot(py)?;
            let rate_before = priced_input_rate(&before);
            let row = host.getattr("Row")?.call(
                (),
                Some(&pyo3::types::IntoPyDict::into_py_dict(
                    [("litellm_provider", "anthropic")],
                    py,
                )?),
            )?;
            row.set_item("input_cost_per_token", 2e-6)?;
            let unchanged = Arc::ptr_eq(&before, &cost_map.cost_snapshot(py)?);
            host.getattr("write")?.call1(("claude-opus-5-5", row))?;
            let after = cost_map.cost_snapshot(py)?;
            Ok((rate_before, priced_input_rate(&after), unchanged))
        })
        .unwrap();

        assert!(unchanged);
        assert_eq!(before, Some(1e-6));
        assert_eq!(after, Some(2e-6));
    }

    /// Python code can edit a row's price in place, which moves no generation. The next call prices the new rate
    #[rstest]
    fn a_price_edited_in_place_is_priced_on_the_next_call() {
        let (before, after, same_snapshot) = with_pricing_host(|py, cost_map, host| {
            let before = cost_map.cost_snapshot(py)?;
            let rate_before = priced_input_rate(&before);
            host.getattr("rows")?
                .get_item("claude-opus-5-5")?
                .set_item("input_cost_per_token", 3e-6)?;
            let after = cost_map.cost_snapshot(py)?;
            Ok((
                rate_before,
                priced_input_rate(&after),
                Arc::ptr_eq(&before, &after),
            ))
        })
        .unwrap();

        assert!(same_snapshot);
        assert_eq!(before, Some(1e-6));
        assert_eq!(after, Some(3e-6));
    }

    /// The total a route call to `claude-opus-5-5` on Anthropic costs for 1000 prompt tokens
    fn route_total(pricing: &cost::RoutePricing) -> Option<f64> {
        let usage = litellm_cost::query::TokenUsage {
            prompt_tokens: 1000,
            input: litellm_cost::query::Modalities {
                text: 1000,
                ..Default::default()
            },
            ..Default::default()
        };
        litellm_inference_messages::Pricing::price(
            pricing,
            &litellm_inference_messages::PricedCall {
                provider: "anthropic".into(),
                request_model: "claude-opus-5-5".into(),
                response_model: Some("claude-opus-5-5".into()),
                identity: litellm_inference_messages::PricingIdentity::default(),
                usage,
                service_tier: None,
            },
        )
        .map(|receipt| receipt.total)
    }

    #[rstest]
    fn a_route_call_prices_over_the_row_the_table_holds_when_it_finishes_as_main_logging_did() {
        let _host = HOST_MODULE.lock().unwrap_or_else(PoisonError::into_inner);
        Python::initialize();
        let outcome = Python::attach(|py| -> PyResult<(Option<f64>, Option<f64>)> {
            let cost_map = Py::new(py, NativeModelCostMap::bundled(Some(0), Some(0.0))?)?;
            let module_name = std::ffi::CString::new(host::MODULE).unwrap();
            let host =
                PyModule::from_code(py, PRICING_HOST, c_str!("host_model_cost.py"), &module_name)?;
            host.setattr("cost_map", &cost_map)?;
            py.run(
                c_str!("def inputs():\n    return cost_map\n"),
                Some(&host.dict()),
                None,
            )?;
            let modules = py.import("sys")?.getattr("modules")?;
            let packages = ["litellm", "litellm.rust_bridge", "litellm.rust_bridge.host"];
            for package in packages {
                modules.set_item(package, PyModule::new(py, package)?)?;
            }
            modules.set_item(host::MODULE, &host)?;
            let priced = (|| -> PyResult<(Option<f64>, Option<f64>)> {
                let set_up = route_pricing(py)?;
                let row = host.getattr("Row")?.call(
                    (),
                    Some(&pyo3::types::IntoPyDict::into_py_dict(
                        [("litellm_provider", "anthropic")],
                        py,
                    )?),
                )?;
                row.set_item("input_cost_per_token", 2e-6)?;
                host.getattr("write")?.call1(("claude-opus-5-5", row))?;
                Ok((route_total(&set_up), route_total(&route_pricing(py)?)))
            })();
            for name in [host::MODULE].into_iter().chain(packages) {
                modules.del_item(name)?;
            }
            priced
        });

        let (set_up, later) = outcome.unwrap();
        assert_eq!(set_up, Some(1000.0 * 2e-6));
        assert_eq!(later, set_up);
    }
}
