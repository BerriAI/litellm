use std::sync::Arc;

use litellm_callbacks_prometheus::{
    FamilySpec, Kind, LogLine, LogSink, ManualClock, Op, Prometheus, SeriesLimits, Store,
    StoreError, SystemClock, Target,
    ops::{self, ErrorKind, OpError},
    store::{Clock, FamilyId},
};
use litellm_callbacks_python::RustCallback;
use pyo3::{
    exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyDict, PyFloat, PyTuple},
};

struct ShareClock(Arc<ManualClock>);

impl Clock for ShareClock {
    fn monotonic(&self) -> f64 {
        self.0.monotonic()
    }

    fn wall(&self) -> f64 {
        self.0.wall()
    }
}

fn store_error(error: StoreError) -> PyErr {
    match error {
        StoreError::MostRecent(_) => PyRuntimeError::new_err(error.to_string()),
        _ => PyValueError::new_err(error.to_string()),
    }
}

fn op_error(error: OpError) -> PyErr {
    match error.kind {
        ErrorKind::Value => PyValueError::new_err(error.message),
        ErrorKind::Key => PyKeyError::new_err(error.message),
        ErrorKind::Type => PyTypeError::new_err(error.message),
        ErrorKind::Runtime => PyRuntimeError::new_err(error.message),
    }
}

#[pyclass(frozen, name = "NativePrometheusStore")]
pub struct NativePrometheusStore {
    store: Arc<Store>,
    clock: Option<Arc<ManualClock>>,
}

#[pymethods]
impl NativePrometheusStore {
    #[new]
    #[pyo3(signature = (max_series=None, ttl_seconds=None, cleanup_interval_seconds=None, manual_clock=false))]
    fn new(
        max_series: Option<i64>,
        ttl_seconds: Option<f64>,
        cleanup_interval_seconds: Option<f64>,
        manual_clock: bool,
    ) -> Self {
        let limits = SeriesLimits {
            max_series,
            ttl_seconds,
            cleanup_interval_seconds,
        };
        let clock = manual_clock.then(|| Arc::new(ManualClock::default()));
        let boxed: Box<dyn Clock> = match &clock {
            Some(clock) => Box::new(ShareClock(Arc::clone(clock))),
            None => Box::new(SystemClock::default()),
        };
        Self {
            store: Arc::new(Store::new(limits, boxed)),
            clock,
        }
    }

    fn set_time(&self, monotonic: f64, wall: f64) -> PyResult<()> {
        let clock = self
            .clock
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("this store reads the system clock"))?;
        clock.set(monotonic, wall);
        Ok(())
    }

    fn declare_labels(&self, metric: &str, labels: Vec<String>) {
        self.store.declare_labels(metric, labels);
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (kind, declared, documentation, labelnames, excluded, buckets, most_recent, wrapped, shares_overflow))]
    fn metric(
        &self,
        kind: &str,
        declared: String,
        documentation: String,
        labelnames: Vec<String>,
        excluded: Vec<String>,
        buckets: Vec<f64>,
        most_recent: bool,
        wrapped: bool,
        shares_overflow: bool,
    ) -> PyResult<NativePrometheusMetric> {
        let kind: Kind = kind
            .parse()
            .map_err(|_| PyValueError::new_err(format!("unknown metric kind {kind}")))?;
        let id = self
            .store
            .register(FamilySpec {
                declared: declared.clone(),
                documentation,
                kind,
                labelnames: labelnames.clone(),
                excluded,
                buckets,
                most_recent,
                wrapped,
                shares_overflow,
            })
            .map_err(store_error)?;
        Ok(NativePrometheusMetric {
            store: Arc::clone(&self.store),
            id,
            kind,
            declared,
            labelnames,
            wrapped,
            target: Target::Parent,
        })
    }

    #[pyo3(signature = (metric, values, max_series=None, ttl_seconds=None, cleanup_interval_seconds=None))]
    fn track_end_user(
        &self,
        metric: &str,
        values: Vec<Option<String>>,
        max_series: Option<i64>,
        ttl_seconds: Option<f64>,
        cleanup_interval_seconds: Option<f64>,
    ) {
        let Some(id) = self.store.family(metric) else {
            return;
        };
        self.store.track_end_user(
            id,
            &values,
            &SeriesLimits {
                max_series,
                ttl_seconds,
                cleanup_interval_seconds,
            },
        );
    }

    fn op(&self, name: &str, facts: &str) -> PyResult<()> {
        ops::apply(&self.store, name, facts).map_err(op_error)
    }

    fn callback(&self, log: Py<PyAny>) -> PyResult<RustCallback> {
        Ok(RustCallback::new(Arc::new(Prometheus::new(
            Arc::clone(&self.store),
            Arc::new(PythonLog(log)),
        )))?)
    }
}

struct PythonLog(Py<PyAny>);

impl LogSink for PythonLog {
    fn log(&self, line: &LogLine) {
        Python::attach(|py| {
            let level: &'static str = line.level.into();
            if let Err(error) = self.0.call1(py, (level, line.format, line.args.clone())) {
                error.write_unraisable(py, None);
            }
        });
    }
}

type SeriesRow = (Vec<String>, f64, f64, Vec<f64>, f64);
type FamilyRow = (
    String,
    String,
    &'static str,
    Vec<String>,
    Vec<f64>,
    Vec<SeriesRow>,
);

#[pyclass(frozen, name = "NativePrometheusMetric")]
pub struct NativePrometheusMetric {
    store: Arc<Store>,
    id: FamilyId,
    kind: Kind,
    declared: String,
    labelnames: Vec<String>,
    wrapped: bool,
    target: Target,
}

fn py_strings(values: &Bound<'_, PyTuple>) -> PyResult<Vec<String>> {
    values
        .iter()
        .map(|value| Ok(value.str()?.to_string()))
        .collect()
}

impl NativePrometheusMetric {
    fn with_target(&self, target: Target) -> Self {
        Self {
            store: Arc::clone(&self.store),
            id: self.id,
            kind: self.kind,
            declared: self.declared.clone(),
            labelnames: self.labelnames.clone(),
            wrapped: self.wrapped,
            target,
        }
    }

    fn describe(&self) -> String {
        let name = match self.kind {
            Kind::Counter => self
                .declared
                .strip_suffix("_total")
                .unwrap_or(&self.declared),
            Kind::Gauge | Kind::Histogram => &self.declared,
        };
        format!("{}:{name}", <&str>::from(self.kind))
    }

    fn apply(&self, op: Op) -> PyResult<()> {
        self.store
            .apply(self.id, &self.target, op)
            .map_err(store_error)
    }

    fn wrapped_values(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<String>> {
        if !args.is_empty() {
            return py_strings(args);
        }
        self.labelnames
            .iter()
            .map(|name| {
                kwargs
                    .and_then(|kwargs| kwargs.get_item(name).ok().flatten())
                    .ok_or_else(|| PyKeyError::new_err(name.clone()))
                    .and_then(|value| Ok(value.str()?.to_string()))
            })
            .collect()
    }

    fn plain_values(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Vec<String>> {
        if self.labelnames.is_empty() {
            return Err(PyValueError::new_err(format!(
                "No label names were set when constructing {}",
                self.describe()
            )));
        }
        let kwargs = kwargs.filter(|kwargs| !kwargs.is_empty());
        match kwargs {
            Some(_) if !args.is_empty() => {
                Err(PyValueError::new_err("Can't pass both *args and **kwargs"))
            }
            Some(kwargs) => {
                let mut given: Vec<String> = kwargs
                    .keys()
                    .iter()
                    .map(|key| key.extract::<String>())
                    .collect::<PyResult<_>>()?;
                let mut wanted = self.labelnames.clone();
                given.sort();
                wanted.sort();
                if given != wanted {
                    return Err(PyValueError::new_err("Incorrect label names"));
                }
                self.labelnames
                    .iter()
                    .map(|name| {
                        let value = kwargs
                            .get_item(name)?
                            .ok_or_else(|| PyKeyError::new_err(name.clone()))?;
                        Ok(value.str()?.to_string())
                    })
                    .collect()
            }
            None if args.len() != self.labelnames.len() => {
                Err(PyValueError::new_err("Incorrect label count"))
            }
            None => py_strings(args),
        }
    }
}

#[pymethods]
impl NativePrometheusMetric {
    #[getter]
    fn name(&self) -> &str {
        &self.declared
    }

    #[pyo3(signature = (*args, **kwargs))]
    fn labels(
        &self,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if self.target != Target::Parent {
            return Err(PyValueError::new_err(format!(
                "{} already has labels set; can not chain calls to .labels()",
                self.describe()
            )));
        }
        let values = if self.wrapped {
            self.wrapped_values(args, kwargs)?
        } else {
            self.plain_values(args, kwargs)?
        };
        let target = self.store.labels(self.id, &values).map_err(store_error)?;
        Ok(self.with_target(target))
    }

    #[pyo3(signature = (amount=None))]
    fn inc(&self, py: Python<'_>, amount: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let amount = match amount {
            Some(amount) => to_float(py, &amount)?,
            None => 1.0,
        };
        self.apply(Op::Inc(amount))
    }

    #[pyo3(signature = (amount=None))]
    fn dec(&self, py: Python<'_>, amount: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let amount = match amount {
            Some(amount) => to_float(py, &amount)?,
            None => 1.0,
        };
        self.apply(Op::Dec(amount))
    }

    fn set(&self, py: Python<'_>, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.apply(Op::Set(to_float(py, &value)?))
    }

    fn observe(&self, py: Python<'_>, amount: Bound<'_, PyAny>) -> PyResult<()> {
        self.apply(Op::Observe(to_float(py, &amount)?))
    }

    #[pyo3(signature = (*values))]
    fn remove(&self, values: &Bound<'_, PyTuple>) -> PyResult<()> {
        let strings = py_strings(values)?;
        if !self.wrapped && self.labelnames.is_empty() {
            return Err(PyValueError::new_err(format!(
                "No label names were set when constructing {}",
                self.describe()
            )));
        }
        if !self.wrapped && strings.len() != self.labelnames.len() {
            return Err(PyValueError::new_err(format!(
                "Incorrect label count (expected {}, got {})",
                self.labelnames.len(),
                values.repr()?
            )));
        }
        match self.store.remove(self.id, &strings) {
            Ok(()) => Ok(()),
            Err(StoreError::Missing) => Err(PyKeyError::new_err(
                PyTuple::new(values.py(), strings)?.unbind(),
            )),
            Err(error) => Err(store_error(error)),
        }
    }

    fn clear(&self) {
        self.store.clear(self.id);
    }

    fn collect(&self) -> FamilyRow {
        let family = self.store.snapshot(self.id);
        (
            family.name,
            family.documentation,
            family.kind.into(),
            family.labelnames,
            family.buckets,
            family
                .series
                .into_iter()
                .map(|(values, series)| {
                    (
                        values,
                        series.value,
                        series.sum,
                        series.buckets,
                        series.created,
                    )
                })
                .collect(),
        )
    }
}

fn to_float(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<f64> {
    py.get_type::<PyFloat>().call1((value,))?.extract()
}
