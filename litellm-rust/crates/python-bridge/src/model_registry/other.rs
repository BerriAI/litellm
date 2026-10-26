use std::collections::BTreeSet;
use std::num::NonZeroU64;

use litellm_cost::money::UnitPrice;
use litellm_cost::{
    DefaultEligibility, Measurement, OtherContext, OtherSelection, Quantity, QuantityUnit, ReportTo,
};
use litellm_model_catalog::Model;
use litellm_model_catalog::wire::Fields;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyMapping};

use super::live::{RowErrors, parse_meter, row_fields};
use super::row_error;

#[derive(Clone, Copy)]
pub(super) struct LegacyCount {
    pub value: u64,
    pub boolean: bool,
}

impl LegacyCount {
    pub const ZERO: Self = Self {
        value: 0,
        boolean: false,
    };
}

impl FromPyObject<'_, '_> for LegacyCount {
    type Error = PyErr;
    fn extract(value: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            value: value.extract()?,
            boolean: value.is_instance_of::<PyBool>(),
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct StrictCount(pub u64);

impl FromPyObject<'_, '_> for StrictCount {
    type Error = PyErr;

    fn extract(value: Borrowed<'_, '_, PyAny>) -> PyResult<Self> {
        if value.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(
                "count must be an integer, not a boolean",
            ));
        }
        value.extract::<u64>().map(Self)
    }
}

fn finite(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(
            "value must be finite and non-negative, not a boolean",
        ));
    }
    let value = value.extract::<f64>()?;
    Measurement::new(value).map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(value)
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct MeasuredQuantity {
    value: Measurement,
    unit: QuantityUnit,
}

#[pymethods]
impl MeasuredQuantity {
    #[new]
    #[pyo3(signature = (value, *, unit="meter"))]
    fn new(value: &Bound<'_, PyAny>, unit: &str) -> PyResult<Self> {
        let value = Measurement::new(finite(value)?)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let unit = unit
            .parse()
            .map_err(|_| PyValueError::new_err("unit must be meter or milliseconds"))?;
        Ok(Self { value, unit })
    }
}

impl MeasuredQuantity {
    pub(super) fn quantity(&self, meter: &litellm_cost::price::Meter) -> PyResult<Quantity> {
        Quantity::measured(self.value, self.unit, meter)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

#[pyclass(frozen, skip_from_py_object, module = "litellm.rust_bridge._native")]
#[derive(Clone)]
pub(crate) struct PriceRowInput {
    source_id: String,
    fields: Fields,
}

#[pymethods]
impl PriceRowInput {
    #[new]
    fn new(source_id: String, row: &Bound<'_, PyMapping>) -> PyResult<Self> {
        if source_id.trim().is_empty() || source_id == "primary" {
            return Err(PyValueError::new_err(
                "source_id must be nonempty and cannot be primary",
            ));
        }
        let snapshot = row.py().get_type::<PyDict>().call1((row,))?;
        let fields = row_fields(&source_id, &snapshot).map_err(|_| {
            PyValueError::new_err("fallback row must contain finite JSON-compatible values")
        })?;
        Ok(Self { source_id, fields })
    }
}

#[pyclass(frozen, skip_from_py_object, module = "litellm.rust_bridge._native")]
#[derive(Clone)]
pub(crate) struct DefaultRateInput {
    meter: litellm_cost::price::Meter,
    rate: UnitPrice,
    per_units: NonZeroU64,
    eligibility: DefaultEligibility,
}

#[pymethods]
impl DefaultRateInput {
    #[new]
    #[pyo3(signature = (meter, value, *, per_units=StrictCount(1), apply_when), text_signature = "(meter, value, *, per_units=1, apply_when)")]
    fn new(
        meter: &str,
        value: &Bound<'_, PyAny>,
        per_units: StrictCount,
        apply_when: &str,
    ) -> PyResult<Self> {
        let meter =
            parse_meter(meter).ok_or_else(|| PyValueError::new_err("unknown default meter"))?;
        let rate = UnitPrice::new(finite(value)?)
            .ok_or_else(|| PyValueError::new_err("invalid default rate"))?;
        let per_units = NonZeroU64::new(per_units.0)
            .ok_or_else(|| PyValueError::new_err("per_units must be positive"))?;
        let eligibility = apply_when
            .parse()
            .map_err(|_| PyValueError::new_err("apply_when must be absent or absent_or_null"))?;
        Ok(Self {
            meter,
            rate,
            per_units,
            eligibility,
        })
    }
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Additive;
#[pymethods]
impl Additive {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct PromptUnits;
#[pymethods]
impl PromptUnits {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Ocr;
#[pymethods]
impl Ocr {
    #[new]
    fn new() -> Self {
        Self
    }
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct SearchTiers;
#[pymethods]
impl SearchTiers {
    #[new]
    #[pyo3(signature = (*, outside="last"))]
    fn new(outside: &str) -> PyResult<Self> {
        if outside != "last" {
            return Err(PyValueError::new_err("outside must be last"));
        }
        Ok(Self)
    }
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct FirstUnit {
    meters: Vec<litellm_cost::price::Meter>,
}
#[pymethods]
impl FirstUnit {
    #[new]
    fn new(meters: Vec<String>) -> PyResult<Self> {
        let meters = meters
            .iter()
            .map(|meter| {
                parse_meter(meter).ok_or_else(|| PyValueError::new_err("unknown alternative meter"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        if meters.is_empty() || meters.iter().collect::<BTreeSet<_>>().len() != meters.len() {
            return Err(PyValueError::new_err(
                "alternative meters must be nonempty and unique",
            ));
        }
        Ok(Self { meters })
    }
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct DirectionalCharacters {
    threshold: NonZeroU64,
    comparison_scale: NonZeroU64,
}
#[pymethods]
impl DirectionalCharacters {
    #[new]
    fn new(threshold: StrictCount, comparison_scale: StrictCount) -> PyResult<Self> {
        let positive = |value| {
            NonZeroU64::new(value).ok_or_else(|| {
                PyValueError::new_err("threshold and comparison_scale must be positive")
            })
        };
        Ok(Self {
            threshold: positive(threshold.0)?,
            comparison_scale: positive(comparison_scale.0)?,
        })
    }
}

#[derive(FromPyObject)]
enum SelectionInput<'py> {
    Additive(PyRef<'py, Additive>),
    PromptUnits(PyRef<'py, PromptUnits>),
    FirstUnit(PyRef<'py, FirstUnit>),
    SearchTiers(PyRef<'py, SearchTiers>),
    Ocr(PyRef<'py, Ocr>),
    DirectionalCharacters(PyRef<'py, DirectionalCharacters>),
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct OtherMeterInput {
    selection: OtherSelection,
    sources: Vec<PriceRowInput>,
    defaults: Vec<DefaultRateInput>,
    report_to: Option<ReportTo>,
}

#[pymethods]
impl OtherMeterInput {
    #[new]
    #[pyo3(signature = (selection, *, fallback_sources=Vec::new(), defaults=Vec::new(), report_to=None), text_signature = "(selection, *, fallback_sources=(), defaults=(), report_to=None)")]
    fn new(
        selection: SelectionInput<'_>,
        fallback_sources: Vec<PyRef<'_, PriceRowInput>>,
        defaults: Vec<PyRef<'_, DefaultRateInput>>,
        report_to: Option<&str>,
    ) -> PyResult<Self> {
        let selection = match selection {
            SelectionInput::Additive(value) => {
                drop(value);
                OtherSelection::Additive
            }
            SelectionInput::PromptUnits(value) => {
                drop(value);
                OtherSelection::PromptUnits
            }
            SelectionInput::FirstUnit(value) => OtherSelection::FirstUnit(value.meters.clone()),
            SelectionInput::SearchTiers(value) => {
                drop(value);
                OtherSelection::SearchTiers
            }
            SelectionInput::Ocr(value) => {
                drop(value);
                OtherSelection::Ocr
            }
            SelectionInput::DirectionalCharacters(value) => OtherSelection::DirectionalCharacters {
                threshold: value.threshold,
                comparison_scale: value.comparison_scale,
            },
        };
        if fallback_sources
            .iter()
            .map(|source| &source.source_id)
            .collect::<BTreeSet<_>>()
            .len()
            != fallback_sources.len()
        {
            return Err(PyValueError::new_err("fallback source IDs must be unique"));
        }
        if defaults
            .iter()
            .map(|rate| &rate.meter)
            .collect::<BTreeSet<_>>()
            .len()
            != defaults.len()
        {
            return Err(PyValueError::new_err("default meters must be unique"));
        }
        let report_to = report_to
            .map(|value| {
                value
                    .parse()
                    .map_err(|_| PyValueError::new_err("report_to must be input or output"))
            })
            .transpose()?;
        Ok(Self {
            selection,
            sources: fallback_sources
                .into_iter()
                .map(|source| (*source).clone())
                .collect(),
            defaults: defaults.into_iter().map(|rate| (*rate).clone()).collect(),
            report_to,
        })
    }
}

impl OtherMeterInput {
    pub(super) fn directional(&self) -> bool {
        matches!(self.selection, OtherSelection::DirectionalCharacters { .. })
    }

    pub(super) fn prepare(
        &self,
        row: &Bound<'_, PyAny>,
    ) -> Result<(Model, OtherContext), RowErrors> {
        let fields = row_fields("", row)?;
        let errors = |errors: Vec<litellm_model_catalog::Located>| RowErrors {
            errors: errors.iter().map(row_error).collect(),
        };
        let (mut model, primary) =
            Model::from_fields_with_other_source(&fields, "primary".to_owned()).map_err(errors)?;
        let mut context = OtherContext {
            selection: self.selection.clone(),
            sources: vec![primary],
            report_to: self.report_to,
            ..OtherContext::default()
        };
        for source in &self.sources {
            let (fallback, metadata) =
                Model::from_fields_with_other_source(&source.fields, source.source_id.clone())
                    .map_err(errors)?;
            model
                .prices
                .append_source(context.sources.len(), fallback.prices);
            context.sources.push(metadata);
        }
        for default in &self.defaults {
            context.insert_default(
                &mut model.prices,
                default.meter.clone(),
                default.rate,
                default.per_units,
                default.eligibility,
            );
        }
        Ok((model, context))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use std::ffi::CString;

    #[rstest]
    #[case::zero("0", true)]
    #[case::fraction("1.5", true)]
    #[case::negative("-1", false)]
    #[case::boolean("True", false)]
    #[case::string("'1.5'", false)]
    #[case::nan("float('nan')", false)]
    #[case::infinite("float('inf')", false)]
    #[case::none("None", false)]
    fn measured_constructor_validates_python_values(
        #[case] expression: &str,
        #[case] accepted: bool,
    ) {
        Python::attach(|py| {
            let value = py
                .eval(&CString::new(expression).unwrap(), None, None)
                .unwrap();
            assert_eq!(MeasuredQuantity::new(&value, "meter").is_ok(), accepted);
        });
    }

    #[rstest]
    #[case::zero("0", Some(0))]
    #[case::large_exact("9007199254740993", Some(9007199254740993))]
    #[case::maximum("18446744073709551615", Some(u64::MAX))]
    #[case::overflow("18446744073709551616", None)]
    #[case::negative("-1", None)]
    #[case::fraction("1.5", None)]
    #[case::boolean("True", None)]
    fn new_context_counts_are_strict_and_exact(
        #[case] expression: &str,
        #[case] expected: Option<u64>,
    ) {
        Python::attach(|py| {
            let value = py
                .eval(&CString::new(expression).unwrap(), None, None)
                .unwrap();
            assert_eq!(
                value.extract::<StrictCount>().ok().map(|count| count.0),
                expected
            );
        });
    }

    #[rstest]
    #[case::finite("{'input_cost_per_token': 1.0}", true)]
    #[case::nan("{'input_cost_per_token': float('nan')}", false)]
    #[case::infinite("{'input_cost_per_token': float('inf')}", false)]
    fn fallback_source_rejects_nonfinite_before_json_conversion(
        #[case] expression: &str,
        #[case] accepted: bool,
    ) {
        Python::attach(|py| {
            let value = py
                .eval(&CString::new(expression).unwrap(), None, None)
                .unwrap();
            assert_eq!(
                PriceRowInput::new("published".to_owned(), value.cast::<PyMapping>().unwrap())
                    .is_ok(),
                accepted
            );
        });
    }
}
