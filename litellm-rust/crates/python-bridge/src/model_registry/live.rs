use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use litellm_cost::price::{Band, Channel, Meter, ServiceTier};
use litellm_cost::{SelectionPolicy, Usage, When};
use litellm_host_python::from_py;
use litellm_model_catalog::wire::{Fields, FlatKey};
use litellm_model_catalog::{self as resolution, Keys, Query, Resolved};
use litellm_model_catalog::{Error, Located, Model};
use pyo3::IntoPyObjectExt;
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyModule, PySet, PyString};
use time::OffsetDateTime;

use super::{RowError, escape_pointer_segment, fields_to_dict, nonfinite_errors, row_error};

struct LowercaseIndex {
    dict: Py<PyDict>,
    len: usize,
    keys: HashMap<String, String>,
}

impl LowercaseIndex {
    fn new(dict: &Bound<'_, PyDict>) -> Self {
        Self {
            dict: dict.clone().unbind(),
            len: dict.len(),
            keys: dict
                .keys()
                .iter()
                .filter_map(|key| key.extract::<String>().ok())
                .map(|key| (key.to_lowercase(), key))
                .collect(),
        }
    }
}

#[pyclass(frozen)]
#[derive(Default)]
struct ModelKeyIndex {
    state: Mutex<Option<LowercaseIndex>>,
}

#[pymethods]
impl ModelKeyIndex {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(index) = state.as_ref() {
            visit.call(&index.dict)?;
        }
        Ok(())
    }

    fn __clear__(&self) {
        let old = self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        drop(old);
    }
}

impl ModelKeyIndex {
    fn lowercase(&self, dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
        let previous = self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        let key = key.to_lowercase();
        let index = match previous {
            Some(index)
                if index.dict.bind(dict.py()).is(dict)
                    && index.len == dict.len()
                    && match index.keys.get(&key) {
                        Some(hit) => dict.contains(hit)?,
                        None => true,
                    } =>
            {
                index
            }
            _ => LowercaseIndex::new(dict),
        };
        let hit = index.keys.get(&key).cloned();
        let old = self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .replace(index);
        drop(old);
        Ok(hit)
    }
}

pub(crate) fn init_model_key_index(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.dict().set_item(
        "_model_key_index",
        Py::new(module.py(), ModelKeyIndex::default())?,
    )
}

#[pyfunction(pass_module)]
pub(crate) fn invalidate_model_cost_index(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module
        .getattr("_model_key_index")?
        .extract::<PyRef<'_, ModelKeyIndex>>()?
        .__clear__();
    Ok(())
}

struct PyView<'a, 'py> {
    dict: &'a Bound<'py, PyDict>,
    index: &'a ModelKeyIndex,
    error: RefCell<Option<PyErr>>,
}

impl PyView<'_, '_> {
    fn read<T: Default>(&self, result: PyResult<T>) -> T {
        match result {
            Ok(value) => value,
            Err(error) => {
                self.error.borrow_mut().get_or_insert(error);
                T::default()
            }
        }
    }

    fn finish(&self) -> PyResult<()> {
        self.error.borrow_mut().take().map_or(Ok(()), Err)
    }
}

impl Keys for PyView<'_, '_> {
    fn contains(&self, key: &str) -> bool {
        self.read(self.dict.contains(key))
    }

    fn lowercase(&self, key: &str) -> Option<String> {
        self.read(self.index.lowercase(self.dict, key))
    }

    fn provider_of(&self, key: &str) -> Option<String> {
        let row = self.read(self.dict.get_item(key))?;
        let row = row.cast::<PyDict>().ok()?;
        self.read(row.get_item("litellm_provider"))?.extract().ok()
    }
}

struct PyProvider<'a, 'py> {
    value: Bound<'py, PyAny>,
    error: &'a RefCell<Option<PyErr>>,
}

impl PyProvider<'_, '_> {
    fn comparison(&self, result: PyResult<bool>) -> Option<bool> {
        match result {
            Ok(value) => Some(value),
            Err(error) => {
                if !error.is_instance_of::<PyException>(self.value.py()) {
                    self.error.borrow_mut().get_or_insert(error);
                }
                None
            }
        }
    }
}

impl resolution::ProviderValue for PyProvider<'_, '_> {
    fn differs(&self, provider: &str) -> Option<bool> {
        self.comparison(self.value.ne(provider))
    }

    fn equals(&self, provider: &str) -> Option<bool> {
        self.comparison(
            PyString::new(self.value.py(), provider)
                .as_any()
                .eq(&self.value),
        )
    }

    fn starts_with(&self, prefix: &str) -> Option<bool> {
        self.comparison(
            self.value
                .call_method1("startswith", (prefix,))
                .and_then(|value| value.is_truthy()),
        )
    }
}

impl resolution::InfoKeys for PyView<'_, '_> {
    fn info_provider_match(&self, key: &str, provider: &str) -> resolution::ProviderMatch {
        let value = self
            .read(self.dict.get_item(key))
            .and_then(|row| row.cast::<PyDict>().ok().cloned())
            .and_then(|row| self.read(row.get_item("litellm_provider")));
        match value {
            Some(value) if !value.is_none() => resolution::info_provider_match(
                provider,
                &PyProvider {
                    value,
                    error: &self.error,
                },
            ),
            _ => resolution::ProviderMatch::Match,
        }
    }
}

#[pyclass(frozen, get_all, module = "litellm.rust_bridge._native")]
pub(crate) struct ResolvedModel {
    key: String,
    via: String,
}

#[pyclass(frozen, get_all, module = "litellm.rust_bridge._native")]
pub(crate) struct ModelNotFound {
    tried: Vec<String>,
}

#[pyclass(frozen, get_all, module = "litellm.rust_bridge._native")]
pub(crate) struct InvalidProvider {
    key: String,
    provider: String,
}

#[pyclass(frozen, get_all, module = "litellm.rust_bridge._native")]
pub(crate) struct RegistrationTarget {
    write: String,
    inherit_from: Option<String>,
    names_the_row: bool,
}

#[derive(IntoPyObject)]
pub(crate) enum ResolutionResult {
    Found(ResolvedModel),
    NotFound(ModelNotFound),
    Invalid(InvalidProvider),
}

#[pyfunction(pass_module, signature = (model_cost, model, *, provider=None, base_model=None, region=None, inferred=None, info_read=false))]
#[expect(
    clippy::too_many_arguments,
    reason = "preserve the keyword-only binding contract"
)]
pub(crate) fn resolve_model(
    module: &Bound<'_, PyModule>,
    model_cost: &Bound<'_, PyDict>,
    model: &str,
    provider: Option<&str>,
    base_model: Option<&str>,
    region: Option<&str>,
    inferred: Option<(&str, Option<&str>)>,
    info_read: bool,
) -> PyResult<ResolutionResult> {
    let index = module.getattr("_model_key_index")?;
    let index = index.extract::<PyRef<'_, ModelKeyIndex>>()?;
    let view = PyView {
        dict: model_cost,
        index: &index,
        error: RefCell::default(),
    };
    let query = Query {
        model,
        provider,
        base_model,
        region,
    };
    let inferred =
        inferred.map(|(model, provider)| resolution::InferredProvider { model, provider });
    let result = if info_read {
        resolution::resolve_info(&view, &query, inferred)
    } else {
        Ok(match inferred {
            Some(inferred) => resolution::resolve_inferred(&view, &query, inferred),
            None => resolution::resolve(&view, &query),
        })
    };
    view.finish()?;
    Ok(match result {
        Ok(Resolved::Found { key, via }) => ResolutionResult::Found(ResolvedModel {
            key,
            via: format!("{via:?}"),
        }),
        Ok(Resolved::NotFound { tried }) => ResolutionResult::NotFound(ModelNotFound { tried }),
        Err(error) => ResolutionResult::Invalid(InvalidProvider {
            key: error.key,
            provider: error.provider,
        }),
    })
}

#[pyfunction(pass_module, signature = (model_cost, key, *, provider=None))]
pub(crate) fn registration_target(
    module: &Bound<'_, PyModule>,
    model_cost: &Bound<'_, PyDict>,
    key: &str,
    provider: Option<&str>,
) -> PyResult<RegistrationTarget> {
    let index = module.getattr("_model_key_index")?;
    let index = index.extract::<PyRef<'_, ModelKeyIndex>>()?;
    let view = PyView {
        dict: model_cost,
        index: &index,
        error: RefCell::default(),
    };
    let providers = PyModule::import(model_cost.py(), "litellm.types.utils")?
        .getattr("LlmProvidersSet")?
        .cast_into::<PySet>()?;
    let target =
        resolution::registration_target_with_provider_names(&view, key, provider, |prefix| {
            view.read(providers.contains(prefix))
        });
    view.finish()?;
    Ok(RegistrationTarget {
        write: target.write,
        inherit_from: target.inherit_from,
        names_the_row: target.names_the_row,
    })
}

#[pyclass(frozen, get_all, module = "litellm.rust_bridge._native")]
pub(crate) struct RowErrors {
    errors: Vec<RowError>,
}

fn lower_row(key: &str, row: &Bound<'_, PyAny>) -> Result<(Fields, Model), RowErrors> {
    lower_row_with_policy(key, row, litellm_model_catalog::PriceLowering::Generic)
}

fn lower_row_with_policy(
    key: &str,
    row: &Bound<'_, PyAny>,
    policy: litellm_model_catalog::PriceLowering,
) -> Result<(Fields, Model), RowErrors> {
    let path = format!("/{}", escape_pointer_segment(key));
    let located = |error| RowErrors {
        errors: vec![row_error(&Located::new(&path, error))],
    };
    if !row.is_instance_of::<PyDict>() {
        return Err(located(Error::NotAnObject));
    }
    let mut errors = Vec::new();
    let valid = nonfinite_errors(row, &path, &mut errors, 0);
    if !errors.is_empty() {
        return Err(RowErrors { errors });
    }
    let invalid = || {
        located(Error::Invalid(
            <serde_json::Error as serde::de::Error>::custom(
                "row must contain JSON-compatible values",
            ),
        ))
    };
    valid.map_err(|_| invalid())?;
    let fields: Fields = from_py(row).map_err(|_| invalid())?;
    let model = Model::from_fields_with_policy(&fields, policy).map_err(|errors| RowErrors {
        errors: errors
            .into_iter()
            .map(|error| row_error(&Located::new(format!("{path}{}", error.path), error.error)))
            .collect(),
    })?;
    Ok((fields, model))
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct ModelInfoView {
    #[pyo3(get)]
    key: String,
    fields: Fields,
    model: Box<Model>,
}

#[pymethods]
impl ModelInfoView {
    #[getter]
    fn fields(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        fields_to_dict(py, &self.fields)
    }

    #[getter]
    fn capabilities(&self) -> std::collections::BTreeMap<String, bool> {
        self.model.capabilities.clone()
    }

    #[getter]
    fn provider(&self) -> Option<String> {
        self.model.attributes.litellm_provider.clone()
    }

    #[getter]
    fn mode(&self) -> Option<String> {
        self.fields
            .get("mode")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    }

    #[getter]
    fn max_tokens(&self) -> Option<u64> {
        self.model.attributes.max_tokens
    }

    #[getter]
    fn max_input_tokens(&self) -> Option<u64> {
        self.model.attributes.max_input_tokens
    }

    #[getter]
    fn max_output_tokens(&self) -> Option<u64> {
        self.model.attributes.max_output_tokens
    }

    #[getter]
    fn input_cost_per_token(&self) -> Option<f64> {
        self.model
            .prices
            .rate(
                &Meter::Input(litellm_cost::price::InputUnit::Token(
                    litellm_cost::price::TokenKind::Text,
                )),
                ServiceTier::Standard,
                Band::Base,
            )
            .map(|rate| rate.get())
    }

    #[getter]
    fn output_cost_per_token(&self) -> Option<f64> {
        self.model
            .prices
            .rate(
                &Meter::Output(litellm_cost::price::OutputUnit::Token(
                    litellm_cost::price::TokenKind::Text,
                )),
                ServiceTier::Standard,
                Band::Base,
            )
            .map(|rate| rate.get())
    }
}

#[derive(IntoPyObject)]
pub(crate) enum ModelInfoResult {
    Info(ModelInfoView),
    Errors(RowErrors),
}

#[pyfunction]
pub(crate) fn model_info(model_cost: &Bound<'_, PyDict>, key: &str) -> PyResult<ModelInfoResult> {
    let row = model_cost
        .get_item(key)?
        .unwrap_or_else(|| model_cost.py().None().into_bound(model_cost.py()));
    Ok(match lower_row(key, &row) {
        Ok((fields, model)) => ModelInfoResult::Info(ModelInfoView {
            key: key.to_owned(),
            fields,
            model: Box::new(model),
        }),
        Err(errors) => ModelInfoResult::Errors(errors),
    })
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct UsageInput {
    usage: Usage,
}

#[pymethods]
impl UsageInput {
    #[new]
    #[pyo3(signature = (quantities, *, prompt_tokens=0, inclusive_threshold=false, search_results=0))]
    fn new(
        quantities: &Bound<'_, PyDict>,
        prompt_tokens: u64,
        inclusive_threshold: bool,
        search_results: u64,
    ) -> PyResult<Self> {
        let quantities = quantities
            .iter()
            .map(|(key, quantity)| {
                let key: String = key.extract()?;
                let meter = parse_meter(&key)
                    .ok_or_else(|| PyValueError::new_err(format!("unknown usage meter: {key}")))?;
                Ok((meter, quantity.extract::<u64>()?))
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            usage: Usage {
                quantities,
                prompt_tokens,
                inclusive_threshold,
                search_results,
            },
        })
    }
}

fn parse_meter(key: &str) -> Option<Meter> {
    use litellm_cost::price::{GuardrailUnit, SearchContextSize};
    if let Some(counter) = key.strip_prefix("guardrail/") {
        return (!counter.is_empty()).then(|| Meter::Guardrail(GuardrailUnit::new(counter)));
    }
    match key {
        "web_search/low" => Some(Meter::WebSearch(SearchContextSize::Low)),
        "web_search/medium" => Some(Meter::WebSearch(SearchContextSize::Medium)),
        "web_search/high" => Some(Meter::WebSearch(SearchContextSize::High)),
        key => FlatKey::parse(key)
            .filter(|flat| !flat.has_suffix())
            .map(|flat| flat.meter),
    }
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct WhenInput {
    when: When,
}

#[pymethods]
impl WhenInput {
    #[new]
    #[pyo3(signature = (at, *, tier="standard", channel="online"))]
    fn new(at: i64, tier: &str, channel: &str) -> PyResult<Self> {
        let tier = match tier {
            "standard" => ServiceTier::Standard,
            "priority" => ServiceTier::Priority,
            "flex" => ServiceTier::Flex,
            "ultrafast" => ServiceTier::Ultrafast,
            "balanced" => ServiceTier::Balanced,
            _ => return Err(PyValueError::new_err("unknown service tier")),
        };
        let channel = match channel {
            "online" => Channel::Online,
            "batch" => Channel::Batch,
            "decisions" => Channel::Decisions,
            _ => return Err(PyValueError::new_err("unknown pricing channel")),
        };
        let at = OffsetDateTime::from_unix_timestamp(at)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            when: When { at, tier, channel },
        })
    }
}

#[pyclass(
    frozen,
    get_all,
    skip_from_py_object,
    module = "litellm.rust_bridge._native"
)]
#[derive(Clone)]
pub(crate) struct PriceLine {
    meter: String,
    quantity: u64,
    unit_price: f64,
    amount: f64,
    fallback_from: Option<String>,
}

#[pyclass(frozen, get_all, module = "litellm.rust_bridge._native")]
pub(crate) struct Priced {
    lines: Vec<PriceLine>,
    total: f64,
    input: f64,
    output: f64,
}

#[pyclass(frozen, get_all, module = "litellm.rust_bridge._native")]
pub(crate) struct Unpriced {
    missing: Vec<String>,
    partial: Py<Priced>,
}

impl From<litellm_cost::Breakdown> for Priced {
    fn from(breakdown: litellm_cost::Breakdown) -> Self {
        Self {
            total: breakdown.total.get(),
            input: breakdown.input().get(),
            output: breakdown.output().get(),
            lines: breakdown
                .lines
                .into_iter()
                .map(|line| PriceLine {
                    meter: meter_name(&line.meter),
                    quantity: line.quantity,
                    unit_price: line.unit_price.get(),
                    amount: line.amount.get(),
                    fallback_from: line.fallback_from.as_ref().map(meter_name),
                })
                .collect(),
        }
    }
}

#[derive(IntoPyObject)]
pub(crate) enum PriceResult {
    Priced(Priced),
    Unpriced(Unpriced),
    Errors(RowErrors),
}

#[pyclass(
    frozen,
    get_all,
    skip_from_py_object,
    module = "litellm.rust_bridge._native"
)]
#[derive(Clone)]
pub(crate) struct SelectedRate {
    requested_meter: String,
    source_meter: Option<String>,
    unit_price: Option<f64>,
}

#[pyclass(frozen, get_all, module = "litellm.rust_bridge._native")]
pub(crate) struct PriceEvaluation {
    pricing: Py<PyAny>,
    rates: Vec<SelectedRate>,
}

#[pymethods]
impl PriceEvaluation {
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.pricing)
    }
}

#[derive(IntoPyObject)]
pub(crate) enum EvaluationResult {
    Evaluated(PriceEvaluation),
    Errors(RowErrors),
}

#[derive(Clone, Copy, strum::EnumString, strum::VariantArray)]
enum PricePolicy {
    #[strum(serialize = "generic")]
    Generic,
    #[strum(serialize = "sdk_tokens")]
    SdkTokens,
}

impl PricePolicy {
    fn selection(self, model: &Model) -> SelectionPolicy<'_> {
        match self {
            Self::Generic => SelectionPolicy::Generic,
            Self::SdkTokens => SelectionPolicy::SdkTokens(&model.token_price_sources),
        }
    }

    fn lower(self, row: &Bound<'_, PyAny>, channel: Channel) -> Result<Model, RowErrors> {
        let lowering = match (self, channel) {
            (Self::SdkTokens, Channel::Online | Channel::Batch) => {
                litellm_model_catalog::PriceLowering::SdkTokens
            }
            _ => litellm_model_catalog::PriceLowering::Generic,
        };
        lower_row_with_policy("", row, lowering).map(|(_, model)| model)
    }

    fn prepare(
        self,
        mut model: Model,
        channel: Channel,
        ratio: Option<litellm_cost::money::ScaleFactor>,
    ) -> PyResult<Model> {
        if let Some(ratio) = ratio
            && channel == Channel::Online
        {
            model.prices = model
                .prices
                .with_cache_read_default(
                    ratio,
                    model.token_price_sources.off_peak_cache_read_declared,
                )
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
        }
        Ok(model)
    }
}

fn cache_read_default(
    policy: PricePolicy,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<litellm_cost::money::ScaleFactor>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if !matches!(policy, PricePolicy::SdkTokens) {
        return Err(PyValueError::new_err(
            "cache read defaults require sdk_tokens policy",
        ));
    }
    if value.is_instance_of::<PyBool>() {
        return Err(PyValueError::new_err(
            "cache read default ratio must be finite and non-negative",
        ));
    }
    litellm_cost::money::ScaleFactor::new(value.extract()?)
        .map(Some)
        .ok_or_else(|| {
            PyValueError::new_err("cache read default ratio must be finite and non-negative")
        })
}

fn pricing_policy(policy: &str) -> PyResult<PricePolicy> {
    policy
        .parse()
        .map_err(|_| PyValueError::new_err("unknown pricing policy"))
}

#[derive(IntoPyObject)]
pub(crate) enum TokenPricePresence {
    Present(bool),
    Errors(RowErrors),
}

#[pyfunction]
#[pyo3(signature = (row, *, policy="generic"))]
pub(crate) fn has_token_prices(
    row: &Bound<'_, PyAny>,
    policy: &str,
) -> PyResult<TokenPricePresence> {
    Ok(match pricing_policy(policy)?.lower(row, Channel::Online) {
        Ok(model) => TokenPricePresence::Present(model.prices.has_token_prices()),
        Err(errors) => TokenPricePresence::Errors(errors),
    })
}

#[pyfunction]
#[pyo3(signature = (row, usage, when, *, policy="generic", cache_read_default_ratio=None))]
pub(crate) fn price(
    row: &Bound<'_, PyAny>,
    usage: &UsageInput,
    when: &WhenInput,
    policy: &str,
    cache_read_default_ratio: Option<&Bound<'_, PyAny>>,
) -> PyResult<PriceResult> {
    let policy = pricing_policy(policy)?;
    let ratio = cache_read_default(policy, cache_read_default_ratio)?;
    let model = match policy.lower(row, when.when.channel) {
        Ok(model) => model,
        Err(errors) => return Ok(PriceResult::Errors(errors)),
    };
    let model = policy.prepare(model, when.when.channel, ratio)?;
    let policy = policy.selection(&model);
    price_result(
        row.py(),
        litellm_cost::calculate_with_policy(&model.prices, &usage.usage, &when.when, policy),
    )
}

fn price_result(py: Python<'_>, priced: litellm_cost::Priced) -> PyResult<PriceResult> {
    Ok(match priced {
        litellm_cost::Priced::Known(breakdown) => PriceResult::Priced(breakdown.into()),
        litellm_cost::Priced::Unpriced { missing, partial } => PriceResult::Unpriced(Unpriced {
            missing: missing.iter().map(meter_name).collect(),
            partial: Py::new(py, Priced::from(partial))?,
        }),
    })
}

#[pyfunction]
#[pyo3(
    signature = (row, usage, when, *, policy="generic", project=Vec::new(), factors=Vec::new(), cache_read_default_ratio=None),
    text_signature = "(row, usage, when, *, policy='generic', project=(), factors=(), cache_read_default_ratio=None)"
)]
pub(crate) fn price_with_rates(
    row: &Bound<'_, PyAny>,
    usage: &UsageInput,
    when: &WhenInput,
    policy: &str,
    project: Vec<String>,
    factors: Vec<Bound<'_, PyAny>>,
    cache_read_default_ratio: Option<&Bound<'_, PyAny>>,
) -> PyResult<EvaluationResult> {
    let policy = pricing_policy(policy)?;
    let ratio = cache_read_default(policy, cache_read_default_ratio)?;
    let project = project
        .iter()
        .map(|key| {
            parse_meter(key)
                .filter(Meter::is_token)
                .ok_or_else(|| PyValueError::new_err("only token meters can be projected"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let factors = factors
        .iter()
        .map(|factor| {
            if factor.is_instance_of::<PyBool>() {
                return Err(PyValueError::new_err(
                    "factors must be finite non-negative numbers",
                ));
            }
            litellm_cost::money::ScaleFactor::new(factor.extract()?)
                .ok_or_else(|| PyValueError::new_err("factors must be finite non-negative numbers"))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let model = match policy.lower(row, when.when.channel) {
        Ok(model) => model,
        Err(errors) => return Ok(EvaluationResult::Errors(errors)),
    };
    let model = policy.prepare(model, when.when.channel, ratio)?;
    let evaluation = litellm_cost::calculate_with_rates(
        &model.prices,
        &usage.usage,
        &when.when,
        policy.selection(&model),
        &project,
        &factors,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(EvaluationResult::Evaluated(PriceEvaluation {
        pricing: price_result(row.py(), evaluation.pricing)?.into_py_any(row.py())?,
        rates: evaluation
            .rates
            .into_iter()
            .map(|rate| SelectedRate {
                requested_meter: meter_name(&rate.requested_meter),
                source_meter: rate.resolved.as_ref().map(|(meter, _)| meter_name(meter)),
                unit_price: rate.resolved.map(|(_, price)| price.get()),
            })
            .collect(),
    }))
}

fn meter_name(meter: &Meter) -> String {
    use litellm_cost::price::*;
    let token = |kind: &TokenKind| match kind {
        TokenKind::Text => "token",
        TokenKind::Audio => "audio_token",
        TokenKind::Image => "image_token",
        TokenKind::Video => "video_token",
    };
    match meter {
        Meter::Input(unit) => format!(
            "input_cost_per_{}",
            match unit {
                InputUnit::Token(kind) => token(kind).to_owned(),
                InputUnit::Character => "character".into(),
                InputUnit::Image => "image".into(),
                InputUnit::Pixel => "pixel".into(),
                InputUnit::Query { over_results: None } => "query".into(),
                InputUnit::Query {
                    over_results: Some(results),
                } => format!("query_above_{results}_results"),
                InputUnit::Request => "request".into(),
                InputUnit::Second => "second".into(),
                InputUnit::TextMessage => "text_message".into(),
                InputUnit::AudioSecond => "audio_per_second".into(),
                InputUnit::VideoSecond { over_seconds: None } => "video_per_second".into(),
                InputUnit::VideoSecond {
                    over_seconds: Some(seconds),
                } => format!("video_per_second_above_{seconds}s_interval"),
            }
        ),
        Meter::Output(unit) => format!(
            "output_cost_per_{}",
            match unit {
                OutputUnit::Token(kind) => token(kind).to_owned(),
                OutputUnit::ReasoningToken => "reasoning_token".into(),
                OutputUnit::Character => "character".into(),
                OutputUnit::Image(None) => "image".into(),
                OutputUnit::Image(Some(size)) => format!("image_{}", size.as_str()),
                OutputUnit::Pixel => "pixel".into(),
                OutputUnit::Second(None) => "second".into(),
                OutputUnit::Second(Some(resolution)) => format!("second_{}", resolution.as_str()),
                OutputUnit::VideoSecond => "video_per_second".into(),
            }
        ),
        Meter::CacheRead {
            kind: TokenKind::Text,
            mode: CacheReadMode::Explicit,
        } => "explicit_cache_read_input_token_cost".into(),
        Meter::CacheRead { kind, .. } => format!("cache_read_input_{}_cost", token(kind)),
        Meter::CacheWrite { kind, ttl } => format!(
            "cache_creation_input_{}_cost{}",
            token(kind),
            match ttl {
                CacheTtl::Default => "",
                CacheTtl::OneHour => "_above_1hr",
                CacheTtl::TwelveHours => "_above_12hr",
            }
        ),
        Meter::WallClockSecond => "cost_per_second".into(),
        Meter::Feature(feature) => match feature {
            Feature::AnnotationPage => "annotation_cost_per_page",
            Feature::CitationToken => "citation_cost_per_token",
            Feature::CodeInterpreterSession => "code_interpreter_cost_per_session",
            Feature::GoogleMapsGroundingQuery => "google_maps_grounding_cost_per_query",
            Feature::OcrCredit => "ocr_cost_per_credit",
            Feature::OcrPage => "ocr_cost_per_page",
        }
        .into(),
        Meter::WebSearch(context) => format!(
            "web_search/{}",
            match context {
                SearchContextSize::Low => "low",
                SearchContextSize::Medium => "medium",
                SearchContextSize::High => "high",
            }
        ),
        Meter::Guardrail(counter) => format!("guardrail/{}", counter.as_str()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use litellm_host_python::{json_loads, to_py};
    use rstest::{fixture, rstest};
    use serde::Deserialize;
    use serde_json::{Value, json};
    use std::ffi::CString;

    #[fixture]
    fn interpreter() {
        Python::initialize();
    }

    fn module(py: Python<'_>) -> Bound<'_, PyModule> {
        let module = PyModule::new(py, "_live_test").unwrap();
        init_model_key_index(&module).unwrap();
        module
    }

    fn dict(py: Python<'_>, value: Value) -> Bound<'_, PyDict> {
        to_py(py, &value)
            .unwrap()
            .into_bound(py)
            .cast_into::<PyDict>()
            .unwrap()
    }

    #[rstest]
    #[case::bare("o3-pro", None, None, None, "o3-pro", "Exact")]
    #[case::prefixed(
        "o3-pro",
        Some("openai"),
        None,
        None,
        "openai/o3-pro",
        "ProviderPrefix"
    )]
    #[case::already_prefixed("openai/o3-pro", Some("openai"), None, None, "openai/o3-pro", "Exact")]
    #[case::bare_fallback("gpt-4o", Some("openai"), None, None, "gpt-4o", "Exact")]
    #[case::strip(
        "openai/gpt-4o",
        Some("openai"),
        None,
        None,
        "gpt-4o",
        "ProviderStripped"
    )]
    #[case::inferred("openai/gpt-4o", None, None, None, "gpt-4o", "ProviderStripped")]
    #[case::date("gpt-4o-2026-01-01", None, None, None, "gpt-4o", "SuffixStripped")]
    #[case::fine_tune(
        "ft:gpt-4o:org::id",
        Some("openai"),
        None,
        None,
        "ft:gpt-4o",
        "SuffixStripped"
    )]
    #[case::version(
        "databricks-claude-opus-4-99",
        Some("databricks"),
        None,
        None,
        "databricks/databricks-claude-opus-4",
        "ProviderSuffixStripped"
    )]
    #[case::bedrock_date(
        "bedrock/anthropic.claude-sonnet-4-6-20260101",
        None,
        None,
        None,
        "bedrock/anthropic.claude-sonnet-4-6",
        "ProviderSuffixStripped"
    )]
    #[case::folded(
        "hosted_vllm/MODEL",
        None,
        None,
        None,
        "hosted_vllm/Model",
        "Lowercase"
    )]
    #[case::alias("gpt-alias", None, None, None, "gpt-alias", "Exact")]
    #[case::alias_provider(
        "openai/gpt-alias",
        Some("openai"),
        None,
        None,
        "gpt-alias",
        "ProviderStripped"
    )]
    #[case::alias_folded("GPT-ALIAS", None, None, None, "gpt-alias", "Lowercase")]
    #[case::unicode("部署", None, None, None, "部署", "Exact")]
    #[case::base("deployment", None, Some("gpt-4o"), None, "gpt-4o", "BaseModel")]
    #[case::provider_base(
        "gpt-4o",
        Some("openai"),
        Some("o3-pro"),
        None,
        "openai/o3-pro",
        "ProviderPrefix"
    )]
    #[case::base_over_deployment("o3-pro", None, Some("gpt-4o"), None, "gpt-4o", "BaseModel")]
    #[case::regional(
        "model",
        Some("bedrock_mantle"),
        None,
        Some("us-east-1"),
        "bedrock_mantle/us-east-1/model",
        "Region"
    )]
    #[case::embedded_region(
        "bedrock_mantle/us-east-1/model",
        Some("bedrock_mantle"),
        None,
        None,
        "bedrock_mantle/us-east-1/model",
        "Exact"
    )]
    #[case::already_regional(
        "bedrock_mantle/us-east-1/model",
        Some("bedrock_mantle"),
        None,
        Some("us-east-1"),
        "bedrock_mantle/us-east-1/model",
        "Region"
    )]
    #[case::region_fallback(
        "bedrock_mantle/us-west-2/model",
        Some("bedrock_mantle"),
        None,
        None,
        "bedrock_mantle/model",
        "RegionStripped"
    )]
    #[case::query_region_fallback(
        "model",
        Some("bedrock_mantle"),
        None,
        Some("eu-west-1"),
        "bedrock_mantle/model",
        "ProviderPrefix"
    )]
    fn resolves_r3_candidates(
        #[from(interpreter)] _interpreter: (),
        #[case] model: &str,
        #[case] provider: Option<&str>,
        #[case] base: Option<&str>,
        #[case] region: Option<&str>,
        #[case] key: &str,
        #[case] via: &str,
    ) {
        Python::attach(|py| {
            let rows = dict(
                py,
                json!({
                    "o3-pro": {"litellm_provider": "openai"},
                    "openai/o3-pro": {"litellm_provider": "openai"},
                    "gpt-4o": {"litellm_provider": "openai"},
                    "gpt-alias": {"litellm_provider": "openai"},
                    "ft:gpt-4o": {"litellm_provider": "openai"},
                    "databricks/databricks-claude-opus-4": {"litellm_provider": "databricks"},
                    "databricks-claude-opus-4": {"litellm_provider": "databricks"},
                    "bedrock/anthropic.claude-sonnet-4-6": {"litellm_provider": "bedrock"},
                    "hosted_vllm/Model": {"litellm_provider": "hosted_vllm"},
                    "bedrock_mantle/us-east-1/model": {"litellm_provider":"bedrock_mantle"},
                    "bedrock_mantle/model": {"litellm_provider":"bedrock_mantle"},
                    "model": {"litellm_provider":"bedrock_mantle"},
                    "部署": {"litellm_provider": null}
                }),
            );
            let ResolutionResult::Found(found) = resolve_model(
                &module(py),
                &rows,
                model,
                provider,
                base,
                region,
                None,
                false,
            )
            .unwrap() else {
                panic!("not found")
            };
            assert_eq!((found.key.as_str(), found.via.as_str()), (key, via));
        });
    }

    #[rstest]
    #[case::cross_region(
        "us.anthropic.model-v1:0",
        "bedrock",
        "anthropic.model-v1:0",
        "SuffixStripped"
    )]
    #[case::routed(
        "bedrock/converse/anthropic.model-v1:0",
        "bedrock",
        "anthropic.model-v1:0",
        "ProviderStripped"
    )]
    #[case::region(
        "bedrock/us-east-1/anthropic.model-v1:0",
        "bedrock",
        "bedrock/anthropic.model-v1:0",
        "ProviderSuffixStripped"
    )]
    #[case::throughput(
        "anthropic.model-v1:0:51k",
        "bedrock",
        "anthropic.model-v1:0",
        "SuffixStripped"
    )]
    #[case::context(
        "anthropic.model-v1:0[1m]",
        "bedrock",
        "anthropic.model-v1:0",
        "SuffixStripped"
    )]
    #[case::arn(
        "bedrock/arn:aws:bedrock:us-east-1:123:inference-profile/us.anthropic.model-v1:0",
        "bedrock",
        "anthropic.model-v1:0",
        "SuffixStripped"
    )]
    #[case::nova(
        "bedrock/nova/arn:aws:bedrock:us-east-1:123:custom-model/id",
        "bedrock",
        "amazon.nova-custom",
        "SuffixStripped"
    )]
    #[case::nova2(
        "bedrock/converse/nova-2/arn:aws:bedrock:us-east-1:123:custom-model/id",
        "bedrock",
        "amazon.nova-2-custom",
        "SuffixStripped"
    )]
    #[case::gemini("model-001", "gemini", "gemini/model", "ProviderSuffixStripped")]
    #[case::vertex("model-002", "vertex_ai", "model", "SuffixStripped")]
    #[case::bare_first("databricks/model-99", "databricks", "model-99", "ProviderStripped")]
    #[case::double(
        "openai/gpt-model",
        "openai",
        "openai/openai/gpt-model",
        "RepeatedProvider"
    )]
    #[case::fireworks(
        "model",
        "fireworks_ai",
        "fireworks_ai/accounts/fireworks/models/model",
        "ProviderResource"
    )]
    #[case::fireworks_models(
        "models/model",
        "fireworks_ai",
        "fireworks_ai/accounts/fireworks/models/model",
        "ProviderResource"
    )]
    #[case::fireworks_router(
        "model-fast",
        "fireworks_ai",
        "fireworks_ai/accounts/fireworks/routers/model-fast",
        "ProviderResource"
    )]
    fn resolves_r3_provider_resources(
        #[from(interpreter)] _interpreter: (),
        #[case] model: &str,
        #[case] provider: &str,
        #[case] key: &str,
        #[case] via: &str,
    ) {
        Python::attach(|py| {
            let rows = dict(py, json!({key: {"litellm_provider": provider}}));
            let ResolutionResult::Found(found) = resolve_model(
                &module(py),
                &rows,
                model,
                Some(provider),
                None,
                None,
                None,
                false,
            )
            .unwrap() else {
                panic!("not found")
            };
            assert_eq!((found.key.as_str(), found.via.as_str()), (key, via));
        });
    }

    #[rstest]
    #[case::same("openai", Some("openai"), true)]
    #[case::missing("openai", None, true)]
    #[case::mismatch("openai", Some("anthropic"), false)]
    #[case::vertex("vertex_ai", Some("vertex_ai-language-models"), true)]
    #[case::fireworks("fireworks_ai", Some("fireworks_ai-embedding-models"), true)]
    #[case::bedrock("bedrock_converse", Some("bedrock"), true)]
    #[case::azure("azure_ai", Some("openai"), true)]
    #[case::proxy("litellm_proxy", Some("anthropic"), true)]
    #[case::github("github", Some("openai"), true)]
    #[case::nadir("nadir", Some("anthropic"), true)]
    fn reads_live_provider_metadata(
        #[from(interpreter)] _interpreter: (),
        #[case] provider: &str,
        #[case] actual: Option<&str>,
        #[case] found: bool,
    ) {
        Python::attach(|py| {
            let rows = dict(py, json!({"model": {"litellm_provider": actual}}));
            assert_eq!(
                matches!(
                    resolve_model(
                        &module(py),
                        &rows,
                        "model",
                        Some(provider),
                        None,
                        None,
                        None,
                        false
                    )
                    .unwrap(),
                    ResolutionResult::Found(_)
                ),
                found
            );
        });
    }

    #[rstest]
    fn info_reads_distinguish_provider_mismatch_from_invalid(
        #[from(interpreter)] _interpreter: (),
        #[values("openai", "bedrock", "github")] provider: &str,
        #[values(json!(1), json!(true), json!([]), json!({}))] value: Value,
        #[values(None, Some(""), Some("explicit"))] explicit: Option<&str>,
    ) {
        Python::attach(|py| {
            let model = format!("{provider}/model");
            let rows = dict(
                py,
                json!({
                    &model: {"litellm_provider": value},
                    "model": {"litellm_provider": provider}
                }),
            );
            let result = resolve_model(
                &module(py),
                &rows,
                &model,
                explicit.map(|explicit| if explicit.is_empty() { "" } else { provider }),
                None,
                None,
                Some(("model", Some(provider))),
                true,
            )
            .unwrap();
            match result {
                ResolutionResult::Found(found) => {
                    assert_eq!(
                        found.key,
                        if provider == "openai" {
                            "model"
                        } else {
                            &model
                        }
                    );
                    assert_ne!(provider, "bedrock");
                }
                ResolutionResult::Invalid(invalid) => {
                    assert_eq!(provider, "bedrock");
                    assert_eq!(
                        (invalid.key.as_str(), invalid.provider.as_str()),
                        (model.as_str(), provider)
                    );
                }
                ResolutionResult::NotFound(_) => {
                    panic!("mismatch must continue, invalid must abort")
                }
            }
        });
    }

    #[rstest]
    #[case::prefixed("openai/o3-pro", Some("o3-pro"))]
    #[case::versioned(
        "databricks/databricks-claude-opus-4-99",
        Some("databricks/databricks-claude-opus-4")
    )]
    #[case::dated(
        "bedrock/anthropic.claude-sonnet-4-6-20260101",
        Some("bedrock/anthropic.claude-sonnet-4-6")
    )]
    #[case::exact("gpt-4o", Some("gpt-4o"))]
    #[case::router("a042a7be2fb64875abf123ef", None)]
    #[case::azure("azure/my-deployment", None)]
    #[case::case("hosted_vllm/model", None)]
    #[case::case_suffix("hosted_vllm/model-2026-01-01", None)]
    fn registration_preserves_write_and_inheritance(
        #[from(interpreter)] _interpreter: (),
        #[case] key: &str,
        #[case] inherited: Option<&str>,
    ) {
        Python::attach(|py| {
            let rows = dict(
                py,
                json!({
                    "o3-pro": {"litellm_provider":"openai"},
                    "gpt-4o": {"litellm_provider":"openai"},
                    "databricks/databricks-claude-opus-4": {"litellm_provider":"databricks"},
                    "bedrock/anthropic.claude-sonnet-4-6": {"litellm_provider":"bedrock"},
                    "hosted_vllm/Model": {"litellm_provider":"hosted_vllm"}
                }),
            );
            let before: Value = from_py(rows.as_any()).unwrap();
            let target = registration_target(&module(py), &rows, key, None).unwrap();
            assert_eq!(
                (target.write.as_str(), target.inherit_from.as_deref()),
                (key, inherited)
            );
            assert_eq!(from_py::<Value>(rows.as_any()).unwrap(), before);
        });
    }

    #[rstest]
    #[case::empty("")]
    #[case::missing("missing")]
    #[case::not_date("model-2026-01")]
    #[case::numeric("model-20260101")]
    #[case::finetune("ft:model:org")]
    #[case::unicode("未知-版本")]
    fn reports_r3_misses(#[from(interpreter)] _interpreter: (), #[case] model: &str) {
        Python::attach(|py| {
            let rows = dict(py, json!({"model": {}}));
            let ResolutionResult::NotFound(missing) =
                resolve_model(&module(py), &rows, model, None, None, None, None, false).unwrap()
            else {
                panic!("unexpected match")
            };
            assert_eq!(missing.tried, [model]);
        });
    }

    #[rstest]
    #[case::lowercase_then_price("price")]
    #[case::lowercase_then_info("model_info")]
    fn nested_writes_are_observed_by_the_next_read(
        #[from(interpreter)] _interpreter: (),
        #[case] read: &str,
    ) {
        Python::attach(|py| {
            let native = crate::native_module(py);
            let locals = PyDict::new(py);
            locals.set_item("n", native).unwrap();
            locals.set_item("read", read).unwrap();
            py.run(pyo3::ffi::c_str!(r#"
row = {"litellm_provider": "test", "input_cost_per_token": 2.0, "supports_vision": False}
models = {"Model": row}
usage = n.UsageInput({"input_cost_per_token": 3})
when = n.WhenInput(0)
assert n.resolve_model(models, "MODEL").key == "Model"
assert n.price(row, usage, when).total == 6.0
assert n.model_info(models, "Model").input_cost_per_token == 2.0
row["input_cost_per_token"] = 7.0
row["supports_vision"] = True
if read == "price":
    assert n.price(row, usage, when).total == 21.0
else:
    info = n.model_info(models, "Model")
    assert info.input_cost_per_token == 7.0
    assert info.capabilities["supports_vision"] is True
assert n.price({"litellm_provider": "request", "input_cost_per_token": 1.0}, usage, when).total == 3.0
assert row["input_cost_per_token"] == 7.0
assert n.price(row, usage, when).total == 21.0
"#), Some(&locals), None).unwrap();
        });
    }

    #[rstest]
    fn lowercase_cache_handles_removal_rebinding_and_invalidation(
        #[from(interpreter)] _interpreter: (),
    ) {
        Python::attach(|py| {
            let module = module(py);
            let rows = dict(py, json!({"Model": {}}));
            let resolve = |rows: &Bound<'_, PyDict>, key| match resolve_model(
                &module, rows, key, None, None, None, None, false,
            )
            .unwrap()
            {
                ResolutionResult::Found(found) => Some(found.key),
                ResolutionResult::NotFound(_) => None,
                ResolutionResult::Invalid(_) => {
                    panic!("default lookup must not validate provider values")
                }
            };
            assert_eq!(resolve(&rows, "MODEL").as_deref(), Some("Model"));
            rows.del_item("Model").unwrap();
            rows.set_item("mOdel", PyDict::new(py)).unwrap();
            assert_eq!(resolve(&rows, "MODEL").as_deref(), Some("mOdel"));
            rows.set_item("Another", PyDict::new(py)).unwrap();
            assert_eq!(resolve(&rows, "ANOTHER").as_deref(), Some("Another"));
            rows.del_item("Another").unwrap();
            rows.set_item("New", PyDict::new(py)).unwrap();
            assert_eq!(resolve(&rows, "NEW"), None);
            assert_eq!(resolve(&rows, "New").as_deref(), Some("New"));
            invalidate_model_cost_index(&module).unwrap();
            assert_eq!(resolve(&rows, "NEW").as_deref(), Some("New"));
            let other = dict(py, json!({"MODEL": {}}));
            assert_eq!(resolve(&other, "model").as_deref(), Some("MODEL"));
            assert_eq!(resolve(&rows, "MODEL").as_deref(), Some("mOdel"));
        });
    }

    #[rstest]
    fn dictionary_hooks_do_not_override_live_reads(#[from(interpreter)] _interpreter: ()) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    r#"
class HostileDict(dict):
    def __contains__(self, key): raise AssertionError("contains hook")
    def __getitem__(self, key): raise AssertionError("getitem hook")
    def __iter__(self): raise AssertionError("iter hook")
    def __len__(self): raise AssertionError("len hook")
    def keys(self): raise AssertionError("keys hook")
models = HostileDict({"Model": {"litellm_provider": "test", "input_cost_per_token": 2}})
assert n.resolve_model(models, "Model").key == "Model"
assert n.resolve_model(models, "MODEL").key == "Model"
assert n.registration_target(models, "Model").inherit_from == "Model"
assert n.registration_target(models, "Model", provider="test").inherit_from == "Model"
assert n.model_info(models, "Model").input_cost_per_token == 2
"#
                ),
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[rstest]
    #[case::wrong_provider("same", "azure", None)]
    #[case::bare_after_folded_candidate("model", "openai", Some("model"))]
    #[case::qualified_preference("qualified", "openai", Some("openai/qualified"))]
    #[case::case_sensitive("hosted_vllm/model", "hosted_vllm", None)]
    fn registration_binding_keeps_provider_and_case_constraints(
        #[from(interpreter)] _interpreter: (),
        #[case] key: &str,
        #[case] provider: &str,
        #[case] inherited: Option<&str>,
    ) {
        Python::attach(|py| {
            let rows = dict(
                py,
                json!({
                    "same": {"litellm_provider": "openai"},
                    "model": {"litellm_provider": "openai"},
                    "openai/Model": {"litellm_provider": "openai"},
                    "qualified": {"litellm_provider": "openai"},
                    "openai/qualified": {"litellm_provider": "openai"},
                    "hosted_vllm/Model": {"litellm_provider": "hosted_vllm"}
                }),
            );
            let before: Value = from_py(rows.as_any()).unwrap();
            let target = registration_target(&module(py), &rows, key, Some(provider)).unwrap();
            assert_eq!(
                (target.write.as_str(), target.inherit_from.as_deref()),
                (key, inherited)
            );
            assert_eq!(from_py::<Value>(rows.as_any()).unwrap(), before);
        });
    }

    #[rstest]
    #[case::scalar("42", "not_an_object")]
    #[case::missing_provider("{}", "missing_provider")]
    #[case::negative("{'litellm_provider':'test','input_cost_per_token':-1}", "invalid")]
    #[case::infinity(
        "{'litellm_provider':'test','input_cost_per_token':float('inf')}",
        "invalid"
    )]
    #[case::nan(
        "{'litellm_provider':'test','input_cost_per_token':float('nan')}",
        "invalid"
    )]
    #[case::object("{'litellm_provider':'test','x':object()}", "invalid")]
    #[case::bad_key("{'litellm_provider':'test',1:2}", "invalid")]
    #[case::bad_block("{'litellm_provider':'test','tiered_pricing':{}}", "invalid")]
    fn row_errors_are_values(
        #[from(interpreter)] _interpreter: (),
        #[case] expression: &str,
        #[case] kind: &str,
        #[values("generic", "sdk_tokens")] policy: &str,
    ) {
        Python::attach(|py| {
            let row = py
                .eval(&CString::new(expression).unwrap(), None, None)
                .unwrap();
            let map = PyDict::new(py);
            map.set_item("a~/b", &row).unwrap();
            let ModelInfoResult::Errors(errors) = model_info(&map, "a~/b").unwrap() else {
                panic!("expected invalid row")
            };
            assert_eq!(errors.errors[0].key, "a~/b");
            assert_eq!(errors.errors[0].kind, kind);
            let usage = UsageInput::new(&dict(py, json!({"input_cost_per_token": 1})), 0, false, 0)
                .unwrap();
            let PriceResult::Errors(errors) = price(
                &row,
                &usage,
                &WhenInput::new(0, "standard", "online").unwrap(),
                policy,
                None,
            )
            .unwrap() else {
                panic!("expected invalid row")
            };
            assert_eq!(errors.errors[0].key, "");
            assert_eq!(errors.errors[0].kind, kind);
            let TokenPricePresence::Errors(errors) = has_token_prices(&row, policy).unwrap() else {
                panic!("presence must retain validation errors")
            };
            assert_eq!(errors.errors[0].kind, kind);
        });
    }

    #[rstest]
    #[case::metadata(json!({"mode":"chat"}), false)]
    #[case::null(json!({"input_cost_per_token":null}), false)]
    #[case::zero(json!({"input_cost_per_token":0}), true)]
    #[case::output_only(json!({"output_cost_per_token":2}), true)]
    #[case::read_only(json!({"cache_read_input_token_cost":0}), true)]
    #[case::explicit_read(json!({"explicit_cache_read_input_token_cost":1}), true)]
    #[case::write_only(json!({"cache_creation_input_token_cost_above_1hr":1}), true)]
    #[case::twelve_hours(json!({"cache_creation_input_token_cost_above_12hr":1}), true)]
    #[case::reasoning_only(json!({"output_cost_per_reasoning_token":1}), true)]
    #[case::audio_only(json!({"output_cost_per_audio_token":1}), true)]
    #[case::image_token_only(json!({"input_cost_per_image_token":1}), true)]
    #[case::video_token_only(json!({"input_cost_per_video_token":1}), true)]
    #[case::bounded_tier(json!({"tiered_pricing":[{"range":[0,1],"output_cost_per_token":0}]}), true)]
    #[case::service_only(json!({"output_cost_per_token_priority":1}), true)]
    #[case::band_only(json!({"input_cost_per_token_above_200k_tokens":1}), true)]
    #[case::offpeak_only(json!({"off_peak_pricing":{"hours_utc":"01:00-02:00","cache_read_input_token_cost":0}}), true)]
    #[case::batch_only(json!({"input_cost_per_token_batches":0}), true)]
    #[case::decisions_only(json!({"output_cost_per_token_decisions":0}), true)]
    #[case::seconds(json!({"input_cost_per_second":0}), false)]
    #[case::images(json!({"output_cost_per_image":1}), false)]
    #[case::query_tier(json!({"tiered_pricing":[{"max_results_range":[0,10],"input_cost_per_query":1}]}), false)]
    fn explicit_token_presence_uses_validated_rates(
        #[from(interpreter)] _interpreter: (),
        #[case] fields: Value,
        #[case] expected: bool,
        #[values("generic", "sdk_tokens")] policy: &str,
    ) {
        Python::attach(|py| {
            let row = dict(py, fields);
            row.set_item("litellm_provider", "openai").unwrap();
            let before: Value = from_py(row.as_any()).unwrap();
            let TokenPricePresence::Present(present) =
                has_token_prices(row.as_any(), policy).unwrap()
            else {
                panic!("expected valid row")
            };
            assert_eq!(present, expected);
            assert_eq!(from_py::<Value>(row.as_any()).unwrap(), before);
        });
    }

    #[rstest]
    fn presence_binding_is_live_keyword_only_and_source_aware(
        #[from(interpreter)] _interpreter: (),
    ) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    r#"
import json
row = {"litellm_provider": "openai", "tiered_pricing": [{"range": [0, 1]}]}
assert n.has_token_prices(row) is False
assert n.has_token_prices(row, policy="sdk_tokens") is False
row["tiered_pricing"][0]["output_cost_per_token"] = 0
before = json.dumps(row)
assert n.has_token_prices(row) is True
assert n.has_token_prices(row, policy="sdk_tokens") is True
assert json.dumps(row) == before
row["output_cost_per_token"] = 1
assert isinstance(n.has_token_prices(row), n.RowErrors)
assert n.has_token_prices(row, policy="sdk_tokens") is True
row["tiered_pricing"][0]["output_cost_per_token"] = float("inf")
assert isinstance(n.has_token_prices(row, policy="sdk_tokens"), n.RowErrors)
row["tiered_pricing"][0]["output_cost_per_token"] = None
del row["output_cost_per_token"]
assert n.has_token_prices(row, policy="sdk_tokens") is False
try:
    n.has_token_prices(row, policy="unknown")
except ValueError as error:
    assert str(error) == "unknown pricing policy"
else:
    raise AssertionError("unknown policy accepted")
try:
    n.has_token_prices(row, "sdk_tokens")
except TypeError:
    pass
else:
    raise AssertionError("positional policy accepted")
"#
                ),
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[rstest]
    #[case::identity(vec![], 1.0)]
    #[case::speed(vec![6.0], 6.0)]
    #[case::combined(vec![2.0, 3.0], 6.0)]
    #[case::regional(vec![1.1], 1.1)]
    #[case::discount(vec![0.5], 0.5)]
    #[case::zero(vec![0.0], 0.0)]
    fn all_cached_usage_projects_unused_rates_without_fake_lines(
        #[from(interpreter)] _interpreter: (),
        #[case] factors: Vec<f64>,
        #[case] factor: f64,
    ) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            locals.set_item("factors", factors).unwrap();
            locals.set_item("factor", factor).unwrap();
            py.run(pyo3::ffi::c_str!(r#"
import json
row = {"litellm_provider": "anthropic", "input_cost_per_token": 1, "output_cost_per_token": 2,
       "cache_read_input_token_cost": 0.1, "cache_creation_input_token_cost": 1.25,
       "provider_specific_entry": {"fast": 6}, "regional_processing_uplift_multiplier_eu": 1.1}
before = json.dumps(row)
meters = ("input_cost_per_token", "output_cost_per_token", "cache_read_input_token_cost",
          "cache_creation_input_token_cost", "cache_creation_input_token_cost_above_1hr",
          "cache_creation_input_token_cost_above_12hr", "output_cost_per_reasoning_token",
          "cache_read_input_audio_token_cost")
usage = n.UsageInput({"cache_read_input_token_cost": 100, "input_cost_per_token": 0, "output_cost_per_token": 0}, prompt_tokens=100)
when = n.WhenInput(0)
evaluated = n.price_with_rates(row, usage, when, policy="sdk_tokens", project=meters, factors=factors)
assert isinstance(evaluated, n.PriceEvaluation)
assert isinstance(evaluated.pricing, n.Priced)
assert (evaluated.pricing.input, evaluated.pricing.output, evaluated.pricing.total) == (10 * factor, 0, 10 * factor)
assert len(evaluated.pricing.lines) == 1
line = evaluated.pricing.lines[0]
assert (line.meter, line.quantity, line.unit_price, line.amount, line.fallback_from) == (meters[2], 100, 0.1 * factor, 10 * factor, None)
assert tuple(rate.requested_meter for rate in evaluated.rates) == meters
assert tuple(rate.source_meter for rate in evaluated.rates) == (meters[0], meters[1], meters[2], meters[3], meters[3], meters[3], meters[1], meters[2])
assert tuple(rate.unit_price for rate in evaluated.rates) == (factor, 2 * factor, 0.1 * factor, 1.25 * factor, 1.25 * factor, 1.25 * factor, 2 * factor, 0.1 * factor)
default = n.price_with_rates(row, usage, when)
assert default.rates == []
assert default.pricing.total == n.price(row, usage, when).total == 10
assert json.dumps(row) == before
"#), Some(&locals), None).unwrap();
        });
    }

    #[rstest]
    #[case::tier_read(json!({"cache_read_input_token_cost": 9, "tiered_pricing": [{"range": [0,100], "input_cost_per_token": 1, "output_cost_per_token": 2}]}), "cache_read_input_token_cost", (100, false, "standard", "online", 0), Some("input_cost_per_token"), Some(1.0))]
    #[case::tier_hour(json!({"cache_creation_input_token_cost_above_1hr": 9, "tiered_pricing": [{"range": [0,100], "input_cost_per_token": 1, "output_cost_per_token": 2, "cache_creation_input_token_cost": 3}]}), "cache_creation_input_token_cost_above_1hr", (100, false, "standard", "online", 0), Some("cache_creation_input_token_cost"), Some(3.0))]
    #[case::tier_twelve_hours(json!({"cache_creation_input_token_cost_above_12hr": 9, "tiered_pricing": [{"range": [0,100], "input_cost_per_token": 1, "output_cost_per_token": 2, "cache_creation_input_token_cost": 3}]}), "cache_creation_input_token_cost_above_12hr", (100, false, "standard", "online", 0), Some("cache_creation_input_token_cost"), Some(3.0))]
    #[case::tier_reasoning(json!({"output_cost_per_reasoning_token": 5, "tiered_pricing": [{"range": [0,100], "input_cost_per_token": 2, "output_cost_per_token": 3}]}), "output_cost_per_reasoning_token", (100, false, "standard", "online", 0), Some("output_cost_per_token"), Some(3.0))]
    #[case::priority_reasoning(json!({"input_cost_per_token": 1, "output_cost_per_token": 2, "output_cost_per_reasoning_token": 3, "output_cost_per_token_priority": 4}), "output_cost_per_reasoning_token", (100, false, "priority", "online", 0), Some("output_cost_per_token"), Some(4.0))]
    #[case::priority_audio(json!({"input_cost_per_token": 1, "output_cost_per_token": 2, "output_cost_per_audio_token": 3, "output_cost_per_audio_token_priority": 4}), "output_cost_per_audio_token", (100, false, "priority", "online", 0), Some("output_cost_per_audio_token"), Some(3.0))]
    #[case::offpeak_input(json!({"input_cost_per_token": 2, "output_cost_per_token": 4, "off_peak_pricing": {"hours_utc": "00:00-06:00", "input_cost_per_token": 1}}), "input_cost_per_token", (100, false, "standard", "online", 3600), Some("input_cost_per_token"), Some(1.0))]
    #[case::peak_input(json!({"input_cost_per_token": 2, "output_cost_per_token": 4, "off_peak_pricing": {"hours_utc": "00:00-06:00", "input_cost_per_token": 1}}), "input_cost_per_token", (100, false, "standard", "online", 43200), Some("input_cost_per_token"), Some(2.0))]
    #[case::offpeak_preserves_tier_read(json!({"tiered_pricing": [{"range": [0,100], "input_cost_per_token": 2, "output_cost_per_token": 3}], "off_peak_pricing": {"hours_utc": "00:00-06:00", "input_cost_per_token": 1}}), "cache_read_input_token_cost", (100, false, "standard", "online", 3600), Some("input_cost_per_token"), Some(2.0))]
    #[case::inclusive_band(json!({"input_cost_per_token": 1, "output_cost_per_token": 2, "output_cost_per_token_above_200k_tokens": 4}), "output_cost_per_token", (200000, true, "standard", "online", 0), Some("output_cost_per_token"), Some(4.0))]
    #[case::exclusive_band(json!({"input_cost_per_token": 1, "output_cost_per_token": 2, "output_cost_per_token_above_200k_tokens": 4}), "output_cost_per_token", (200000, false, "standard", "online", 0), Some("output_cost_per_token"), Some(2.0))]
    #[case::batch(json!({"input_cost_per_token": 1, "output_cost_per_token": 2}), "output_cost_per_token", (100, false, "standard", "batch", 0), Some("output_cost_per_token"), Some(1.0))]
    #[case::null_tier_output(json!({"input_cost_per_token": 1, "output_cost_per_token": 6, "tiered_pricing": [{"range": [0,100], "input_cost_per_token": 2, "output_cost_per_token": null}]}), "output_cost_per_token", (100, false, "standard", "online", 0), None, None)]
    #[case::absent_tier_output(json!({"input_cost_per_token": 1, "output_cost_per_token": 6, "tiered_pricing": [{"range": [0,100], "input_cost_per_token": 2}]}), "output_cost_per_token", (100, false, "standard", "online", 0), Some("output_cost_per_token"), Some(6.0))]
    #[case::zero_tier_output(json!({"input_cost_per_token": 1, "output_cost_per_token": 6, "tiered_pricing": [{"range": [0,100], "input_cost_per_token": 2, "output_cost_per_token": 0}]}), "output_cost_per_token", (100, false, "standard", "online", 0), Some("output_cost_per_token"), Some(0.0))]
    #[case::batch_reasoning_text(json!({"input_cost_per_token":1,"output_cost_per_token":2,"output_cost_per_reasoning_token":8}), "output_cost_per_reasoning_token", (100,false,"standard","batch",3600), Some("output_cost_per_token"), Some(1.0))]
    #[case::batch_audio_text(json!({"output_cost_per_token":2,"output_cost_per_audio_token":8,"output_cost_per_audio_token_batches":6}), "output_cost_per_audio_token", (100,false,"standard","batch",3600), Some("output_cost_per_token"), Some(1.0))]
    #[case::batch_video_text(json!({"output_cost_per_token":2,"output_cost_per_video_token":8}), "output_cost_per_video_token", (100,false,"standard","batch",3600), Some("output_cost_per_token"), Some(1.0))]
    #[case::batch_reasoning_missing_text(json!({"output_cost_per_reasoning_token":8}), "output_cost_per_reasoning_token", (100,false,"standard","batch",3600), None, None)]
    #[case::batch_output_zero(json!({"output_cost_per_token":2,"output_cost_per_token_batches":0,"output_cost_per_reasoning_token":8}), "output_cost_per_reasoning_token", (100,false,"standard","batch",3600), Some("output_cost_per_token"), Some(0.0))]
    #[case::batch_image_online(json!({"output_cost_per_token":2,"output_cost_per_image_token":6}), "output_cost_per_image_token", (100,false,"standard","batch",3600), Some("output_cost_per_image_token"), Some(3.0))]
    #[case::batch_image_zero(json!({"output_cost_per_token":2,"output_cost_per_image_token":6,"output_cost_per_image_token_batches":0}), "output_cost_per_image_token", (100,false,"standard","batch",3600), Some("output_cost_per_image_token"), Some(0.0))]
    #[case::batch_image_text_fallback(json!({"output_cost_per_token":2,"output_cost_per_token_batches":3}), "output_cost_per_image_token", (100,false,"standard","batch",3600), Some("output_cost_per_token"), Some(3.0))]
    #[case::batch_half_audio_input(json!({"input_cost_per_token":1,"input_cost_per_audio_token":8,"input_cost_per_audio_token_batches":6}), "input_cost_per_audio_token", (100,false,"standard","batch",3600), Some("input_cost_per_token"), Some(0.5))]
    #[case::batch_half_video_input(json!({"input_cost_per_token":1,"input_cost_per_video_token":8}), "input_cost_per_video_token", (100,false,"standard","batch",3600), Some("input_cost_per_token"), Some(0.5))]
    #[case::batch_half_image_input(json!({"input_cost_per_token":1,"input_cost_per_image_token":8}), "input_cost_per_image_token", (100,false,"standard","batch",3600), Some("input_cost_per_token"), Some(0.5))]
    #[case::batch_explicit_audio_input(json!({"input_cost_per_token":1,"input_cost_per_token_batches":4,"input_cost_per_audio_token_batches":6}), "input_cost_per_audio_token", (100,false,"standard","batch",3600), Some("input_cost_per_audio_token"), Some(6.0))]
    #[case::batch_explicit_video_input(json!({"input_cost_per_token_batches":4,"input_cost_per_video_token_batches":6}), "input_cost_per_video_token", (100,false,"standard","batch",3600), Some("input_cost_per_video_token"), Some(6.0))]
    #[case::batch_explicit_audio_fallback(json!({"input_cost_per_token_batches":4,"input_cost_per_audio_token":8}), "input_cost_per_audio_token", (100,false,"standard","batch",3600), Some("input_cost_per_token"), Some(4.0))]
    #[case::batch_zero_input(json!({"input_cost_per_token":1,"input_cost_per_token_batches":0,"input_cost_per_audio_token":8}), "input_cost_per_audio_token", (100,false,"standard","batch",3600), Some("input_cost_per_token"), Some(0.0))]
    #[case::batch_read_ignores_online(json!({"input_cost_per_token":1,"input_cost_per_token_batches":4,"cache_read_input_token_cost":0.2}), "cache_read_input_token_cost", (100,false,"standard","batch",3600), Some("input_cost_per_token"), Some(4.0))]
    #[case::batch_explicit_read(json!({"input_cost_per_token_batches":4,"cache_read_input_token_cost_batches":0.3}), "cache_read_input_token_cost", (100,false,"standard","batch",3600), Some("cache_read_input_token_cost"), Some(0.3))]
    #[case::batch_zero_read(json!({"input_cost_per_token_batches":4,"cache_read_input_token_cost_batches":0}), "cache_read_input_token_cost", (100,false,"standard","batch",3600), Some("cache_read_input_token_cost"), Some(0.0))]
    #[case::batch_half_read_ignores_batch_read(json!({"input_cost_per_token":1,"cache_read_input_token_cost":0.2,"cache_read_input_token_cost_batches":3}), "cache_read_input_token_cost", (100,false,"standard","batch",3600), Some("cache_read_input_token_cost"), Some(0.1))]
    #[case::batch_half_explicit_mode(json!({"input_cost_per_token":1,"cache_read_input_token_cost":0.2,"explicit_cache_read_input_token_cost":9}), "explicit_cache_read_input_token_cost", (100,false,"standard","batch",3600), Some("cache_read_input_token_cost"), Some(0.1))]
    #[case::batch_audio_read_without_text_read(json!({"input_cost_per_token_batches":4,"cache_read_input_audio_token_cost_batches":3}), "cache_read_input_audio_token_cost", (100,false,"standard","batch",3600), Some("input_cost_per_token"), Some(4.0))]
    #[case::batch_audio_read(json!({"input_cost_per_token_batches":4,"cache_read_input_token_cost_batches":2,"cache_read_input_audio_token_cost_batches":3}), "cache_read_input_audio_token_cost", (100,false,"standard","batch",3600), Some("cache_read_input_audio_token_cost"), Some(3.0))]
    #[case::batch_half_hourly_write(json!({"input_cost_per_token":1,"cache_creation_input_token_cost":0.2,"cache_creation_input_token_cost_above_1hr":9}), "cache_creation_input_token_cost_above_1hr", (100,false,"standard","batch",3600), Some("cache_creation_input_token_cost"), Some(0.1))]
    #[case::batch_hourly_write(json!({"input_cost_per_token_batches":4,"cache_creation_input_token_cost_above_1hr_batches":3}), "cache_creation_input_token_cost_above_1hr", (100,false,"standard","batch",3600), Some("cache_creation_input_token_cost_above_1hr"), Some(3.0))]
    #[case::batch_zero_hourly_write(json!({"input_cost_per_token_batches":4,"cache_creation_input_token_cost_above_1hr_batches":0}), "cache_creation_input_token_cost_above_1hr", (100,false,"standard","batch",3600), Some("cache_creation_input_token_cost_above_1hr"), Some(0.0))]
    #[case::batch_twelve_write_fallback(json!({"input_cost_per_token_batches":4,"cache_creation_input_token_cost_batches":2}), "cache_creation_input_token_cost_above_12hr", (100,false,"standard","batch",3600), Some("cache_creation_input_token_cost"), Some(2.0))]
    #[case::batch_exclusive_threshold(json!({"input_cost_per_token_batches":4,"input_cost_per_token_above_100_tokens_batches":6}), "input_cost_per_token", (100,false,"standard","batch",3600), Some("input_cost_per_token"), Some(4.0))]
    #[case::batch_inclusive_threshold(json!({"input_cost_per_token_batches":4,"input_cost_per_token_above_100_tokens_batches":6}), "input_cost_per_token", (100,true,"standard","batch",3600), Some("input_cost_per_token"), Some(6.0))]
    #[case::batch_per_meter_threshold(json!({"input_cost_per_token_above_50_tokens_batches":6,"output_cost_per_token_batches":3,"output_cost_per_token_above_150_tokens_batches":7}), "output_cost_per_token", (100,false,"standard","batch",3600), Some("output_cost_per_token"), Some(3.0))]
    #[case::batch_offpeak_and_service_isolated(json!({"input_cost_per_token":1,"input_cost_per_token_priority":8,"off_peak_pricing":{"hours_utc":"00:00-06:00","input_cost_per_token":9}}), "input_cost_per_token", (100,false,"priority","batch",3600), Some("input_cost_per_token"), Some(0.5))]
    fn projection_uses_the_actual_token_selector_and_context(
        #[from(interpreter)] _interpreter: (),
        #[case] fields: Value,
        #[case] meter: &str,
        #[case] context: (u64, bool, &str, &str, i64),
        #[case] source: Option<&str>,
        #[case] rate: Option<f64>,
    ) {
        let (prompt, inclusive, tier, channel, at) = context;
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            let row = dict(py, fields);
            row.set_item("litellm_provider", "openai").unwrap();
            locals.set_item("row", row).unwrap();
            locals.set_item("meter", meter).unwrap();
            locals.set_item("prompt", prompt).unwrap();
            locals.set_item("inclusive", inclusive).unwrap();
            locals.set_item("tier", tier).unwrap();
            locals.set_item("channel", channel).unwrap();
            locals.set_item("at", at).unwrap();
            locals.set_item("source", source).unwrap();
            locals.set_item("rate", rate).unwrap();
            py.run(pyo3::ffi::c_str!(r#"
import json
before = json.dumps(row)
when = n.WhenInput(at, tier=tier, channel=channel)
unused = n.price_with_rates(row, n.UsageInput({}, prompt_tokens=prompt, inclusive_threshold=inclusive), when, policy="sdk_tokens", project=[meter], factors=[2,3])
assert isinstance(unused.pricing, n.Priced)
assert unused.pricing.total == 0 and unused.pricing.lines == []
assert len(unused.rates) == 1
selected = unused.rates[0]
assert (selected.requested_meter, selected.source_meter, selected.unit_price) == (meter, source, None if rate is None else rate * 6)
used = n.price_with_rates(row, n.UsageInput({meter: 1}, prompt_tokens=prompt, inclusive_threshold=inclusive), when, policy="sdk_tokens", project=[meter], factors=[2,3])
if rate is None:
    assert isinstance(used.pricing, n.Unpriced)
    assert used.pricing.missing == [meter] and used.pricing.partial.total == 0
else:
    assert isinstance(used.pricing, n.Priced)
    line = used.pricing.lines[0]
    assert (line.meter, line.unit_price, line.amount, line.quantity, line.fallback_from) == (source, rate * 6, rate * 6, 1, None if meter == source else meter)
    assert used.pricing.total == rate * 6
assert json.dumps(row) == before
"#), Some(&locals), None).unwrap();
        });
    }

    #[rstest]
    #[case::negative("[-1]")]
    #[case::nan("[float('nan')]")]
    #[case::infinite("[float('inf')]")]
    #[case::zero_does_not_hide_invalid("[0, float('inf')]")]
    #[case::boolean("[True]")]
    #[case::string("['6']")]
    #[case::not_sequence("6")]
    #[case::null("None")]
    #[case::overflow("[1e308, 1e308]")]
    fn projection_binding_rejects_invalid_factors(
        #[from(interpreter)] _interpreter: (),
        #[case] factors: &str,
    ) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            locals.set_item("factors", factors).unwrap();
            py.run(pyo3::ffi::c_str!(r#"
try:
    n.price_with_rates({"litellm_provider": "openai", "input_cost_per_token": 1}, n.UsageInput({"input_cost_per_token": 1}), n.WhenInput(0), factors=eval(factors))
except (ValueError, TypeError):
    pass
else:
    raise AssertionError("invalid factors accepted")
"#), Some(&locals), None).unwrap();
        });
    }

    #[rstest]
    #[case::absent(json!({}), None)]
    #[case::null(json!({"output_cost_per_token": null}), None)]
    #[case::zero(json!({"output_cost_per_token": 0}), Some(0.0))]
    #[case::priced(json!({"output_cost_per_token": 2}), Some(2.0))]
    fn unused_missing_projection_does_not_make_actual_cache_billing_unpriced(
        #[from(interpreter)] _interpreter: (),
        #[case] fields: Value,
        #[case] expected: Option<f64>,
        #[values(0.0, 1.0, 1.1)] factor: f64,
    ) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            let row = dict(py, fields);
            row.set_item("litellm_provider", "openai").unwrap();
            row.set_item("input_cost_per_token", 1).unwrap();
            row.set_item("cache_read_input_token_cost", 0.1).unwrap();
            locals.set_item("row", row).unwrap();
            locals
                .set_item("expected", expected.map(|rate| rate * factor))
                .unwrap();
            locals.set_item("factor", factor).unwrap();
            py.run(pyo3::ffi::c_str!(r#"
usage = n.UsageInput({"cache_read_input_token_cost": 100, "output_cost_per_token": 0}, prompt_tokens=100)
result = n.price_with_rates(row, usage, n.WhenInput(0), policy="sdk_tokens", project=["output_cost_per_token"], factors=[factor])
assert isinstance(result.pricing, n.Priced)
assert result.pricing.total == 10 * factor
assert len(result.pricing.lines) == 1
assert result.rates[0].unit_price == expected
assert result.rates[0].source_meter == (None if expected is None else "output_cost_per_token")
"#), Some(&locals), None).unwrap();
        });
    }

    #[rstest]
    fn projected_rates_validate_live_rows_and_keyword_binding(
        #[from(interpreter)] _interpreter: (),
    ) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            py.run(pyo3::ffi::c_str!(r#"
import inspect
import gc
signature = inspect.signature(n.price_with_rates)
assert signature.parameters['project'].default == ()
assert signature.parameters['factors'].default == ()
row = {"litellm_provider": "openai", "input_cost_per_token": 1, "output_cost_per_token": 2, "tiered_pricing": [{"range": [0,100], "input_cost_per_token": 3, "output_cost_per_token": 4}]}
usage = n.UsageInput({"input_cost_per_token": 10}, prompt_tokens=10)
when = n.WhenInput(0)
assert isinstance(n.price_with_rates(row, usage, when), n.RowErrors)
first = n.price_with_rates(row, usage, when, policy="sdk_tokens", project=["output_cost_per_token"])
assert gc.get_referents(first) == [first.pricing]
assert first.pricing.total == 30 and first.rates[0].unit_price == 4
row["tiered_pricing"][0]["output_cost_per_token"] = 0
assert n.price_with_rates(row, usage, when, policy="sdk_tokens", project=["output_cost_per_token"]).rates[0].unit_price == 0
row["tiered_pricing"][0]["input_cost_per_token"] = None
partial = n.price_with_rates(row, usage, when, policy="sdk_tokens", project=["input_cost_per_token", "output_cost_per_token"], factors=[0])
assert isinstance(partial.pricing, n.Unpriced)
assert partial.pricing.missing == ["input_cost_per_token"] and partial.pricing.partial.total == 0
assert partial.rates[0].unit_price is None and partial.rates[1].unit_price == 0
row["tiered_pricing"][0]["output_cost_per_token"] = "bad"
assert isinstance(n.price_with_rates(row, usage, when, policy="sdk_tokens"), n.RowErrors)
try:
    n.price_with_rates(row, usage, when, policy="unknown")
except ValueError as error:
    assert str(error) == "unknown pricing policy"
else:
    raise AssertionError("unknown policy accepted")
try:
    n.price_with_rates(row, usage, when, "sdk_tokens")
except TypeError:
    pass
else:
    raise AssertionError("positional policy accepted")
try:
    n.price_with_rates({}, usage, when, project=["input_cost_per_image"])
except ValueError as error:
    assert str(error) == "only token meters can be projected"
else:
    raise AssertionError("non-token projection accepted")
"#), Some(&locals), None).unwrap();
        });
    }

    #[rstest]
    #[case::base(json!({"input_cost_per_token":2}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(1.0))]
    #[case::null_read(json!({"input_cost_per_token":2,"cache_read_input_token_cost":null}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(1.0))]
    #[case::zero_read(json!({"input_cost_per_token":2,"cache_read_input_token_cost":0}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(0.0))]
    #[case::explicit_read(json!({"input_cost_per_token":2,"cache_read_input_token_cost":0.25}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(0.25))]
    #[case::null_input(json!({"input_cost_per_token":null}), "cache_read_input_token_cost", (100,false,"standard","online",3600), None, None)]
    #[case::metadata_only(json!({"mode":"chat"}), "cache_read_input_token_cost", (100,false,"standard","online",3600), None, None)]
    #[case::string_input(json!({"input_cost_per_token":"2"}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(1.0))]
    #[case::zero_input(json!({"input_cost_per_token":0}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(0.0))]
    #[case::offpeak_input(json!({"input_cost_per_token":2,"off_peak_pricing":{"hours_utc":"00:00-06:00","input_cost_per_token":6}}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(3.0))]
    #[case::offpeak_only_input(json!({"off_peak_pricing":{"hours_utc":"00:00-06:00","input_cost_per_token":6}}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("input_cost_per_token"), Some(6.0))]
    #[case::offpeak_no_input(json!({"input_cost_per_token":2,"off_peak_pricing":{"hours_utc":"00:00-06:00","output_cost_per_token":6}}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(1.0))]
    #[case::offpeak_null_input(json!({"input_cost_per_token":2,"off_peak_pricing":{"hours_utc":"00:00-06:00","input_cost_per_token":null,"output_cost_per_token":6}}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(1.0))]
    #[case::offpeak_null_read(json!({"input_cost_per_token":2,"off_peak_pricing":{"hours_utc":"00:00-06:00","input_cost_per_token":6,"cache_read_input_token_cost":null}}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(1.0))]
    #[case::offpeak_zero_read(json!({"input_cost_per_token":2,"off_peak_pricing":{"hours_utc":"00:00-06:00","input_cost_per_token":6,"cache_read_input_token_cost":0}}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(0.0))]
    #[case::offpeak_zero_input(json!({"input_cost_per_token":2,"off_peak_pricing":{"hours_utc":"00:00-06:00","input_cost_per_token":0}}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(0.0))]
    #[case::offpeak_end(json!({"input_cost_per_token":2,"off_peak_pricing":{"hours_utc":"00:00-06:00","input_cost_per_token":6}}), "cache_read_input_token_cost", (100,false,"standard","online",21600), Some("cache_read_input_token_cost"), Some(1.0))]
    #[case::zero_flat_blocks_offpeak(json!({"input_cost_per_token":2,"cache_read_input_token_cost":0,"off_peak_pricing":{"hours_utc":"00:00-06:00","input_cost_per_token":6}}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(0.0))]
    #[case::tier_fallback(json!({"input_cost_per_token":2,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":3}]}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("input_cost_per_token"), Some(3.0))]
    #[case::null_tier_read(json!({"input_cost_per_token":2,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":3,"cache_read_input_token_cost":null}]}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("input_cost_per_token"), Some(3.0))]
    #[case::zero_tier_read(json!({"input_cost_per_token":2,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":3,"cache_read_input_token_cost":0}]}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(0.0))]
    #[case::tier_only(json!({"tiered_pricing":[{"range":[0,100],"input_cost_per_token":3}]}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("input_cost_per_token"), Some(3.0))]
    #[case::zero_prompt(json!({"input_cost_per_token":2,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":3}]}), "cache_read_input_token_cost", (0,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(1.0))]
    #[case::tier_offpeak_overlay(json!({"input_cost_per_token":2,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":4,"cache_read_input_token_cost":0.25}],"off_peak_pricing":{"hours_utc":"00:00-06:00","input_cost_per_token":6}}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(3.0))]
    #[case::tier_offpeak_null_blocks(json!({"input_cost_per_token":2,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":4}],"off_peak_pricing":{"hours_utc":"00:00-06:00","input_cost_per_token":6,"cache_read_input_token_cost":null}}), "cache_read_input_token_cost", (100,false,"standard","online",3600), Some("input_cost_per_token"), Some(4.0))]
    #[case::above_threshold_input(json!({"input_cost_per_token":2,"input_cost_per_token_above_100_tokens":8}), "cache_read_input_token_cost", (101,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(1.0))]
    #[case::inclusive_threshold_input(json!({"input_cost_per_token":2,"input_cost_per_token_above_100_tokens":8}), "cache_read_input_token_cost", (100,true,"standard","online",3600), Some("cache_read_input_token_cost"), Some(1.0))]
    #[case::priority_input(json!({"input_cost_per_token":2,"input_cost_per_token_priority":8}), "cache_read_input_token_cost", (100,false,"priority","online",3600), Some("cache_read_input_token_cost"), Some(1.0))]
    #[case::audio_fallback(json!({"input_cost_per_token":2}), "cache_read_input_audio_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(1.0))]
    #[case::explicit_cache_fallback(json!({"input_cost_per_token":2}), "explicit_cache_read_input_token_cost", (100,false,"standard","online",3600), Some("cache_read_input_token_cost"), Some(1.0))]
    #[case::write_unchanged(json!({"input_cost_per_token":2}), "cache_creation_input_token_cost_above_1hr", (100,false,"standard","online",3600), Some("input_cost_per_token"), Some(2.0))]
    #[case::batch_ignores_default(json!({"input_cost_per_token":2}), "cache_read_input_token_cost", (100,false,"standard","batch",3600), Some("input_cost_per_token"), Some(1.0))]
    fn caller_cache_defaults_share_actual_and_projected_selection(
        #[from(interpreter)] _interpreter: (),
        #[case] fields: Value,
        #[case] meter: &str,
        #[case] context: (u64, bool, &str, &str, i64),
        #[case] source: Option<&str>,
        #[case] rate: Option<f64>,
    ) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            let row = dict(py, fields);
            row.set_item("litellm_provider", "openai").unwrap();
            locals.set_item("row", row).unwrap();
            locals.set_item("meter", meter).unwrap();
            locals.set_item("context", context).unwrap();
            locals.set_item("source", source).unwrap();
            locals.set_item("rate", rate).unwrap();
            py.run(pyo3::ffi::c_str!(r#"
import json
before = json.dumps(row)
present = n.has_token_prices(row, policy="sdk_tokens")
prompt, inclusive, tier, channel, at = context
when = n.WhenInput(at, tier=tier, channel=channel)
usage = n.UsageInput({meter: 2}, prompt_tokens=prompt, inclusive_threshold=inclusive)
priced = n.price(row, usage, when, policy="sdk_tokens", cache_read_default_ratio=0.5)
evaluated = n.price_with_rates(row, usage, when, policy="sdk_tokens", project=[meter], factors=[2,3], cache_read_default_ratio=0.5)
unused = n.price_with_rates(row, n.UsageInput({}, prompt_tokens=prompt, inclusive_threshold=inclusive), when, policy="sdk_tokens", project=[meter], factors=[2,3], cache_read_default_ratio=0.5)
assert isinstance(unused.pricing, n.Priced) and unused.pricing.total == 0 and unused.pricing.lines == []
assert (unused.rates[0].source_meter, unused.rates[0].unit_price) == (source, None if rate is None else rate * 6)
assert (evaluated.rates[0].source_meter, evaluated.rates[0].unit_price) == (source, None if rate is None else rate * 6)
if rate is None:
    assert isinstance(priced, n.Unpriced) and priced.missing == [meter] and priced.partial.total == 0
    assert isinstance(evaluated.pricing, n.Unpriced) and evaluated.pricing.missing == [meter]
else:
    assert priced.total == rate * 2 and evaluated.pricing.total == rate * 12
    line = evaluated.pricing.lines[0]
    assert (line.meter, line.fallback_from, line.unit_price, line.quantity, line.amount) == (source, None if source == meter else meter, rate * 6, 2, rate * 12)
assert n.has_token_prices(row, policy="sdk_tokens") == present
assert json.dumps(row) == before
"#), Some(&locals), None).unwrap();
        });
    }

    #[rstest]
    #[case::negative("-1")]
    #[case::nan("float('nan')")]
    #[case::infinite("float('inf')")]
    #[case::boolean("True")]
    #[case::string("'0.5'")]
    #[case::sequence("[0.5]")]
    #[case::object("{}")]
    #[case::overflow("1e308")]
    fn invalid_cache_default_is_rejected(
        #[from(interpreter)] _interpreter: (),
        #[case] argument: &str,
        #[values("price", "price_with_rates")] entry: &str,
    ) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            locals.set_item("argument", argument).unwrap();
            locals.set_item("entry", entry).unwrap();
            py.run(pyo3::ffi::c_str!(r#"
try:
    getattr(n, entry)({"litellm_provider":"openai", "input_cost_per_token":2}, n.UsageInput({}), n.WhenInput(0), policy="sdk_tokens", cache_read_default_ratio=eval(argument))
except (TypeError, ValueError):
    pass
else:
    raise AssertionError("invalid default accepted")
"#), Some(&locals), None).unwrap();
        });
    }

    #[rstest]
    fn cache_defaults_remain_opt_in_live_and_do_not_hide_invalid_rows(
        #[from(interpreter)] _interpreter: (),
    ) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            py.run(pyo3::ffi::c_str!(r#"
row = {"litellm_provider":"openai", "input_cost_per_token":2, "off_peak_pricing":{"hours_utc":"00:00-06:00", "input_cost_per_token":6}}
usage = n.UsageInput({"cache_read_input_token_cost":2})
when = n.WhenInput(3600)
assert n.price(row, usage, when).total == n.price(row, usage, when, policy="generic", cache_read_default_ratio=None).total == 12
assert n.price(row, usage, when, policy="sdk_tokens").total == n.price(row, usage, when, policy="sdk_tokens", cache_read_default_ratio=None).total == 12
assert n.price(row, usage, when, policy="sdk_tokens", cache_read_default_ratio=0.5).total == 6
row["off_peak_pricing"]["cache_read_input_token_cost"] = None
assert n.price(row, usage, when, policy="sdk_tokens", cache_read_default_ratio=0.5).total == 2
row["off_peak_pricing"]["cache_read_input_token_cost"] = 0
assert n.price_with_rates(row, usage, when, policy="sdk_tokens", cache_read_default_ratio=0.5).pricing.total == 0
row["off_peak_pricing"]["cache_read_input_token_cost"] = "invalid"
assert isinstance(n.price(row, usage, when, policy="sdk_tokens", cache_read_default_ratio=0.5), n.RowErrors)
assert isinstance(n.price_with_rates(row, usage, when, policy="sdk_tokens", cache_read_default_ratio=0.5), n.RowErrors)
missing = {"litellm_provider":"openai", "input_cost_per_token":None}
assert isinstance(n.price(missing, usage, when, policy="sdk_tokens", cache_read_default_ratio=0), n.Unpriced)
assert n.has_token_prices(missing, policy="sdk_tokens") is False
known = {"litellm_provider":"openai", "input_cost_per_token":2}
assert n.price(known, usage, when, policy="sdk_tokens", cache_read_default_ratio=0).total == 0
assert n.price_with_rates(known, usage, when, policy="sdk_tokens", cache_read_default_ratio=0, project=["cache_read_input_token_cost"], factors=[6]).rates[0].unit_price == 0
try:
    n.price(known, usage, when, cache_read_default_ratio=0.5)
except ValueError:
    pass
else:
    raise AssertionError("generic default option accepted")
try:
    n.has_token_prices(known, cache_read_default_ratio=0.5)
except TypeError:
    pass
else:
    raise AssertionError("presence accepts a derived default")
"#), Some(&locals), None).unwrap();
        });
    }

    #[rstest]
    #[case::online("online", "sdk_tokens", Some(30.0))]
    #[case::generic("online", "generic", None)]
    #[case::batch("batch", "sdk_tokens", Some(5.0))]
    #[case::batch_generic("batch", "generic", None)]
    #[case::decisions("decisions", "sdk_tokens", None)]
    fn conflicting_flat_and_token_tier_is_only_admitted_for_sdk_token_channels(
        #[from(interpreter)] _interpreter: (),
        #[case] channel: &str,
        #[case] policy: &str,
        #[case] expected: Option<f64>,
    ) {
        Python::attach(|py| {
            let row = dict(
                py,
                json!({"litellm_provider":"openai","input_cost_per_token":1,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":3}]}),
            );
            let usage =
                UsageInput::new(&dict(py, json!({"input_cost_per_token":10})), 10, false, 0)
                    .unwrap();
            let result = price(
                row.as_any(),
                &usage,
                &WhenInput::new(0, "standard", channel).unwrap(),
                policy,
                None,
            )
            .unwrap();
            match result {
                PriceResult::Priced(priced) => {
                    assert_eq!(Some(priced.total), expected);
                }
                PriceResult::Errors(errors) => {
                    assert_eq!(expected, None);
                    assert_eq!(errors.errors[0].kind, "duplicate_price");
                }
                PriceResult::Unpriced(_) => panic!("unexpected missing price"),
            }
        });
    }

    #[rstest]
    #[case::flat_and_tier("flat-and-tier", json!({"input_cost_per_token":1,"output_cost_per_token":2,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":3,"output_cost_per_token":4}]}), (10,json!({}),json!({"input_cost_per_token":10,"output_cost_per_token":10})), (30.0,40.0))]
    #[case::tier_read("tier-read", json!({"cache_read_input_token_cost":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"output_cost_per_token":2}]}), (100,json!({"cached_tokens":40}),json!({"input_cost_per_token":60,"cache_read_input_token_cost":40,"output_cost_per_token":10})), (100.0,20.0))]
    #[case::tier_hour("tier-hour", json!({"cache_creation_input_token_cost_above_1hr":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"output_cost_per_token":2,"cache_creation_input_token_cost":3}]}), (100,json!({"cache_creation_tokens":20,"cache_creation_token_details":{"ephemeral_5m_input_tokens":10,"ephemeral_1h_input_tokens":10}}),json!({"input_cost_per_token":80,"cache_creation_input_token_cost":10,"cache_creation_input_token_cost_above_1hr":10,"output_cost_per_token":10})), (140.0,20.0))]
    #[case::zero_tier("zero-tier", json!({"input_cost_per_token":1,"output_cost_per_token":2,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":0,"output_cost_per_token":0}]}), (10,json!({}),json!({"input_cost_per_token":10,"output_cost_per_token":10})), (0.0,0.0))]
    #[case::output_fallback("output-fallback", json!({"input_cost_per_token":1,"output_cost_per_token":2,"output_cost_per_token_priority":8,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":3}]}), (10,json!({}),json!({"input_cost_per_token":10,"output_cost_per_token":10})), (30.0,20.0))]
    #[case::zero_prompt("zero-prompt", json!({"input_cost_per_token":1,"output_cost_per_token":2,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":3,"output_cost_per_token":4}]}), (0,json!({}),json!({"output_cost_per_token":10})), (0.0,20.0))]
    #[case::no_input("no-input", json!({"input_cost_per_token":1,"output_cost_per_token":2,"tiered_pricing":[{"range":[0,100],"output_cost_per_token":4}]}), (10,json!({}),json!({"input_cost_per_token":10,"output_cost_per_token":10})), (10.0,20.0))]
    #[case::offpeak_hour("offpeak-hour", json!({"cache_creation_input_token_cost_above_1hr":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"output_cost_per_token":2,"cache_creation_input_token_cost":3}],"off_peak_pricing":{"hours_utc":"00:00-00:00","cache_creation_input_token_cost":0}}), (100,json!({"cache_creation_tokens":20,"cache_creation_token_details":{"ephemeral_5m_input_tokens":10,"ephemeral_1h_input_tokens":10}}),json!({"input_cost_per_token":80,"cache_creation_input_token_cost":10,"cache_creation_input_token_cost_above_1hr":10,"output_cost_per_token":10})), (80.0,20.0))]
    #[case::explicit_hour("explicit-hour", json!({"cache_creation_input_token_cost_above_1hr":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"output_cost_per_token":2,"cache_creation_input_token_cost":3,"cache_creation_input_token_cost_above_1hr":4}],"off_peak_pricing":{"hours_utc":"00:00-00:00","cache_creation_input_token_cost":0}}), (100,json!({"cache_creation_tokens":20,"cache_creation_token_details":{"ephemeral_5m_input_tokens":10,"ephemeral_1h_input_tokens":10}}),json!({"input_cost_per_token":80,"cache_creation_input_token_cost":10,"cache_creation_input_token_cost_above_1hr":10,"output_cost_per_token":10})), (120.0,20.0))]
    fn sdk_tier_cache_matches_public_raw_and_shaped(
        #[from(interpreter)] _interpreter: (),
        #[case] name: &str,
        #[case] fields: Value,
        #[case] usage: (u64, Value, Value),
        #[case] expected: (f64, f64),
        #[values("standard", "priority")] tier: &str,
    ) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            let row = dict(py, fields);
            row.set_item("litellm_provider", "openai").unwrap();
            row.set_item("mode", "chat").unwrap();
            locals.set_item("row", &row).unwrap();
            locals.set_item("name", name).unwrap();
            locals.set_item("tier", tier).unwrap();
            locals.set_item("prompt", usage.0).unwrap();
            locals.set_item("details", dict(py, usage.1)).unwrap();
            locals.set_item("quantities", dict(py, usage.2)).unwrap();
            locals.set_item("expected", expected).unwrap();
            py.run(pyo3::ffi::c_str!(r#"
import json
import litellm
before = json.dumps(row)
usage = n.UsageInput(quantities, prompt_tokens=prompt)
when = n.WhenInput(0, tier=tier)
raw = n.price(row, usage, when, policy="sdk_tokens")
assert isinstance(raw, n.Priced)
assert (raw.input, raw.output) == expected
model = f"openai/p2d-followup-{name}-{tier}"
litellm.model_cost[model] = row
try:
    kwargs = dict(model=model, custom_llm_provider="openai", service_tier=tier)
    public_usage = litellm.Usage(prompt_tokens=prompt, completion_tokens=10, prompt_tokens_details=details)
    public = litellm.cost_per_token(prompt_tokens=prompt, completion_tokens=10, usage_object=public_usage, **kwargs)
    assert public == expected, (public, expected)
    assert litellm.completion_cost(completion_response=litellm.ModelResponse(model=model, usage=public_usage), **kwargs) == raw.total
    info = litellm.get_model_info(model=model, custom_llm_provider="openai")
    shaped = n.price(info, usage, when, policy="sdk_tokens")
    assert isinstance(shaped, n.Priced)
    assert (shaped.input, shaped.output, shaped.total) == (raw.input, raw.output, raw.total)
    assert n.has_token_prices(info, policy="sdk_tokens") is True
finally:
    del litellm.model_cost[model]
assert json.dumps(row) == before
"#), Some(&locals), None).unwrap();
        });
    }

    #[rstest]
    #[case::null_input(json!({"input_cost_per_token":1,"output_cost_per_token":2,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":null,"output_cost_per_token":4}]}), json!({"input_cost_per_token":10,"output_cost_per_token":10}), &["input_cost_per_token"], (0.0,40.0), true)]
    #[case::unused_null_input(json!({"input_cost_per_token":1,"output_cost_per_token":2,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":null,"output_cost_per_token":4}]}), json!({"input_cost_per_token":0,"output_cost_per_token":10}), &[], (0.0,40.0), true)]
    #[case::null_only_not_free(json!({"tiered_pricing":[{"range":[0,100],"input_cost_per_token":null}]}), json!({"input_cost_per_token":10}), &["input_cost_per_token"], (0.0,0.0), false)]
    #[case::null_output_no_flat_fallback(json!({"output_cost_per_token":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"output_cost_per_token":null}]}), json!({"input_cost_per_token":10,"output_cost_per_token":10}), &["output_cost_per_token"], (10.0,0.0), true)]
    #[case::absent_output_uses_flat(json!({"output_cost_per_token":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1}]}), json!({"input_cost_per_token":10,"output_cost_per_token":10}), &[], (10.0,90.0), true)]
    #[case::zero_output_is_free(json!({"output_cost_per_token":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"output_cost_per_token":0}]}), json!({"input_cost_per_token":10,"output_cost_per_token":10}), &[], (10.0,0.0), true)]
    #[case::null_read_falls_back_input(json!({"cache_read_input_token_cost":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"cache_read_input_token_cost":null}]}), json!({"cache_read_input_token_cost":10}), &[], (10.0,0.0), true)]
    #[case::null_write_falls_back_input(json!({"cache_creation_input_token_cost":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"cache_creation_input_token_cost":null}]}), json!({"cache_creation_input_token_cost":10}), &[], (10.0,0.0), true)]
    #[case::null_ttl_falls_back_write(json!({"cache_creation_input_token_cost_above_1hr":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"cache_creation_input_token_cost":3,"cache_creation_input_token_cost_above_1hr":null}]}), json!({"cache_creation_input_token_cost_above_1hr":10}), &[], (30.0,0.0), true)]
    #[case::null_ttl_falls_back_offpeak_write(json!({"cache_creation_input_token_cost_above_12hr":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"cache_creation_input_token_cost":3,"cache_creation_input_token_cost_above_12hr":null}],"off_peak_pricing":{"hours_utc":"00:00-01:00","cache_creation_input_token_cost":0}}), json!({"cache_creation_input_token_cost_above_12hr":10}), &[], (0.0,0.0), true)]
    #[case::null_reasoning_uses_tier_output(json!({"output_cost_per_reasoning_token":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"output_cost_per_token":2,"output_cost_per_reasoning_token":null}]}), json!({"output_cost_per_reasoning_token":10}), &[], (0.0,20.0), true)]
    #[case::null_reasoning_without_tier_output_is_unknown(json!({"output_cost_per_token":8,"output_cost_per_reasoning_token":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"output_cost_per_reasoning_token":null}]}), json!({"output_cost_per_reasoning_token":10}), &["output_cost_per_reasoning_token"], (0.0,0.0), true)]
    #[case::null_output_for_reasoning_is_unknown(json!({"output_cost_per_token":8,"output_cost_per_reasoning_token":9,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"output_cost_per_token":null}]}), json!({"output_cost_per_reasoning_token":10}), &["output_cost_per_reasoning_token"], (0.0,0.0), true)]
    #[case::null_cache_input_does_not_use_offpeak_input(json!({"tiered_pricing":[{"range":[0,100],"input_cost_per_token":null}],"off_peak_pricing":{"hours_utc":"00:00-01:00","input_cost_per_token":2}}), json!({"cache_read_input_token_cost":10}), &["cache_read_input_token_cost"], (0.0,0.0), true)]
    #[case::offpeak_supplies_null_input(json!({"tiered_pricing":[{"range":[0,100],"input_cost_per_token":null}],"off_peak_pricing":{"hours_utc":"00:00-01:00","input_cost_per_token":2}}), json!({"input_cost_per_token":10}), &[], (20.0,0.0), true)]
    #[case::offpeak_supplies_null_output(json!({"tiered_pricing":[{"range":[0,100],"input_cost_per_token":1,"output_cost_per_token":null}],"off_peak_pricing":{"hours_utc":"00:00-01:00","output_cost_per_token":2}}), json!({"output_cost_per_token":10}), &[], (0.0,20.0), true)]
    fn sdk_nullable_tiers_preserve_unknown_and_fallback_boundaries(
        #[from(interpreter)] _interpreter: (),
        #[case] fields: Value,
        #[case] quantities: Value,
        #[case] missing: &[&str],
        #[case] amounts: (f64, f64),
        #[case] present: bool,
    ) {
        Python::attach(|py| {
            let row = dict(py, fields);
            row.set_item("litellm_provider", "openai").unwrap();
            let before: Value = from_py(row.as_any()).unwrap();
            let usage = UsageInput::new(&dict(py, quantities), 10, false, 0).unwrap();
            let result = price(
                row.as_any(),
                &usage,
                &WhenInput::new(0, "standard", "online").unwrap(),
                "sdk_tokens",
                None,
            )
            .unwrap();
            let priced = match result {
                PriceResult::Unpriced(unpriced) => {
                    assert!(!missing.is_empty());
                    assert_eq!(unpriced.missing, missing);
                    unpriced.partial
                }
                PriceResult::Priced(priced) => {
                    assert!(missing.is_empty());
                    Py::new(py, priced).unwrap()
                }
                PriceResult::Errors(_) => panic!("nullable fields must not reject valid rows"),
            };
            let priced = priced.borrow(py);
            assert_eq!((priced.input, priced.output), amounts);
            assert_eq!(priced.total, amounts.0 + amounts.1);
            let TokenPricePresence::Present(has_prices) =
                has_token_prices(row.as_any(), "sdk_tokens").unwrap()
            else {
                panic!("invalid presence result")
            };
            assert_eq!(has_prices, present);
            assert_eq!(from_py::<Value>(row.as_any()).unwrap(), before);
        });
    }

    #[rstest]
    #[case::priority_reasoning(
        json!({"input_cost_per_token":1,"output_cost_per_token":2,"output_cost_per_reasoning_token":3,"output_cost_per_token_priority":4}),
        "output_cost_per_reasoning_token", (10, "priority"), 10.0, (35.0, 40.0), Some("output_cost_per_reasoning_token")
    )]
    #[case::tier_reasoning(
        json!({"output_cost_per_reasoning_token":5,"tiered_pricing":[{"range":[0,100],"input_cost_per_token":2,"output_cost_per_token":3},{"range":[100,1000],"input_cost_per_token":4,"output_cost_per_token":6}]}),
        "output_cost_per_reasoning_token", (10, "standard"), 20.0, (40.0, 30.0), Some("output_cost_per_reasoning_token")
    )]
    #[case::priority_audio(
        json!({"input_cost_per_token":1,"output_cost_per_token":2,"output_cost_per_audio_token":3,"output_cost_per_audio_token_priority":4}),
        "output_cost_per_audio_token", (10, "priority"), 10.0, (30.0, 25.0), None
    )]
    #[case::priority_threshold(
        json!({"input_cost_per_token":1,"output_cost_per_token":2,"output_cost_per_reasoning_token":3,"output_cost_per_token_priority":4,"output_cost_per_token_above_200k_tokens_priority":8}),
        "output_cost_per_reasoning_token", (200001, "priority"), 200001.0, (55.0, 80.0), Some("output_cost_per_reasoning_token")
    )]
    #[case::audio_threshold_fallback(
        json!({"input_cost_per_token":1,"output_cost_per_token":2,"output_cost_per_audio_token_priority":4,"output_cost_per_token_above_128k_tokens":8}),
        "output_cost_per_audio_token", (128001, "priority"), 128001.0, (60.0, 80.0), Some("output_cost_per_audio_token")
    )]
    fn sdk_token_policy_is_explicit_and_preserves_live_rows(
        #[from(interpreter)] _interpreter: (),
        #[case] fields: Value,
        #[case] meter: &str,
        #[case] context: (u64, &str),
        #[case] input: f64,
        #[case] outputs: (f64, f64),
        #[case] fallback: Option<&str>,
    ) {
        let (prompt_tokens, tier) = context;
        let (generic_output, sdk_output) = outputs;
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            let row = dict(py, fields);
            row.set_item("litellm_provider", "openai").unwrap();
            row.set_item("mode", "chat").unwrap();
            locals.set_item("row", &row).unwrap();
            locals.set_item("meter", meter).unwrap();
            locals.set_item("tier", tier).unwrap();
            locals.set_item("prompt", prompt_tokens).unwrap();
            locals.set_item("expected_input", input).unwrap();
            locals.set_item("generic_output", generic_output).unwrap();
            locals.set_item("sdk_output", sdk_output).unwrap();
            locals.set_item("fallback", fallback).unwrap();
            py.run(pyo3::ffi::c_str!(r#"
import json
import litellm
before = json.dumps(row)
usage = n.UsageInput({"input_cost_per_token": prompt, "output_cost_per_token": 5, meter: 5}, prompt_tokens=prompt)
when = n.WhenInput(0, tier=tier)
default = n.price(row, usage, when)
generic = n.price(row, usage, when, policy="generic")
sdk = n.price(row, usage, when, policy="sdk_tokens")
project = lambda result: (result.input, result.output, result.total, [(l.meter, l.quantity, l.unit_price, l.amount, l.fallback_from) for l in result.lines])
assert project(default) == project(generic)
assert (default.input, default.output, default.total) == (expected_input, generic_output, expected_input + generic_output)
assert (sdk.input, sdk.output, sdk.total) == (expected_input, sdk_output, expected_input + sdk_output)
assert sdk.lines[-1].fallback_from == fallback
model = f"openai/p2d-{meter}-{tier}-{prompt}"
litellm.model_cost[model] = row
try:
    details = {"reasoning_tokens" if meter == "output_cost_per_reasoning_token" else "audio_tokens": 5}
    public_usage = litellm.Usage(prompt_tokens=prompt, completion_tokens=10, total_tokens=prompt+10, completion_tokens_details=details)
    kwargs = dict(model=model, custom_llm_provider="openai", service_tier=tier)
    assert litellm.cost_per_token(prompt_tokens=prompt, completion_tokens=10, usage_object=public_usage, **kwargs) == (sdk.input, sdk.output)
    assert litellm.completion_cost(completion_response=litellm.ModelResponse(model=model, usage=public_usage), **kwargs) == sdk.total
finally:
    del litellm.model_cost[model]
assert json.dumps(row) == before
row["output_cost_per_reasoning_token_priority"] = 0
live = n.price(row, n.UsageInput({"output_cost_per_reasoning_token": 5}), n.WhenInput(0, tier="priority"), policy="sdk_tokens")
assert live.total == 0
assert live.lines[0].fallback_from is None
assert row["output_cost_per_reasoning_token_priority"] == 0
"#), Some(&locals), None).unwrap();
        });
    }

    #[rstest]
    #[case::unknown("'unknown'")]
    #[case::case_sensitive("'SDK_TOKENS'")]
    #[case::null("None")]
    #[case::positional("'sdk_tokens'")]
    fn pricing_policy_rejects_invalid_arguments(
        #[from(interpreter)] _interpreter: (),
        #[case] argument: &str,
    ) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            locals.set_item("argument", argument).unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    r#"
row = {"litellm_provider": "test", "input_cost_per_token": 1}
usage = n.UsageInput({"input_cost_per_token": 1})
when = n.WhenInput(0)
try:
    if argument == "'sdk_tokens'":
        n.price(row, usage, when, "sdk_tokens")
    else:
        n.price(row, usage, when, policy=eval(argument))
except (TypeError, ValueError):
    pass
else:
    raise AssertionError("invalid policy accepted")
"#
                ),
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[rstest]
    fn cyclic_rows_are_reported_without_recursing_forever(#[from(interpreter)] _interpreter: ()) {
        Python::attach(|py| {
            let row = dict(py, json!({"litellm_provider":"test"}));
            row.set_item("cycle", &row).unwrap();
            let result = lower_row("", row.as_any());
            row.del_item("cycle").unwrap();
            assert!(matches!(result, Err(RowErrors { errors }) if errors[0].kind == "invalid"));
        });
    }

    #[rstest]
    #[case::folded_provider("MODEL", Some("custom"), "custom/model", "Lowercase")]
    #[case::exact_case("custom/Model", Some("custom"), "custom/Model", "Exact")]
    #[case::bare("MODEL", None, "MODEL", "Exact")]
    fn case_precedence_is_per_candidate(
        #[from(interpreter)] _interpreter: (),
        #[case] model: &str,
        #[case] provider: Option<&str>,
        #[case] key: &str,
        #[case] via: &str,
    ) {
        Python::attach(|py| {
            let rows = dict(
                py,
                json!({"custom/Model": {}, "custom/model": {}, "MODEL": {}}),
            );
            let ResolutionResult::Found(found) =
                resolve_model(&module(py), &rows, model, provider, None, None, None, false)
                    .unwrap()
            else {
                panic!("not found")
            };
            assert_eq!((found.key.as_str(), found.via.as_str()), (key, via));
        });
    }

    #[rstest]
    #[case::bare("read-model", "read-model", "openai")]
    #[case::mismatch("azure/read-model", "read-model", "azure")]
    fn inferred_intent_is_bound_without_mutating_rows(
        #[from(interpreter)] _interpreter: (),
        #[case] model: &str,
        #[case] normalized: &str,
        #[case] provider: &str,
    ) {
        Python::attach(|py| {
            let module = crate::native_module(py);
            let rows = dict(
                py,
                json!({
                    "read-model": {"litellm_provider": provider, "supports_vision": true},
                    "openai/read-model": {"litellm_provider": "openai", "supports_vision": false},
                    "azure/read-model": {"litellm_provider": "openai", "supports_vision": false}
                }),
            );
            let locals = PyDict::new(py);
            locals.set_item("n", &module).unwrap();
            locals.set_item("rows", &rows).unwrap();
            locals.set_item("model", model).unwrap();
            locals.set_item("inferred", (normalized, provider)).unwrap();
            py.run(
                cr#"
before = tuple((key, id(row), dict(row)) for key, row in rows.items())
result = n.resolve_model(rows, model, inferred=inferred)
assert result.key == "read-model"
assert n.model_info(rows, result.key).capabilities["supports_vision"] is True
assert tuple((key, id(row), dict(row)) for key, row in rows.items()) == before
rows["read-model"]["litellm_provider"] = "wrong-scope"
assert isinstance(n.resolve_model(rows, model, inferred=inferred), n.ModelNotFound)
"#,
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[rstest]
    fn base_misses_and_provider_fallbacks_preserve_order(#[from(interpreter)] _interpreter: ()) {
        Python::attach(|py| {
            let module = module(py);
            let rows = dict(
                py,
                json!({"openai/model": {"litellm_provider":"anthropic"}, "model": {"litellm_provider":"openai"}}),
            );
            let ResolutionResult::NotFound(missing) = resolve_model(
                &module,
                &rows,
                "model",
                None,
                Some("missing-base"),
                None,
                None,
                false,
            )
            .unwrap() else {
                panic!("unexpected base fallback")
            };
            assert_eq!(missing.tried, ["missing-base"]);
            let ResolutionResult::Found(found) = resolve_model(
                &module,
                &rows,
                "model",
                Some("openai"),
                None,
                None,
                None,
                false,
            )
            .unwrap() else {
                panic!("not found")
            };
            assert_eq!((found.key.as_str(), found.via.as_str()), ("model", "Exact"));
            let row = rows.get_item("openai/model").unwrap().unwrap();
            row.cast::<PyDict>()
                .unwrap()
                .set_item("litellm_provider", "openai")
                .unwrap();
            let ResolutionResult::Found(found) = resolve_model(
                &module,
                &rows,
                "model",
                Some("openai"),
                None,
                None,
                None,
                false,
            )
            .unwrap() else {
                panic!("not found")
            };
            assert_eq!(found.key, "openai/model");
            assert_eq!(
                registration_target(&module, &rows, "openai/model", None)
                    .unwrap()
                    .inherit_from
                    .as_deref(),
                Some("openai/model")
            );
            let ResolutionResult::NotFound(missing) = resolve_model(
                &module,
                &PyDict::new(py),
                "databricks/model-99",
                Some("databricks"),
                None,
                None,
                None,
                false,
            )
            .unwrap() else {
                panic!("unexpected match")
            };
            assert_eq!(
                missing.tried,
                [
                    "databricks/model-99",
                    "model-99",
                    "databricks/model",
                    "model",
                    "databricks/databricks/model-99"
                ]
            );
        });
    }

    #[rstest]
    #[case::online("online", 0, 3.0)]
    #[case::peak("online", 3600, 6.0)]
    #[case::batch("batch", 0, 12.0)]
    #[case::decisions("decisions", 0, 15.0)]
    fn pricing_channel_and_time_reach_core(
        #[from(interpreter)] _interpreter: (),
        #[case] channel: &str,
        #[case] at: i64,
        #[case] expected: f64,
    ) {
        Python::attach(|py| {
            let row = dict(
                py,
                json!({"litellm_provider":"test", "input_cost_per_token":2, "input_cost_per_token_batches":4, "input_cost_per_token_decisions":5, "off_peak_pricing":{"hours_utc":"00:00-01:00","input_cost_per_token":1}}),
            );
            let usage =
                UsageInput::new(&dict(py, json!({"input_cost_per_token":3})), 0, false, 0).unwrap();
            let PriceResult::Priced(result) = price(
                row.as_any(),
                &usage,
                &WhenInput::new(at, "standard", channel).unwrap(),
                "generic",
                None,
            )
            .unwrap() else {
                panic!("not priced")
            };
            assert_eq!(result.total, expected);
        });
    }

    #[rstest]
    #[case::search(json!({"search_context_cost_per_query":{"search_context_size_low":2}}), "web_search/low")]
    #[case::guardrail(json!({"guardrail_cost_per_unit":{"contentPolicyUnits":2}}), "guardrail/contentPolicyUnits")]
    fn block_meters_round_trip(
        #[from(interpreter)] _interpreter: (),
        #[case] fields: Value,
        #[case] meter: &str,
    ) {
        Python::attach(|py| {
            let row = dict(py, fields);
            row.set_item("litellm_provider", "test").unwrap();
            let usage = UsageInput::new(&dict(py, json!({meter:3})), 0, false, 0).unwrap();
            let PriceResult::Priced(result) = price(
                row.as_any(),
                &usage,
                &WhenInput::new(0, "standard", "online").unwrap(),
                "generic",
                None,
            )
            .unwrap() else {
                panic!("not priced")
            };
            assert_eq!(result.total, 6.0);
            assert_eq!(result.lines[0].meter, meter);
        });
    }

    #[rstest]
    fn index_eviction_releases_rows_without_holding_a_lock(#[from(interpreter)] _interpreter: ()) {
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("n", crate::native_module(py)).unwrap();
            py.run(
                pyo3::ffi::c_str!(
                    r#"
import weakref
seen = []
class Row(dict):
    def __del__(self):
        n.invalidate_model_cost_index()
        seen.append("released")
row = Row(litellm_provider="test")
ref = weakref.ref(row)
models = {"Model": row}
n.resolve_model(models, "MODEL")
del row, models
assert ref() is not None
n.invalidate_model_cost_index()
assert ref() is None
assert seen == ["released"]
"#
                ),
                Some(&locals),
                None,
            )
            .unwrap();
        });
    }

    #[rstest]
    fn prices_every_existing_r2_reference_row(#[from(interpreter)] _interpreter: ()) {
        Python::attach(|py| {
            for line in include_str!("../../../cost/tests/python_reference.tsv")
                .lines()
                .filter(|line| !line.starts_with('#'))
            {
                let fields: Vec<_> = line.split('\t').collect();
                let count = |i: usize| fields[i].parse::<u64>().unwrap();
                let row = dict(py, json!({"litellm_provider": "test"}));
                for (i, key) in [
                    (5, "input_cost_per_token".to_owned()),
                    (6, "output_cost_per_token".to_owned()),
                    (7, "cache_read_input_token_cost".to_owned()),
                    (8, "cache_creation_input_token_cost".to_owned()),
                    (
                        10,
                        format!("input_cost_per_token_above_{}_tokens", fields[9]),
                    ),
                    (
                        11,
                        format!("output_cost_per_token_above_{}_tokens", fields[9]),
                    ),
                ] {
                    if let Ok(rate) = fields[i].parse::<f64>() {
                        row.set_item(key, rate).unwrap();
                    }
                }
                let usage = UsageInput::new(
                    &dict(
                        py,
                        json!({
                            "input_cost_per_token": count(1) - count(3) - count(4),
                            "output_cost_per_token": count(2),
                            "cache_read_input_token_cost": count(3),
                            "cache_creation_input_token_cost": count(4),
                        }),
                    ),
                    count(1),
                    false,
                    0,
                )
                .unwrap();
                let PriceResult::Priced(result) = price(
                    row.as_any(),
                    &usage,
                    &WhenInput::new(0, "standard", "online").unwrap(),
                    "generic",
                    None,
                )
                .unwrap() else {
                    panic!("not priced: {}", fields[0])
                };
                assert_eq!(
                    (result.input, result.output),
                    (
                        fields[12].parse::<f64>().unwrap(),
                        fields[13].parse::<f64>().unwrap()
                    ),
                    "{}",
                    fields[0]
                );
            }
        });
    }

    #[derive(Deserialize)]
    struct Reference {
        models: Vec<ReferenceModel>,
    }
    #[derive(Deserialize)]
    struct ReferenceModel {
        info: Fields,
        inclusive: bool,
        checks: Vec<(Option<String>, u64, [f64; 5])>,
    }

    #[rstest]
    fn prices_existing_catalog_reference_through_live_rows(#[from(interpreter)] _interpreter: ()) {
        Python::attach(|py| {
            let document = json_loads(
                py,
                include_bytes!("../../../model-catalog/tests/price_lookup_reference.json"),
            )
            .unwrap();
            let reference: Reference = from_py(document.bind(py)).unwrap();
            for model in reference.models {
                let row = fields_to_dict(py, &model.info).unwrap().into_bound(py);
                row.set_item("litellm_provider", "test").unwrap();
                for (tier, prompt, expected) in model.checks {
                    let when =
                        WhenInput::new(0, tier.as_deref().unwrap_or("standard"), "online").unwrap();
                    for (meter, expected) in [
                        "input_cost_per_token",
                        "output_cost_per_token",
                        "cache_creation_input_token_cost",
                        "cache_creation_input_token_cost_above_1hr",
                        "cache_read_input_token_cost",
                    ]
                    .into_iter()
                    .zip(expected)
                    {
                        let usage = UsageInput::new(
                            &dict(py, json!({meter: 1})),
                            prompt,
                            model.inclusive,
                            0,
                        )
                        .unwrap();
                        match price(row.as_any(), &usage, &when, "generic", None).unwrap() {
                            PriceResult::Priced(result) => assert_eq!(result.total, expected),
                            PriceResult::Unpriced(result) => {
                                assert_eq!(expected, 0.0);
                                assert_eq!(result.missing, [meter]);
                            }
                            PriceResult::Errors(errors) => {
                                panic!("invalid fixture: {}", errors.errors[0].message)
                            }
                        }
                    }
                }
            }
        });
    }

    #[rstest]
    fn unknown_free_and_fallback_lines_remain_distinct(#[from(interpreter)] _interpreter: ()) {
        Python::attach(|py| {
            let usage = UsageInput::new(
                &dict(
                    py,
                    json!({"input_cost_per_audio_token": 3, "output_cost_per_token": 2}),
                ),
                3,
                false,
                0,
            )
            .unwrap();
            let when = WhenInput::new(0, "standard", "online").unwrap();
            let row = dict(
                py,
                json!({"litellm_provider":"test", "input_cost_per_token": 2}),
            );
            let PriceResult::Unpriced(result) =
                price(row.as_any(), &usage, &when, "generic", None).unwrap()
            else {
                panic!("expected unpriced")
            };
            assert_eq!(result.missing, ["output_cost_per_token"]);
            let partial = result.partial.borrow(py);
            assert_eq!(partial.total, 6.0);
            assert_eq!(
                (
                    partial.lines[0].meter.as_str(),
                    partial.lines[0].fallback_from.as_deref()
                ),
                ("input_cost_per_token", Some("input_cost_per_audio_token"))
            );
            row.set_item("output_cost_per_token", 0).unwrap();
            row.set_item("input_cost_per_audio_token", 0).unwrap();
            let PriceResult::Priced(result) =
                price(row.as_any(), &usage, &when, "generic", None).unwrap()
            else {
                panic!("expected free")
            };
            assert_eq!(result.total, 0.0);
            assert_eq!(result.lines[0].fallback_from, None);
        });
    }

    #[rstest]
    #[case::character("input_cost_per_character")]
    #[case::audio_second("input_cost_per_audio_per_second")]
    #[case::second("cost_per_second")]
    #[case::image("output_cost_per_image_1K")]
    #[case::video("output_cost_per_second_720p")]
    #[case::query("input_cost_per_query")]
    #[case::ttl("cache_creation_input_token_cost_above_12hr")]
    #[case::reasoning("output_cost_per_reasoning_token")]
    fn meters_preserve_names_and_units(#[from(interpreter)] _interpreter: (), #[case] meter: &str) {
        Python::attach(|py| {
            let row = dict(py, json!({"litellm_provider":"test", meter: 2.0}));
            let usage = UsageInput::new(&dict(py, json!({meter: 3})), 0, false, 0).unwrap();
            let PriceResult::Priced(result) = price(
                row.as_any(),
                &usage,
                &WhenInput::new(0, "standard", "online").unwrap(),
                "generic",
                None,
            )
            .unwrap() else {
                panic!("not priced")
            };
            assert_eq!(result.total, 6.0);
            assert_eq!(result.lines[0].meter, meter);
            assert_eq!(
                (
                    result.lines[0].quantity,
                    result.lines[0].unit_price,
                    result.lines[0].amount,
                    &result.lines[0].fallback_from
                ),
                (3, 2.0, 6.0, &None)
            );
        });
    }
}
