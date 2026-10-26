use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use litellm_cost::price::{Band, Channel, Meter, ServiceTier};
use litellm_cost::{Usage, When};
use litellm_host_python::from_py;
use litellm_model_catalog::wire::{Fields, FlatKey};
use litellm_model_catalog::{self as resolution, Keys, Query, Resolved};
use litellm_model_catalog::{Error, Located, Model};
use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule, PySet};
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
pub(crate) struct RegistrationTarget {
    write: String,
    inherit_from: Option<String>,
    names_the_row: bool,
}

#[derive(IntoPyObject)]
pub(crate) enum ResolutionResult {
    Found(ResolvedModel),
    NotFound(ModelNotFound),
}

#[pyfunction(pass_module, signature = (model_cost, model, *, provider=None, base_model=None, region=None))]
pub(crate) fn resolve_model(
    module: &Bound<'_, PyModule>,
    model_cost: &Bound<'_, PyDict>,
    model: &str,
    provider: Option<&str>,
    base_model: Option<&str>,
    region: Option<&str>,
) -> PyResult<ResolutionResult> {
    let index = module.getattr("_model_key_index")?;
    let index = index.extract::<PyRef<'_, ModelKeyIndex>>()?;
    let view = PyView {
        dict: model_cost,
        index: &index,
        error: RefCell::default(),
    };
    let result = resolution::resolve(
        &view,
        &Query {
            model,
            provider,
            base_model,
            region,
        },
    );
    view.finish()?;
    Ok(match result {
        Resolved::Found { key, via } => ResolutionResult::Found(ResolvedModel {
            key,
            via: format!("{via:?}"),
        }),
        Resolved::NotFound { tried } => ResolutionResult::NotFound(ModelNotFound { tried }),
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
    let model = Model::from_fields(&fields).map_err(|errors| RowErrors {
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

#[pyfunction]
pub(crate) fn price(
    row: &Bound<'_, PyAny>,
    usage: &UsageInput,
    when: &WhenInput,
) -> PyResult<PriceResult> {
    let model = match lower_row("", row) {
        Ok((_, model)) => model,
        Err(errors) => return Ok(PriceResult::Errors(errors)),
    };
    Ok(
        match litellm_cost::calculate(&model.prices, &usage.usage, &when.when) {
            litellm_cost::Priced::Known(breakdown) => PriceResult::Priced(breakdown.into()),
            litellm_cost::Priced::Unpriced { missing, partial } => {
                PriceResult::Unpriced(Unpriced {
                    missing: missing.iter().map(meter_name).collect(),
                    partial: Py::new(row.py(), Priced::from(partial))?,
                })
            }
        },
    )
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
            let ResolutionResult::Found(found) =
                resolve_model(&module(py), &rows, model, provider, base, region).unwrap()
            else {
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
            let ResolutionResult::Found(found) =
                resolve_model(&module(py), &rows, model, Some(provider), None, None).unwrap()
            else {
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
                    resolve_model(&module(py), &rows, "model", Some(provider), None, None).unwrap(),
                    ResolutionResult::Found(_)
                ),
                found
            );
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
                resolve_model(&module(py), &rows, model, None, None, None).unwrap()
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
                &module, rows, key, None, None, None,
            )
            .unwrap()
            {
                ResolutionResult::Found(found) => Some(found.key),
                ResolutionResult::NotFound(_) => None,
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
            )
            .unwrap() else {
                panic!("expected invalid row")
            };
            assert_eq!(errors.errors[0].key, "");
            assert_eq!(errors.errors[0].kind, kind);
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
                resolve_model(&module(py), &rows, model, provider, None, None).unwrap()
            else {
                panic!("not found")
            };
            assert_eq!((found.key.as_str(), found.via.as_str()), (key, via));
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
            let ResolutionResult::NotFound(missing) =
                resolve_model(&module, &rows, "model", None, Some("missing-base"), None).unwrap()
            else {
                panic!("unexpected base fallback")
            };
            assert_eq!(missing.tried, ["missing-base"]);
            let ResolutionResult::Found(found) =
                resolve_model(&module, &rows, "model", Some("openai"), None, None).unwrap()
            else {
                panic!("not found")
            };
            assert_eq!((found.key.as_str(), found.via.as_str()), ("model", "Exact"));
            let row = rows.get_item("openai/model").unwrap().unwrap();
            row.cast::<PyDict>()
                .unwrap()
                .set_item("litellm_provider", "openai")
                .unwrap();
            let ResolutionResult::Found(found) =
                resolve_model(&module, &rows, "model", Some("openai"), None, None).unwrap()
            else {
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
                        match price(row.as_any(), &usage, &when).unwrap() {
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
            let PriceResult::Unpriced(result) = price(row.as_any(), &usage, &when).unwrap() else {
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
            let PriceResult::Priced(result) = price(row.as_any(), &usage, &when).unwrap() else {
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
