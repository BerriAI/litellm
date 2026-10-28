use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use litellm_cost::call::CallContext as CostCallContext;
use litellm_cost::price::{
    Adjustments, CacheReadMode, CacheTtl, Channel, InputUnit, Meter, OutputUnit,
    Prices as CostPrices, Residency, Resolution, ServiceTier, TextUnit, TokenKind,
};
use litellm_cost::priced::{Outcome, Quantity, Side};
use litellm_cost::usage::{Characters, DurationMeter, Usage as CostUsage};
use litellm_host_python::from_py;
use litellm_model_catalog::rejects_row;
use litellm_model_catalog::wire::{EntryWire, Fields, FlatKey};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::model_registry::Invalid;

/// A row's typed prices and uplifts, an immutable snapshot taken when the row was resolved
#[derive(Default)]
#[pyclass(frozen, name = "Prices", module = "litellm.rust_bridge._native")]
pub(crate) struct PricesHandle {
    prices: Arc<CostPrices>,
    adjustments: Arc<Adjustments>,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Usage(CostUsage);

/// The call facts pricing reads. A custom pricing overlay that fails validation is kept as its errors
#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct CallContext {
    call: CostCallContext,
    invalid_overlay: Option<(Vec<String>, Arc<Fields>)>,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Priced {
    #[pyo3(get)]
    total: f64,
    #[pyo3(get)]
    prompt: f64,
    #[pyo3(get)]
    completion: f64,
    lines: Vec<Py<Line>>,
    missing: Vec<String>,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Line {
    #[pyo3(get)]
    meter: String,
    quantity: Quantity,
    #[pyo3(get)]
    unit_price: f64,
    #[pyo3(get)]
    amount: f64,
    #[pyo3(get)]
    fallback_from: Option<String>,
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct Unpriced {
    missing: Vec<String>,
}

impl PricesHandle {
    /// Lowers a row the way the catalog lowers its entries, `Err` with every error that rejects the row
    pub(crate) fn lower(key: &str, fields: &Fields) -> Result<Self, Vec<String>> {
        let wire = EntryWire::read(&as_main_reads_tiers(fields));
        let adjustments = Adjustments::from(&wire);
        let (prices, price_errors) = match CostPrices::try_from(&wire) {
            Ok(prices) => (Some(prices), Vec::new()),
            Err(errors) => (None, errors),
        };
        let errors: Vec<String> = wire
            .problems
            .into_iter()
            .chain(price_errors)
            .filter(|error| rejects_row(&error.error))
            .map(|mut error| {
                error.prepend(key);
                error.to_string()
            })
            .collect();
        match prices {
            Some(prices) if errors.is_empty() => Ok(Self {
                prices: Arc::new(prices),
                adjustments: Arc::new(adjustments),
            }),
            _ => Err(errors),
        }
    }
}

#[pymethods]
impl PricesHandle {
    /// Prices for a row that did not come from `resolve`, such as a deployment's `model_info`
    #[staticmethod]
    #[pyo3(signature = (row, key = "model_info"))]
    fn from_row(py: Python<'_>, row: &Bound<'_, PyDict>, key: &str) -> PyResult<Py<PyAny>> {
        let (fields, converted) = json_fields(row);
        let lowered = match converted {
            true => Self::lower(key, &fields),
            false => Err(vec![format!("{key} must contain JSON-compatible prices")]),
        };
        match lowered {
            Ok(prices) => Ok(prices.into_pyobject(py)?.into_any().unbind()),
            Err(errors) => Ok(Invalid::new(key.to_owned(), errors, Arc::new(fields))
                .into_pyobject(py)?
                .into_any()
                .unbind()),
        }
    }

    /// `"character"` when the row bills text by the character and not by the token (decision 32)
    #[getter]
    fn text_unit(&self) -> &'static str {
        match self.prices.billing().text_unit {
            TextUnit::Token => "token",
            TextUnit::Character => "character",
        }
    }

    /// Pure arithmetic over owned values, it calls no Python and never reads `litellm.model_cost`
    fn price(&self, py: Python<'_>, usage: &Usage, call: &CallContext) -> PyResult<Py<PyAny>> {
        if let Some((errors, fields)) = &call.invalid_overlay {
            return Ok(
                Invalid::new("custom_pricing".to_owned(), errors.clone(), fields.clone())
                    .into_pyobject(py)?
                    .into_any()
                    .unbind(),
            );
        }
        match litellm_cost::calculate(&self.prices, &self.adjustments, &usage.0, &call.call) {
            Outcome::Unpriced { missing } => Ok(Unpriced {
                missing: missing.iter().map(meter_name).collect(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind()),
            Outcome::Priced(priced) => {
                let side_total = |side: Side| {
                    priced
                        .lines
                        .iter()
                        .filter(|line| line.side == side)
                        .map(|line| line.amount.get())
                        .sum()
                };
                let lines = priced
                    .lines
                    .iter()
                    .map(|line| {
                        Py::new(
                            py,
                            Line {
                                meter: meter_name(&line.meter),
                                quantity: line.quantity,
                                unit_price: line.unit_price.get(),
                                amount: line.amount.get(),
                                fallback_from: line.fallback_from.as_ref().map(meter_name),
                            },
                        )
                    })
                    .collect::<PyResult<_>>()?;
                Ok(Priced {
                    total: priced.total.get(),
                    prompt: side_total(Side::Prompt),
                    completion: side_total(Side::Completion),
                    lines,
                    missing: priced.missing.iter().map(meter_name).collect(),
                }
                .into_pyobject(py)?
                .into_any()
                .unbind())
            }
        }
    }
}

#[pymethods]
impl Usage {
    #[new]
    #[pyo3(signature = (
        *, prompt_tokens, text, output_text, audio = 0, image = 0, video = 0, cache_read = 0, cache_read_audio = 0,
        cache_read_explicit = 0, cache_write_5m = 0, cache_write_1h = 0, cache_write_12h = 0, output_audio = 0,
        output_image = 0, output_video = 0, reasoning = 0, prompt_characters = 0, completion_characters = 0,
        wall_clock = None, input_audio = None, live_voice = None, output_video_duration = None,
        video_resolution = None, text_messages = 0,
    ))]
    #[expect(
        clippy::too_many_arguments,
        reason = "one keyword argument per token meter, the Python signature"
    )]
    fn new(
        prompt_tokens: u64,
        text: u64,
        output_text: u64,
        audio: u64,
        image: u64,
        video: u64,
        cache_read: u64,
        cache_read_audio: u64,
        cache_read_explicit: u64,
        cache_write_5m: u64,
        cache_write_1h: u64,
        cache_write_12h: u64,
        output_audio: u64,
        output_image: u64,
        output_video: u64,
        reasoning: u64,
        prompt_characters: u64,
        completion_characters: u64,
        wall_clock: Option<Duration>,
        input_audio: Option<Duration>,
        live_voice: Option<Duration>,
        output_video_duration: Option<Duration>,
        video_resolution: Option<&str>,
        text_messages: u64,
    ) -> Self {
        let mut usage = CostUsage {
            prompt_tokens,
            ..CostUsage::default()
        };
        usage.tokens.input = BTreeMap::from([
            (TokenKind::Text, text),
            (TokenKind::Audio, audio),
            (TokenKind::Image, image),
            (TokenKind::Video, video),
        ]);
        usage.tokens.cache_read = BTreeMap::from([
            ((TokenKind::Text, CacheReadMode::Implicit), cache_read),
            (
                (TokenKind::Text, CacheReadMode::Explicit),
                cache_read_explicit,
            ),
            (
                (TokenKind::Audio, CacheReadMode::Implicit),
                cache_read_audio,
            ),
        ]);
        usage.tokens.cache_write = BTreeMap::from([
            ((TokenKind::Text, CacheTtl::Default), cache_write_5m),
            ((TokenKind::Text, CacheTtl::OneHour), cache_write_1h),
            ((TokenKind::Text, CacheTtl::TwelveHours), cache_write_12h),
        ]);
        usage.tokens.output = BTreeMap::from([
            (TokenKind::Text, output_text),
            (TokenKind::Audio, output_audio),
            (TokenKind::Image, output_image),
            (TokenKind::Video, output_video),
        ]);
        usage.tokens.reasoning = reasoning;
        usage.characters = Characters {
            prompt: prompt_characters,
            completion: completion_characters,
        };
        usage.durations = [
            (DurationMeter::WallClock, wall_clock),
            (DurationMeter::InputAudio, input_audio),
            (DurationMeter::LiveVoice, live_voice),
            (
                DurationMeter::OutputVideo {
                    resolution: video_resolution
                        .map(|resolution| resolution.trim().to_lowercase())
                        .and_then(|resolution| Resolution::new(&resolution)),
                },
                output_video_duration,
            ),
        ]
        .into_iter()
        .filter_map(|(meter, duration)| Some((meter, duration?)))
        .collect();
        usage.text_messages = text_messages;
        Self(usage)
    }

    /// The provider-reported prompt size, which picks the band
    #[getter]
    fn prompt_tokens(&self) -> u64 {
        self.0.prompt_tokens
    }
}

#[pymethods]
impl CallContext {
    #[new]
    #[pyo3(signature = (*, service_tier, channel, data_residency, vertex_location, at, custom_pricing))]
    fn new(
        service_tier: Option<&str>,
        channel: &str,
        data_residency: Option<&str>,
        vertex_location: Option<&str>,
        at: &Bound<'_, PyAny>,
        custom_pricing: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let channel = match channel {
            "online" => Channel::Online,
            "batch" => Channel::Batch,
            "decisions" => Channel::Decisions,
            other => return Err(PyValueError::new_err(format!("unknown channel {other:?}"))),
        };
        let overlay = custom_pricing.map(|pricing| {
            let (fields, converted) = json_fields(pricing);
            let lowered = match converted {
                true => PricesHandle::lower("custom_pricing", &fields),
                false => Err(vec![
                    "custom_pricing must contain JSON-compatible prices".to_owned(),
                ]),
            };
            lowered
                .map(|handle| Arc::unwrap_or_clone(handle.prices))
                .map_err(|errors| (errors, Arc::new(fields)))
        });
        let (overlay, invalid_overlay) = match overlay {
            None => (None, None),
            Some(Ok(prices)) => (Some(prices), None),
            Some(Err(invalid)) => (None, Some(invalid)),
        };
        Ok(Self {
            call: CostCallContext {
                tier: service_tier.map_or(ServiceTier::Standard, tier),
                channel,
                residency: data_residency.and_then(residency),
                non_global_endpoint: vertex_location
                    .is_some_and(|location| !location.eq_ignore_ascii_case("global")),
                speed: None,
                inference_geo: None,
                at: timestamp(at)?,
                overlay,
            },
            invalid_overlay,
        })
    }
}

#[pymethods]
impl Priced {
    #[getter]
    fn lines<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.lines.iter().map(|line| line.clone_ref(py)))
    }

    #[getter]
    fn missing<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.missing)
    }
}

#[pymethods]
impl Line {
    /// A count, or a `timedelta` for a second meter
    #[getter]
    fn quantity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.quantity {
            Quantity::Count(count) => Ok(count.into_pyobject(py)?.into_any()),
            Quantity::Duration(duration) => Ok(duration.into_pyobject(py)?.into_any()),
        }
    }
}

#[pymethods]
impl Unpriced {
    #[getter]
    fn missing<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.missing)
    }
}

/// The row's fields that convert to JSON, and whether every price did. A price only Python can hold, such as `nan`,
/// makes the row invalid (decision 20). Any other such value, a router deployment's `datetime` say, is left out
fn json_fields(row: &Bound<'_, PyDict>) -> (Fields, bool) {
    let converted: Vec<(String, Option<serde_json::Value>)> = row
        .iter()
        .filter_map(|(name, value)| {
            let non_finite = value
                .extract::<f64>()
                .is_ok_and(|number| !number.is_finite());
            Some((
                name.extract::<String>().ok()?,
                from_py(&value).ok().filter(|_| !non_finite),
            ))
        })
        .collect();
    let prices_convert = converted
        .iter()
        .all(|(name, value)| value.is_some() || FlatKey::parse(name).is_none());
    let fields = converted
        .into_iter()
        .filter_map(|(name, value)| Some((name, value?)))
        .collect();
    (fields, prices_convert)
}

/// Flat prices a priced tier table replaces, main's `_get_tiered_base_costs`. The row's own output price stays as the
/// fallback for a tier that sets none
const TIER_REPLACED_PRICES: [&str; 5] = [
    "input_cost_per_token",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "cache_creation_input_token_cost_above_1hr",
    "cache_creation_input_token_cost_above_12hr",
];

/// main bills a call with a prompt-token tier table at its tier's rates and never at the row's flat ones, and ignores
/// a table whose first tier sets no input price. The catalog lowers the first tier and the flat prices into one band,
/// so the flat prices the table replaces are dropped before lowering
fn as_main_reads_tiers(fields: &Fields) -> Cow<'_, Fields> {
    let first_tier = fields
        .get("tiered_pricing")
        .and_then(serde_json::Value::as_array)
        .and_then(|tiers| {
            tiers
                .iter()
                .filter_map(serde_json::Value::as_object)
                .find(|tier| {
                    tier.get("range")
                        .and_then(|range| range.get(0))
                        .and_then(serde_json::Value::as_u64)
                        == Some(0)
                })
        });
    let Some(first_tier) = first_tier else {
        return Cow::Borrowed(fields);
    };
    let priced = first_tier.contains_key("input_cost_per_token");
    let replaced = |name: &str| match priced {
        true => TIER_REPLACED_PRICES.contains(&name) || first_tier.contains_key(name),
        false => name == "tiered_pricing",
    };
    Cow::Owned(
        fields
            .iter()
            .filter(|(name, _)| !replaced(name))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
    )
}

/// main's `_SERVICE_TIER_TO_COST_KEY_SUFFIX`, `"fast"` bills the priority tier
fn tier(service_tier: &str) -> ServiceTier {
    match service_tier.to_lowercase().as_str() {
        "priority" | "fast" => ServiceTier::Priority,
        "flex" => ServiceTier::Flex,
        "balanced" => ServiceTier::Balanced,
        "ultrafast" => ServiceTier::Ultrafast,
        _ => ServiceTier::Standard,
    }
}

fn residency(data_residency: &str) -> Option<Residency> {
    match data_residency.to_lowercase().as_str() {
        "eu" => Some(Residency::Eu),
        "us" => Some(Residency::Us),
        _ => None,
    }
}

/// An aware datetime is read as its instant, a naive one as UTC, as main's off-peak check does
fn timestamp(at: &Bound<'_, PyAny>) -> PyResult<jiff::Timestamp> {
    let aware = if at.getattr("tzinfo")?.is_none() {
        let utc = at
            .py()
            .import("datetime")?
            .getattr("timezone")?
            .getattr("utc")?;
        let replace_kwargs = PyDict::new(at.py());
        replace_kwargs.set_item("tzinfo", utc)?;
        at.call_method("replace", (), Some(&replace_kwargs))?
    } else {
        at.clone()
    };
    let seconds: f64 = aware.call_method0("timestamp")?.extract()?;
    jiff::Timestamp::from_microsecond((seconds * 1e6).round() as i64)
        .map_err(|error| PyValueError::new_err(format!("billing time out of range: {error}")))
}

/// main's flat key name for the meter, the name a caller already knows from the JSON
fn meter_name(meter: &Meter) -> String {
    let token = |kind: &TokenKind| match kind {
        TokenKind::Text => "",
        TokenKind::Audio => "audio_",
        TokenKind::Image => "image_",
        TokenKind::Video => "video_",
    };
    match meter {
        Meter::Input(InputUnit::Token(kind)) => format!("input_cost_per_{}token", token(kind)),
        Meter::Output(OutputUnit::Token(kind)) => format!("output_cost_per_{}token", token(kind)),
        Meter::Output(OutputUnit::ReasoningToken) => "output_cost_per_reasoning_token".to_owned(),
        Meter::CacheRead {
            kind,
            mode: CacheReadMode::Implicit,
        } => format!("cache_read_input_{}token_cost", token(kind)),
        Meter::CacheRead {
            kind,
            mode: CacheReadMode::Explicit,
        } => format!("explicit_cache_read_input_{}token_cost", token(kind)),
        Meter::CacheWrite { kind, ttl } => {
            let ttl = match ttl {
                CacheTtl::Default => "",
                CacheTtl::OneHour => "_above_1hr",
                CacheTtl::TwelveHours => "_above_12hr",
            };
            format!("cache_creation_input_{}token_cost{ttl}", token(kind))
        }
        Meter::Input(InputUnit::Character) => "input_cost_per_character".to_owned(),
        Meter::Output(OutputUnit::Character) => "output_cost_per_character".to_owned(),
        Meter::WallClockSecond => "cost_per_second".to_owned(),
        Meter::Input(InputUnit::Second) => "input_cost_per_second".to_owned(),
        Meter::Output(OutputUnit::Second(None)) => "output_cost_per_second".to_owned(),
        Meter::Output(OutputUnit::Second(Some(resolution))) => {
            format!("output_cost_per_second_{}", resolution.as_str())
        }
        Meter::Input(InputUnit::AudioSecond) => "input_cost_per_audio_per_second".to_owned(),
        Meter::Output(OutputUnit::VideoSecond) => "output_cost_per_video_per_second".to_owned(),
        Meter::Input(InputUnit::TextMessage) => "input_cost_per_text_message".to_owned(),
        other => format!("{other:?}"),
    }
}
