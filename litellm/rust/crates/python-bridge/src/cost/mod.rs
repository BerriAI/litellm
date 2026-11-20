//! Pricing over one row: lowered once into `litellm-cost` prices, then priced or quoted per call. `query` holds the
//! whole-call entry points over the live table

mod query;

pub(crate) use query::{
    NativeBilledRates, NativeRateCard, NativeReceipt, adjust, adjustment_config, bound, price,
    rates,
};

use std::time::Duration;

use crate::model_cost::row_fields;
use jiff::Timestamp;
use litellm_cost::call::CallContext;
use litellm_cost::money::UnitPrice;
use litellm_cost::price::{
    Adjustments, CacheReadMode, CacheTtl, Channel, InputUnit, Meter, OutputUnit, Residency,
    TokenKind,
};
use litellm_cost::priced::Outcome;
use litellm_cost::query::TokenUsage;
use litellm_cost::quote::Quote;
use litellm_cost::usage::Usage;
use litellm_host_python::from_py_argument;
use litellm_model_catalog::wire::{EntryWire, Fields};
use litellm_model_catalog::{BilledPrices, billed_prices};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyMapping;
use serde_json::Value;

/// One row's prices and uplifts. A key whose value JSON cannot hold or the catalog cannot read is left out, as a
/// price main could not parse was. A row the catalog checks would reject is still priced, see `billed_prices`
#[pyclass(frozen, name = "NativePrices", module = "litellm.rust_bridge._native")]
pub(crate) struct NativePrices {
    prices: BilledPrices,
    adjustments: Adjustments,
}

#[pyclass(
    frozen,
    name = "NativeTokenUsage",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeTokenUsage(Usage);

#[pyclass(
    frozen,
    name = "NativeCallContext",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeCallContext(CallContext);

#[pyclass(
    frozen,
    get_all,
    name = "NativePriced",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativePriced {
    prompt_cost: f64,
    completion_cost: f64,
}

/// The rates one call bills at before `uplift`. A meter the row has no price for reads 0.0
#[pyclass(
    frozen,
    get_all,
    name = "NativeQuote",
    module = "litellm.rust_bridge._native"
)]
pub(crate) struct NativeQuote {
    input: f64,
    output: f64,
    reasoning: f64,
    cache_read: f64,
    explicit_cache_read: f64,
    /// `None` when the row has no cached audio price of its own
    cache_read_audio: Option<f64>,
    /// `None` when the row has no cached image price of its own
    cache_read_image: Option<f64>,
    cache_write: f64,
    cache_write_1h: f64,
    cache_write_12h: f64,
    /// The residency, FedRAMP and regional endpoint multipliers of the call, multiplied together
    uplift: f64,
}

#[pymethods]
impl NativePrices {
    /// `provider` is the call's provider and replaces the row's `litellm_provider`
    #[new]
    fn new(row: &Bound<'_, PyMapping>, provider: Option<&str>) -> PyResult<Self> {
        let wire = EntryWire::read(&fields(row, provider)?);
        Ok(Self {
            prices: billed_prices(&wire),
            adjustments: Adjustments::from(&wire),
        })
    }

    fn calculate(&self, usage: &NativeTokenUsage, call: &NativeCallContext) -> NativePriced {
        let prices = self.prices.for_prompt(usage.0.prompt_tokens);
        match litellm_cost::calculate(prices, &self.adjustments, &usage.0, &call.0) {
            Outcome::Priced(priced) => NativePriced {
                prompt_cost: priced.prompt.get(),
                completion_cost: priced.completion.get(),
            },
            Outcome::Unpriced { .. } => NativePriced {
                prompt_cost: 0.0,
                completion_cost: 0.0,
            },
        }
    }

    fn quote(&self, prompt_tokens: u64, call: &NativeCallContext) -> NativeQuote {
        let quote = self
            .prices
            .for_prompt(prompt_tokens)
            .quote(&call.0, prompt_tokens);
        let cache_write = |ttl| {
            billed(
                &quote,
                &Meter::CacheWrite {
                    kind: TokenKind::Text,
                    ttl,
                },
            )
        };
        NativeQuote {
            input: billed(&quote, &text(Meter::Input, InputUnit::Token)),
            output: billed(&quote, &text(Meter::Output, OutputUnit::Token)),
            reasoning: billed(&quote, &Meter::Output(OutputUnit::ReasoningToken)),
            cache_read: billed(
                &quote,
                &cache_read(TokenKind::Text, CacheReadMode::Implicit),
            ),
            explicit_cache_read: billed(
                &quote,
                &cache_read(TokenKind::Text, CacheReadMode::Explicit),
            ),
            cache_read_audio: own(&quote, TokenKind::Audio),
            cache_read_image: own(&quote, TokenKind::Image),
            cache_write: cache_write(CacheTtl::Default),
            cache_write_1h: cache_write(CacheTtl::OneHour),
            cache_write_12h: cache_write(CacheTtl::TwelveHours),
            uplift: self
                .adjustments
                .for_call(&call.0)
                .fold(1.0, |uplift, multiplier| uplift * multiplier.get()),
        }
    }
}

#[pymethods]
impl NativeTokenUsage {
    /// The billed quantities main's parsing already settled, as the pricing core's `TokenUsage` document reads them
    #[new]
    fn new(quantities: &Bound<'_, PyAny>) -> PyResult<Self> {
        let usage: TokenUsage = from_py_argument(quantities)?;
        for length in [usage.input_audio_seconds, usage.input_video_seconds] {
            seconds(length)?;
        }
        Ok(Self(usage.billed()))
    }
}

#[pymethods]
impl NativeCallContext {
    /// `at` is the aware moment the call is billed at, for off-peak windows
    #[new]
    #[pyo3(signature = (*, service_tier, data_residency, vertex_location, at))]
    fn new(
        service_tier: Option<&str>,
        data_residency: Option<&str>,
        vertex_location: Option<&str>,
        at: Timestamp,
    ) -> Self {
        let residency = data_residency.map(str::to_lowercase);
        Self(CallContext {
            tier: litellm_cost::candidates::service_tier_named(service_tier),
            channel: Channel::Online,
            residency: match residency.as_deref() {
                Some("eu") => Some(Residency::Eu),
                Some("us") => Some(Residency::Us),
                _ => None,
            },
            fedramp_endpoint: residency.as_deref() == Some("fedramp"),
            non_global_endpoint: vertex_location
                .is_some_and(|location| location.to_lowercase() != "global"),
            speed: None,
            inference_geo: None,
            at,
            overlay: None,
        })
    }
}

fn fields(row: &Bound<'_, PyMapping>, provider: Option<&str>) -> PyResult<Fields> {
    Ok(row_fields(row)?
        .into_iter()
        .chain(provider.map(|provider| ("litellm_provider".to_owned(), Value::from(provider))))
        .collect())
}

fn seconds(length: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(length)
        .map_err(|error| PyValueError::new_err(format!("{length} seconds: {error}")))
}

fn text<U>(side: fn(U) -> Meter, unit: fn(TokenKind) -> U) -> Meter {
    side(unit(TokenKind::Text))
}

fn cache_read(kind: TokenKind, mode: CacheReadMode) -> Meter {
    Meter::CacheRead { kind, mode }
}

fn billed(quote: &Quote<'_>, meter: &Meter) -> f64 {
    quote.rate(meter).map_or(0.0, |rate| rate.unit_price.get())
}

fn own(quote: &Quote<'_>, kind: TokenKind) -> Option<f64> {
    quote
        .rate(&cache_read(kind, CacheReadMode::Implicit))
        .filter(|rate| rate.fallback_from.is_none())
        .map(|rate| UnitPrice::get(rate.unit_price))
}
