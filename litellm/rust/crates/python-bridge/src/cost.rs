//! Pricing over one row: lowered once into `litellm-cost` prices, then priced or quoted per call

use std::collections::BTreeMap;
use std::time::Duration;

use jiff::Timestamp;
use litellm_cost::call::CallContext;
use litellm_cost::money::UnitPrice;
use litellm_cost::price::{
    Adjustments, CacheReadMode, CacheTtl, Channel, InputUnit, Meter, OutputUnit, Prices, Residency,
    ServiceTier, TokenKind,
};
use litellm_cost::priced::Outcome;
use litellm_cost::quote::Quote;
use litellm_cost::usage::{Characters, DurationMeter, Images, Search, Tokens, Usage};
use litellm_host_python::json_from_py;
use litellm_model_catalog::billed_prices;
use litellm_model_catalog::wire::{EntryWire, Fields};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyMapping, PyString};
use serde_json::Value;

/// One row's prices and uplifts. A key whose value JSON cannot hold or the catalog cannot read is left out, as a
/// price main could not parse was. A row the catalog checks would reject is still priced, see `billed_prices`
#[pyclass(frozen, name = "NativePrices", module = "litellm.rust_bridge._native")]
pub(crate) struct NativePrices {
    prices: Prices,
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
        match litellm_cost::calculate(&self.prices, &self.adjustments, &usage.0, &call.0) {
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
        let quote = self.prices.quote(&call.0, prompt_tokens);
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
    /// Quantities main's parsing already settled: uncached prompt tokens by modality, cached tokens by modality,
    /// cache writes by lifetime (`cache_write_tokens` is every write not reported as 1h or 12h), completion tokens by
    /// modality with reasoning apart. `prompt_tokens` only picks the band. `explicit_cache_read` bills the cached
    /// tokens as reads of a cache the caller created
    #[new]
    #[pyo3(signature = (
        *,
        prompt_tokens,
        text_tokens,
        audio_tokens,
        image_tokens,
        video_tokens,
        cached_text_tokens,
        cached_audio_tokens,
        cached_image_tokens,
        explicit_cache_read,
        cache_write_tokens,
        cache_write_1h_tokens,
        cache_write_12h_tokens,
        characters,
        images,
        video_seconds,
        audio_seconds,
        queries,
        output_text_tokens,
        output_audio_tokens,
        reasoning_tokens,
        output_image_tokens,
        output_video_tokens,
    ))]
    #[allow(clippy::too_many_arguments)] // one keyword per billed quantity, read by name in Python
    fn new(
        prompt_tokens: u64,
        text_tokens: u64,
        audio_tokens: u64,
        image_tokens: u64,
        video_tokens: u64,
        cached_text_tokens: u64,
        cached_audio_tokens: u64,
        cached_image_tokens: u64,
        explicit_cache_read: bool,
        cache_write_tokens: u64,
        cache_write_1h_tokens: u64,
        cache_write_12h_tokens: u64,
        characters: u64,
        images: u64,
        video_seconds: f64,
        audio_seconds: f64,
        queries: u64,
        output_text_tokens: u64,
        output_audio_tokens: u64,
        reasoning_tokens: u64,
        output_image_tokens: u64,
        output_video_tokens: u64,
    ) -> PyResult<Self> {
        let mode = if explicit_cache_read {
            CacheReadMode::Explicit
        } else {
            CacheReadMode::Implicit
        };
        Ok(Self(Usage {
            prompt_tokens,
            tokens: Tokens {
                input: BTreeMap::from([
                    (TokenKind::Text, text_tokens),
                    (TokenKind::Audio, audio_tokens),
                    (TokenKind::Image, image_tokens),
                    (TokenKind::Video, video_tokens),
                ]),
                cache_read: BTreeMap::from([
                    ((TokenKind::Text, mode), cached_text_tokens),
                    ((TokenKind::Audio, mode), cached_audio_tokens),
                    ((TokenKind::Image, mode), cached_image_tokens),
                ]),
                cache_write: BTreeMap::from([
                    ((TokenKind::Text, CacheTtl::Default), cache_write_tokens),
                    ((TokenKind::Text, CacheTtl::OneHour), cache_write_1h_tokens),
                    (
                        (TokenKind::Text, CacheTtl::TwelveHours),
                        cache_write_12h_tokens,
                    ),
                ]),
                output: BTreeMap::from([
                    (TokenKind::Text, output_text_tokens),
                    (TokenKind::Audio, output_audio_tokens),
                    (TokenKind::Image, output_image_tokens),
                    (TokenKind::Video, output_video_tokens),
                ]),
                reasoning: reasoning_tokens,
                citation: 0,
            },
            characters: Characters {
                prompt: characters,
                completion: 0,
            },
            durations: BTreeMap::from([
                (DurationMeter::InputVideo, seconds(video_seconds)?),
                (DurationMeter::InputAudio, seconds(audio_seconds)?),
            ]),
            images: Images {
                input: images,
                ..Images::default()
            },
            search: Search {
                queries,
                ..Search::default()
            },
            text_messages: 0,
        }))
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
            tier: service_tier_named(service_tier),
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

fn service_tier_named(name: Option<&str>) -> ServiceTier {
    match name.map(str::to_lowercase).as_deref() {
        Some("flex") => ServiceTier::Flex,
        Some("balanced") => ServiceTier::Balanced,
        Some("priority" | "fast") => ServiceTier::Priority,
        Some("ultrafast") => ServiceTier::Ultrafast,
        _ => ServiceTier::Standard,
    }
}

fn fields(row: &Bound<'_, PyMapping>, provider: Option<&str>) -> PyResult<Fields> {
    let written = row.items()?.iter().filter_map(|item| {
        let (key, value) = item
            .extract::<(Bound<'_, PyString>, Bound<'_, PyAny>)>()
            .ok()?;
        Some((key.to_str().ok()?.to_owned(), json_from_py(&value).ok()?))
    });
    Ok(written
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

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::none(None, ServiceTier::Standard)]
    #[case::flex(Some("flex"), ServiceTier::Flex)]
    #[case::upper(Some("PRIORITY"), ServiceTier::Priority)]
    #[case::fast_is_priority(Some("fast"), ServiceTier::Priority)]
    #[case::ultrafast(Some("ultrafast"), ServiceTier::Ultrafast)]
    #[case::balanced(Some("balanced"), ServiceTier::Balanced)]
    #[case::auto_is_standard(Some("auto"), ServiceTier::Standard)]
    #[case::unknown_is_standard(Some("scale"), ServiceTier::Standard)]
    fn a_service_tier_name_picks_the_tier_main_priced_it_at(
        #[case] name: Option<&str>,
        #[case] expected: ServiceTier,
    ) {
        assert_eq!(service_tier_named(name), expected);
    }
}
