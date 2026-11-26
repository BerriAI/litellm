//! `bound`: what to hold against a budget before a call runs, a parity port of `estimate_request_max_cost`,
//! `estimate_request_input_cost` and their helpers in `proxy/spend_tracking/budget_reservation.py`. Policy `Legacy`
//! keeps main's heuristic exactly, including where it under-holds: uplifts, discount and margin never reach the hold.
//! A counted prompt reads the `_above_<n>_tokens` prices it reaches. A true worst case is a later, separate change
//!
//! The adapter reads the request: which output caps the serving deployments honor, how many outputs it asks for, and
//! whether a Gemini body carries inline audio or names a cached content. This module reads the rows

use crate::money::UnitPrice;
use crate::price::{
    Band, CacheReadMode, InputUnit, Meter, OutputUnit, Price, Prices, ServiceTier, TokenKind, When,
};
use crate::row::{LoweredRow, RowMode, RowRef, RowSource};

/// Output tokens held when neither the request nor the row caps them, `DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK`
pub const DEFAULT_MAX_OUTPUT_TOKENS: i64 = 16_384;

const AUDIO_INPUT: Meter = Meter::Input(InputUnit::Token(TokenKind::Audio));
const AUDIO_CACHE_READ: Meter = Meter::CacheRead {
    kind: TokenKind::Audio,
    mode: CacheReadMode::Implicit,
};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct BoundQuery {
    /// One entry per model the request names
    pub models: Vec<BoundModel>,
    pub request: RequestCaps,
    #[cfg_attr(feature = "serde", serde(default))]
    pub policy: BoundPolicy,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(rename_all = "snake_case")
)]
pub enum BoundPolicy {
    #[default]
    Legacy,
}

/// The cost shapes one requested model may be served at. Reservation runs before routing, so every one is held
/// against and the highest wins
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct BoundModel {
    /// Input tokens counted for this model, `None` when the request could not be counted
    #[cfg_attr(feature = "serde", serde(default))]
    pub counted_input: Option<i64>,
    /// The output tokens the request asks this model for, `_requested_output_tokens`: the largest cap it sends, or
    /// the default when no serving deployment honors any of them
    #[cfg_attr(feature = "serde", serde(default = "default_output"))]
    pub requested_output: i64,
    /// How many outputs each serving deployment may return, `_output_multiplier`
    #[cfg_attr(feature = "serde", serde(default = "one"))]
    pub output_multiplier: i64,
    pub candidates: Vec<CostShape>,
}

#[cfg(feature = "serde")]
fn default_output() -> i64 {
    DEFAULT_MAX_OUTPUT_TOKENS
}

#[cfg(feature = "serde")]
fn one() -> i64 {
    1
}

/// Flat rates, optionally beside a deployment's `tiered_pricing` table
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct CostShape {
    pub flat: FlatSource,
    /// The deployment row whose `tiered_pricing` replaces the flat rates for a counted prompt
    #[cfg_attr(feature = "serde", serde(default))]
    pub tiers: Option<RowRef>,
    /// The prices each deployment of the group bills at, its `deployment_pricing`, read for the audio, cache-read
    /// and `_above_<n>_tokens` rates a group's own prices leave out
    #[cfg_attr(feature = "serde", serde(default))]
    pub deployments: Vec<RowRef>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(tag = "from", rename_all = "snake_case")
)]
pub enum FlatSource {
    /// A model info mapping, as `get_model_info` or the caller returned it
    Row { row: RowRef },
    /// The group rates the router reports (`ModelGroupInfo`), which carry no reasoning, cache or image rates
    Group(GroupRates),
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize), serde(default))]
pub struct GroupRates {
    pub input: Option<UnitPrice>,
    pub output: Option<UnitPrice>,
    pub max_input_tokens: Option<i64>,
    pub max_output_tokens: Option<i64>,
}

/// The request facts the hold reads beside each model's output caps
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize), serde(default))]
pub struct RequestCaps {
    /// `n` as sent, read by the per-image hold only
    pub n: Option<i64>,
    /// An embeddings, rerank or moderations route, which produces no output
    pub input_only: bool,
    /// A Gemini body carrying inline audio, held at the higher of the text and audio input rates
    pub inline_audio: bool,
    /// A Gemini body naming a cached content, whose tokens up to the row's input window are held at the cache-read
    /// rate
    pub cached_content: bool,
}

/// `max` is the hold, `input_floor` what a cancelled call has already spent. `None` when no candidate prices
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bound {
    pub max: Option<f64>,
    pub input_floor: Option<f64>,
}

/// Where a shape's own rates come from: a lowered row, or the router's group rates
enum Own<'a> {
    Row(&'a Prices),
    Group(&'a GroupRates),
}

/// The rates one cost shape is read at
struct Shape<'a> {
    own: Own<'a>,
    max_input_tokens: Option<i64>,
    max_output_tokens: Option<i64>,
    image: Option<f64>,
    tiers: Option<&'a Prices>,
    deployments: Vec<&'a Prices>,
}

/// `_HoldRates`: the per-token rates one shape holds a prompt at
struct HoldRates {
    input: Option<f64>,
    cached: Option<f64>,
    output: Option<f64>,
}

/// What one model holds before it runs
struct Held {
    counted: Option<i64>,
    requested_output: i64,
    output_multiplier: i64,
}

pub fn bound(source: &impl RowSource, query: &BoundQuery) -> Bound {
    let BoundPolicy::Legacy = query.policy;
    let request = &query.request;
    let per_model = |estimate: &dyn Fn(&Shape<'_>, &Held) -> Option<f64>| {
        highest(query.models.iter().filter_map(|model| {
            let held = Held {
                counted: model.counted_input,
                requested_output: model.requested_output,
                output_multiplier: model.output_multiplier,
            };
            highest(
                model
                    .candidates
                    .iter()
                    .filter_map(|shape| Shape::read(source, shape))
                    .filter_map(|shape| estimate(&shape, &held)),
            )
        }))
    };
    Bound {
        max: per_model(&|shape, held| shape.max_cost(request, held)),
        input_floor: per_model(&|shape, held| shape.input_cost(request, held.counted)),
    }
}

fn highest(estimates: impl Iterator<Item = f64>) -> Option<f64> {
    estimates.reduce(|kept, next| if next > kept { next } else { kept })
}

impl<'a> Shape<'a> {
    /// `None` where main's `_get_model_pricing` raises and holds nothing for the shape
    fn read<S: RowSource>(source: &'a S, shape: &'a CostShape) -> Option<Self> {
        let tiers = shape
            .tiers
            .as_ref()
            .and_then(|at| source.row(at))
            .and_then(|row| row.prices().tiers());
        let deployments = shape
            .deployments
            .iter()
            .filter_map(|at| source.row(at))
            .map(|row| row.prices().flat())
            .collect();
        Some(match &shape.flat {
            FlatSource::Group(rates) => Self {
                own: Own::Group(rates),
                max_input_tokens: rates.max_input_tokens,
                max_output_tokens: rates.max_output_tokens,
                image: None,
                tiers,
                deployments,
            },
            FlatSource::Row { row: at } => {
                let row = source.row(at)?;
                let flat = row.prices().flat();
                let facts = row.facts();
                let image_generation =
                    matches!(facts.mode, RowMode::ImageGeneration | RowMode::ImageEdit);
                Self {
                    own: Own::Row(flat),
                    max_input_tokens: facts
                        .max_input_tokens
                        .and_then(|tokens| i64::try_from(tokens).ok()),
                    max_output_tokens: facts
                        .max_output_tokens
                        .and_then(|tokens| i64::try_from(tokens).ok()),
                    image: image_generation.then(|| image_cost(flat)).flatten(),
                    tiers: tiers.or_else(|| row.prices().tiers()),
                    deployments,
                }
            }
        })
    }

    /// `max_cost_for_cost_info`
    fn max_cost(&self, request: &RequestCaps, held: &Held) -> Option<f64> {
        if let Some(per_image) = self.image {
            return Some(per_image * request.n.filter(|n| *n != 0).unwrap_or(1).max(1) as f64);
        }
        let input = held.counted.or(self.max_input_tokens)?;
        let output = if request.input_only {
            0
        } else {
            let ceiling = self
                .max_output_tokens
                .filter(|tokens| *tokens > 0)
                .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);
            held.requested_output.min(ceiling)
        };
        let cached = self.cached_content_tokens(request, input);
        let rates = self.hold_rates(request, input + cached, held.counted.is_some());
        let prompt = prompt_cost(&rates, input, cached)?;
        if rates.output.is_none() && output > 0 {
            return None;
        }
        Some(prompt + (output * held.output_multiplier) as f64 * rates.output.unwrap_or(0.0))
    }

    /// `input_cost_for_cost_info`
    fn input_cost(&self, request: &RequestCaps, counted: Option<i64>) -> Option<f64> {
        let input = counted.or(self.max_input_tokens)?;
        let cached = self.cached_content_tokens(request, input);
        let rates = self.hold_rates(request, input + cached, counted.is_some());
        prompt_cost(&rates, input, cached)
    }

    /// `_cached_content_tokens`: a body naming a cached content holds the rest of the row's input window as cached
    fn cached_content_tokens(&self, request: &RequestCaps, input: i64) -> i64 {
        match self.max_input_tokens {
            Some(window) if request.cached_content => (window - input).max(0),
            _ => 0,
        }
    }

    /// `_hold_rates`: the rates of a prompt of `prompt_tokens`, cached tokens included, which pick the tier or, once
    /// the prompt was counted, the `_above_<n>_tokens` prices. Output holds at the higher of the output and
    /// reasoning rates, since the reasoning share is unknown before the call runs
    fn hold_rates(&self, request: &RequestCaps, prompt_tokens: i64, counted: bool) -> HoldRates {
        if let Some(tier) = self.tier(prompt_tokens) {
            let text = tier.rate(&Meter::TEXT_INPUT);
            return HoldRates {
                input: Some(if request.inline_audio {
                    text.max(tier.rate(&AUDIO_INPUT))
                } else {
                    text
                }),
                cached: Some(
                    tier.own(&Meter::CACHE_READ)
                        .unwrap_or_else(|| tier.rate(&Meter::TEXT_INPUT)),
                ),
                output: Some(
                    tier.rate(&Meter::TEXT_OUTPUT)
                        .max(tier.rate(&Meter::REASONING)),
                ),
            };
        }
        let sized = counted
            .then(|| u64::try_from(prompt_tokens).ok())
            .flatten();
        let input_meters: &[Meter] = if request.inline_audio {
            &[Meter::TEXT_INPUT, AUDIO_INPUT]
        } else {
            &[Meter::TEXT_INPUT]
        };
        let output = [Meter::TEXT_OUTPUT, Meter::REASONING]
            .iter()
            .filter_map(|meter| {
                self.prompt_size_rate(sized, self.own(meter), std::slice::from_ref(meter))
            })
            .reduce(f64::max);
        HoldRates {
            input: self.prompt_size_rate(sized, self.input_rate(request), input_meters),
            cached: self.prompt_size_rate(
                sized,
                self.cache_read_rate(),
                &[Meter::CACHE_READ, AUDIO_CACHE_READ],
            ),
            output,
        }
    }

    /// The shape's own price for `meter` at the base band, read by plain key. A group lists input and output only
    fn own(&self, meter: &Meter) -> Option<f64> {
        match self.own {
            Own::Row(prices) => base(prices, meter.clone()),
            Own::Group(rates) => match meter {
                Meter::Input(InputUnit::Token(TokenKind::Text)) => rates.input,
                Meter::Output(OutputUnit::Token(TokenKind::Text)) => rates.output,
                _ => None,
            }
            .map(UnitPrice::get),
        }
    }

    /// The price tables `_above_<n>_tokens` prices are read from: the shape's own, then each deployment's
    fn tables(&self) -> impl Iterator<Item = &'a Prices> + '_ {
        let own = match self.own {
            Own::Row(prices) => Some(prices),
            Own::Group(_) => None,
        };
        own.into_iter().chain(self.deployments.iter().copied())
    }

    /// `_input_rate`: the text rate, raised to the highest audio rate a deployment bills when the body holds audio
    fn input_rate(&self, request: &RequestCaps) -> Option<f64> {
        let text = self.own(&Meter::TEXT_INPUT);
        let audio = request
            .inline_audio
            .then(|| self.highest_deployment_rate(&AUDIO_INPUT))
            .flatten();
        match (text, audio) {
            (Some(text), Some(audio)) => Some(text.max(audio)),
            _ => text,
        }
    }

    /// `_cache_read_rate`: the highest cache-read price any deployment lists for text or audio, since the hold
    /// cannot see what the cache holds
    fn cache_read_rate(&self) -> Option<f64> {
        let text = self.highest_deployment_rate(&Meter::CACHE_READ)?;
        Some(
            self.listed(AUDIO_CACHE_READ)
                .fold(text, |highest, rate| highest.max(rate)),
        )
    }

    /// `_listed_rates`: the base price of `meter` in every table that lists it
    fn listed(&self, meter: Meter) -> impl Iterator<Item = f64> + '_ {
        let own = self.own(&meter);
        own.into_iter().chain(
            self.deployments
                .iter()
                .filter_map(move |prices| base(prices, meter.clone())),
        )
    }

    /// `_highest_deployment_rate`: each deployment's price for `meter`, or its input price when it lists none. With
    /// no deployment pricing the shape's own price stands, or its input price. `None` when any of them is unknown
    fn highest_deployment_rate(&self, meter: &Meter) -> Option<f64> {
        let input = self.own(&Meter::TEXT_INPUT);
        if self.deployments.is_empty() {
            return self.own(meter).or(input);
        }
        self.deployments
            .iter()
            .map(|prices| {
                base(prices, meter.clone())
                    .or_else(|| base(prices, Meter::TEXT_INPUT))
                    .or(input)
            })
            .try_fold(f64::NEG_INFINITY, |highest, rate| Some(highest.max(rate?)))
    }

    /// `_prompt_size_rate`: `rate` raised to the highest `_above_<n>_tokens` price of `meters` a prompt of
    /// `prompt_tokens` reaches in any table, the threshold itself included. A prompt that was not sized keeps `rate`
    fn prompt_size_rate(
        &self,
        prompt_tokens: Option<u64>,
        rate: Option<f64>,
        meters: &[Meter],
    ) -> Option<f64> {
        let (rate, prompt_tokens) = (rate?, prompt_tokens);
        let Some(prompt_tokens) = prompt_tokens else {
            return Some(rate);
        };
        Some(
            self.tables()
                .flat_map(|prices| {
                    meters
                        .iter()
                        .filter_map(move |meter| above_threshold_rate(prices, meter, prompt_tokens))
                })
                .fold(rate, f64::max),
        )
    }

    /// `select_tier_for_input`: the highest tier whose start the prompt passes. A prompt of no tokens picks none
    fn tier(&self, input: i64) -> Option<Tier<'a>> {
        let tiers = self.tiers?;
        let prompt = u64::try_from(input).ok().filter(|tokens| *tokens > 0)?;
        let band = tiers
            .iter()
            .map(|(price, _)| price.when.band)
            .filter(|band| match band {
                Band::Base => true,
                Band::OverPromptTokens(floor) => prompt > floor.get(),
            })
            .max()?;
        Some(Tier { tiers, band })
    }
}

/// `_prompt_cost`: the prompt's uncached and cached tokens at their rates. `None` when tokens have no rate
fn prompt_cost(rates: &HoldRates, input: i64, cached: i64) -> Option<f64> {
    let priced = [(input, rates.input), (cached, rates.cached)];
    if priced
        .iter()
        .any(|(tokens, rate)| rate.is_none() && *tokens > 0)
    {
        return None;
    }
    Some(priced.iter().fold(0.0, |sum, (tokens, rate)| {
        sum + *tokens as f64 * rate.unwrap_or(0.0)
    }))
}

/// `_above_threshold_rate`: the price of `meter` at the highest `_above_<n>_tokens` band a prompt of `prompt_tokens`
/// reaches, the threshold itself included. `None` when it reaches none
fn above_threshold_rate(prices: &Prices, meter: &Meter, prompt_tokens: u64) -> Option<f64> {
    prices
        .iter()
        .filter_map(|(price, rate)| match price.when {
            When {
                tier: ServiceTier::Standard,
                band: Band::OverPromptTokens(floor),
            } if price.meter == *meter
                && price.channel == crate::price::Channel::Online
                && prompt_tokens >= floor.get() =>
            {
                Some((floor.get(), rate.get()))
            }
            _ => None,
        })
        .max_by_key(|(floor, _)| *floor)
        .map(|(_, rate)| rate)
}

/// The tier of a `tiered_pricing` table a prompt picked, as the band it starts at
struct Tier<'a> {
    tiers: &'a Prices,
    band: Band,
}

impl Tier<'_> {
    /// `tier_rate`: the tier's own rate, 0.0 when it has none
    fn rate(&self, meter: &Meter) -> f64 {
        self.own(meter).unwrap_or(0.0)
    }

    fn own(&self, meter: &Meter) -> Option<f64> {
        self.tiers
            .get(&Price::online(
                meter.clone(),
                When {
                    tier: ServiceTier::Standard,
                    band: self.band,
                },
            ))
            .map(UnitPrice::get)
    }
}

fn base(prices: &Prices, meter: Meter) -> Option<f64> {
    prices
        .get(&Price::online(meter, When::default()))
        .map(UnitPrice::get)
}

/// `_estimate_image_generation_cost`: the input and output per-image prices summed, nothing when that is not positive
fn image_cost(prices: &Prices) -> Option<f64> {
    let per_image = base(prices, Meter::Output(OutputUnit::Image(None))).unwrap_or(0.0)
        + base(prices, Meter::Input(InputUnit::Image)).unwrap_or(0.0);
    (per_image > 0.0).then_some(per_image)
}

#[cfg(test)]
mod parity {
    //! Parity with main's legacy budget hold. `fixtures/bound_parity.json` was captured once from main's Python
    //! (f631dbd13213, `LITELLM_LOCAL_MODEL_COST_MAP=True`): `estimate_request_max_cost` and `estimate_request_input_cost`
    //! on catalog rows without a router, and `_max_cost_for_cost_info` and `_input_cost_for_cost_info` on the group rates
    //! a router reports. Since then, the three cases whose counted prompt passes an `_above_<n>_tokens` threshold hold
    //! at those prices, which main read only off a Gemini `contents` body

    use rstest::rstest;
    use serde_json::Value;

    use super::*;
    use crate::testing::{TestSource, fixture_row, row_ref};

    const FIXTURE: &str = include_str!("../fixtures/bound_parity.json");

    fn case(name: &str) -> Value {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("no fixture case {name}"))
            .clone()
    }

    /// What the adapter reads off the request body and route for these single-cap bodies: the cap sent, or the
    /// default, and the larger of `n` and `best_of`
    fn caps(body: &Value, route: &str) -> (RequestCaps, i64, i64) {
        let requested = [
            &body["max_completion_tokens"],
            &body["max_tokens"],
            &body["max_output_tokens"],
            &body["inferenceConfig"]["maxTokens"],
        ];
        let request = RequestCaps {
            n: body["n"].as_i64(),
            input_only: ["embeddings", "rerank", "moderations"]
                .iter()
                .any(|part| route.contains(part)),
            ..RequestCaps::default()
        };
        let output = requested
            .into_iter()
            .find_map(Value::as_i64)
            .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);
        let multiplier = [body["n"].as_i64(), body["best_of"].as_i64()]
            .into_iter()
            .flatten()
            .fold(1, i64::max);
        (request, output, multiplier)
    }

    /// The row as `get_model_info` returns it, which reads a missing input or output rate as 0
    fn model_info(row: &Value) -> Value {
        let mut info = row.clone();
        for key in ["input_cost_per_token", "output_cost_per_token"] {
            info.as_object_mut()
                .unwrap()
                .entry(key)
                .or_insert(Value::from(0.0));
        }
        info
    }

    fn bound_for(case: &Value) -> Bound {
        let row = &case["row"];
        let (source, shape) = match case["shape"].as_str() {
            Some("group") => (
                TestSource::default(),
                CostShape {
                    flat: FlatSource::Group(GroupRates {
                        input: row["input_cost_per_token"]
                            .as_f64()
                            .and_then(UnitPrice::new),
                        output: row["output_cost_per_token"]
                            .as_f64()
                            .and_then(UnitPrice::new),
                        max_input_tokens: row["max_input_tokens"].as_i64(),
                        max_output_tokens: row["max_output_tokens"].as_i64(),
                    }),
                    tiers: None,
                    deployments: vec![],
                },
            ),
            _ => (
                TestSource::with_row("row", fixture_row(&model_info(row))),
                CostShape {
                    flat: FlatSource::Row {
                        row: row_ref("row"),
                    },
                    tiers: None,
                    deployments: vec![],
                },
            ),
        };
        let (request, requested_output, output_multiplier) =
            caps(&case["body"], case["route"].as_str().unwrap());
        bound(
            &source,
            &BoundQuery {
                models: vec![BoundModel {
                    counted_input: case["counted"].as_i64(),
                    requested_output,
                    output_multiplier,
                    candidates: vec![shape],
                }],
                request,
                policy: BoundPolicy::Legacy,
            },
        )
    }

    #[rstest]
    #[case::anthropic_requested_output("anthropic_requested_output")]
    #[case::anthropic_default_output("anthropic_default_output")]
    #[case::anthropic_output_clamped_to_row("anthropic_output_clamped_to_row")]
    #[case::bedrock_n_and_best_of("bedrock_n_and_best_of")]
    #[case::bedrock_inference_config("bedrock_inference_config")]
    #[case::vertex_max_completion_tokens("vertex_max_completion_tokens")]
    #[case::azure_ai_input_only_route("azure_ai_input_only_route")]
    #[case::tiered_low("tiered_low")]
    #[case::tiered_high("tiered_high")]
    #[case::tiered_empty_prompt("tiered_empty_prompt")]
    #[case::image_generation("image_generation")]
    #[case::group_uncounted_reads_max_input("group_uncounted_reads_max_input")]
    #[case::group_missing_output_rate("group_missing_output_rate")]
    #[case::group_zero_max_output_reads_default("group_zero_max_output_reads_default")]
    fn rust_holds_what_main_held(#[case] name: &str) {
        let case = case(name);
        let bound = bound_for(&case);

        assert_eq!(bound.max, case["expected"]["max"].as_f64());
        assert_eq!(bound.input_floor, case["expected"]["input"].as_f64());
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::price::ServiceTier::Standard;
    use crate::row::RowFacts;
    use crate::testing::{BASE, TestRow, TestSource, over, prices, row_ref, text_row};

    /// What the adapter reads off the request for one model
    #[derive(Clone, Copy)]
    struct Ask {
        output: i64,
        multiplier: i64,
        request: RequestCaps,
    }

    fn capped(output: i64) -> Ask {
        Ask {
            output,
            multiplier: 1,
            request: RequestCaps::default(),
        }
    }

    fn model(counted: Option<i64>, shapes: Vec<CostShape>, ask: Ask) -> BoundModel {
        BoundModel {
            counted_input: counted,
            requested_output: ask.output,
            output_multiplier: ask.multiplier,
            candidates: shapes,
        }
    }

    fn query(counted: Option<i64>, shapes: Vec<CostShape>, ask: Ask) -> BoundQuery {
        BoundQuery {
            models: vec![model(counted, shapes, ask)],
            request: ask.request,
            policy: BoundPolicy::Legacy,
        }
    }

    fn row(model: &str) -> CostShape {
        CostShape {
            flat: FlatSource::Row {
                row: row_ref(model),
            },
            tiers: None,
            deployments: vec![],
        }
    }

    fn group(input: Option<f64>, output: Option<f64>) -> CostShape {
        CostShape {
            flat: FlatSource::Group(GroupRates {
                input: input.and_then(UnitPrice::new),
                output: output.and_then(UnitPrice::new),
                max_input_tokens: Some(1_000),
                max_output_tokens: Some(100),
            }),
            tiers: None,
            deployments: vec![],
        }
    }

    fn with_deployments(shape: CostShape, deployments: &[&str]) -> CostShape {
        CostShape {
            deployments: deployments.iter().map(|key| row_ref(key)).collect(),
            ..shape
        }
    }

    fn priced(rates: &[(Meter, Band, f64)]) -> TestRow {
        TestRow {
            prices: prices(
                &rates
                    .iter()
                    .map(|(meter, band, rate)| (meter.clone(), Standard, *band, *rate))
                    .collect::<Vec<_>>(),
            ),
            ..TestRow::default()
        }
    }

    fn windowed(row: TestRow, max_input_tokens: u64) -> TestRow {
        TestRow {
            facts: RowFacts {
                max_input_tokens: Some(max_input_tokens),
                ..RowFacts::default()
            },
            ..row
        }
    }

    fn gemini(inline_audio: bool, cached_content: bool) -> RequestCaps {
        RequestCaps {
            inline_audio,
            cached_content,
            ..RequestCaps::default()
        }
    }

    #[rstest]
    #[case::counted_input_and_requested_output(Some(10), capped(5), Some(10.0 + 5.0 * 4.0))]
    #[case::uncounted_reads_max_input(None, capped(5), Some(1_000.0 + 20.0))]
    #[case::output_capped_by_the_row(Some(10), capped(500), Some(10.0 + 100.0 * 4.0))]
    #[case::default_output_capped_by_the_row(
        Some(10),
        capped(DEFAULT_MAX_OUTPUT_TOKENS),
        Some(10.0 + 400.0)
    )]
    #[case::the_multiplier_multiplies_output(
        Some(10),
        Ask { multiplier: 3, ..capped(5) },
        Some(10.0 + 60.0)
    )]
    #[case::input_only_route(
        Some(10),
        Ask { request: RequestCaps { input_only: true, ..RequestCaps::default() }, ..capped(5) },
        Some(10.0)
    )]
    fn the_hold_is_input_plus_capped_output(
        #[case] counted: Option<i64>,
        #[case] ask: Ask,
        #[case] expected: Option<f64>,
    ) {
        let source = TestSource::default();
        assert_eq!(
            bound(
                &source,
                &query(counted, vec![group(Some(1.0), Some(4.0))], ask)
            )
            .max,
            expected
        );
    }

    #[rstest]
    #[case::missing_input_rate_with_input(group(None, Some(4.0)), None)]
    #[case::missing_output_rate_with_output(group(Some(1.0), None), None)]
    fn a_missing_rate_for_used_tokens_holds_nothing(
        #[case] shape: CostShape,
        #[case] expected: Option<f64>,
    ) {
        assert_eq!(
            bound(
                &TestSource::default(),
                &query(Some(10), vec![shape], capped(5))
            )
            .max,
            expected
        );
    }

    #[rstest]
    fn a_row_without_an_input_rate_holds_an_empty_prompt_at_nothing() {
        let source =
            TestSource::with_row("output_only", priced(&[(Meter::TEXT_OUTPUT, BASE, 4.0)]));
        let bound = bound(
            &source,
            &query(Some(0), vec![row("output_only")], capped(5)),
        );
        assert_eq!((bound.max, bound.input_floor), (Some(20.0), Some(0.0)));
    }

    #[rstest]
    #[case::zero(0)]
    #[case::negative(-1)]
    fn a_row_ceiling_that_is_not_positive_reads_the_default(#[case] ceiling: i64) {
        let shape = CostShape {
            flat: FlatSource::Group(GroupRates {
                input: UnitPrice::new(1.0),
                output: UnitPrice::new(1.0),
                max_input_tokens: None,
                max_output_tokens: Some(ceiling),
            }),
            tiers: None,
            deployments: vec![],
        };
        assert_eq!(
            bound(
                &TestSource::default(),
                &query(Some(0), vec![shape], capped(DEFAULT_MAX_OUTPUT_TOKENS + 1))
            )
            .max,
            Some(DEFAULT_MAX_OUTPUT_TOKENS as f64)
        );
    }

    #[rstest]
    fn each_model_holds_its_own_output_caps() {
        let source = TestSource::default();
        let bound = bound(
            &source,
            &BoundQuery {
                models: vec![
                    model(Some(10), vec![group(Some(1.0), Some(4.0))], capped(5)),
                    model(
                        Some(10),
                        vec![group(Some(1.0), Some(1.0))],
                        Ask {
                            multiplier: 2,
                            ..capped(50)
                        },
                    ),
                ],
                request: RequestCaps::default(),
                policy: BoundPolicy::Legacy,
            },
        );
        assert_eq!(bound.max, Some(10.0 + 100.0));
    }

    #[rstest]
    fn the_hold_is_the_highest_shape_and_model() {
        let source =
            TestSource::with_row("cheap", text_row(1.0, 1.0)).and_row("dear", text_row(2.0, 8.0));
        let bound = bound(
            &source,
            &BoundQuery {
                models: vec![
                    model(Some(10), vec![row("cheap"), row("missing")], capped(5)),
                    model(Some(10), vec![row("dear")], capped(5)),
                ],
                request: RequestCaps::default(),
                policy: BoundPolicy::Legacy,
            },
        );
        assert_eq!(bound.max, Some(20.0 + 40.0));
        assert_eq!(bound.input_floor, Some(20.0));
    }

    fn banded() -> TestRow {
        priced(&[
            (Meter::TEXT_INPUT, BASE, 1.0),
            (Meter::TEXT_INPUT, over(5), 100.0),
            (Meter::TEXT_INPUT, over(10), 200.0),
            (Meter::TEXT_OUTPUT, BASE, 2.0),
            (Meter::TEXT_OUTPUT, over(10), 20.0),
            (Meter::REASONING, BASE, 3.0),
        ])
    }

    #[rstest]
    #[case::an_uncounted_body(None, RequestCaps::default(), None)]
    #[case::a_counted_chat_body_reaches_the_threshold_it_equals(
        Some(10),
        RequestCaps::default(),
        Some(10.0 * 200.0 + 5.0 * 20.0)
    )]
    #[case::a_counted_gemini_body_reaches_the_threshold_it_equals(
        Some(10),
        gemini(false, false),
        Some(10.0 * 200.0 + 5.0 * 20.0)
    )]
    #[case::a_counted_body_below_a_threshold(Some(9), RequestCaps::default(), Some(9.0 * 100.0 + 5.0 * 3.0))]
    fn only_a_counted_body_reads_the_above_threshold_prices(
        #[case] counted: Option<i64>,
        #[case] request: RequestCaps,
        #[case] expected: Option<f64>,
    ) {
        let source = TestSource::with_row("banded", banded());
        let ask = Ask {
            request,
            ..capped(5)
        };
        assert_eq!(
            bound(&source, &query(counted, vec![row("banded")], ask)).max,
            expected
        );
    }

    #[rstest]
    fn a_group_reads_the_above_threshold_prices_of_its_deployments() {
        let source = TestSource::with_row(
            "long_context",
            priced(&[
                (Meter::TEXT_INPUT, BASE, 1.0),
                (Meter::TEXT_INPUT, over(100), 3.0),
                (Meter::TEXT_OUTPUT, over(100), 9.0),
            ]),
        );
        let shape = with_deployments(group(Some(1.0), Some(4.0)), &["long_context"]);
        let ask = Ask {
            request: gemini(false, false),
            ..capped(5)
        };
        let bound = bound(&source, &query(Some(200), vec![shape], ask));
        assert_eq!(
            (bound.max, bound.input_floor),
            (Some(200.0 * 3.0 + 5.0 * 9.0), Some(600.0))
        );
    }

    #[rstest]
    #[case::the_higher_audio_rate(&["audio"], Some(10.0 * 7.0))]
    #[case::the_input_rate_of_a_deployment_without_one(&["audio", "text_only"], Some(10.0 * 7.0))]
    #[case::the_text_rate_when_it_is_higher(&["cheap_audio"], Some(10.0 * 2.0))]
    #[case::the_text_rate_without_deployments(&[], Some(10.0 * 2.0))]
    fn inline_audio_holds_input_at(#[case] deployments: &[&str], #[case] expected: Option<f64>) {
        let source = TestSource::with_row(
            "audio",
            priced(&[(Meter::TEXT_INPUT, BASE, 1.0), (AUDIO_INPUT, BASE, 7.0)]),
        )
        .and_row("text_only", priced(&[(Meter::TEXT_INPUT, BASE, 5.0)]))
        .and_row("cheap_audio", priced(&[(AUDIO_INPUT, BASE, 0.5)]));
        let shape = with_deployments(group(Some(2.0), Some(4.0)), deployments);
        let ask = Ask {
            request: gemini(true, false),
            ..capped(0)
        };
        assert_eq!(
            bound(&source, &query(Some(10), vec![shape], ask)).input_floor,
            expected
        );
    }

    #[rstest]
    fn an_audio_rate_a_deployment_cannot_price_leaves_the_text_rate() {
        let source = TestSource::with_row("unpriced", priced(&[]));
        let shape = with_deployments(group(None, Some(4.0)), &["unpriced"]);
        let ask = Ask {
            request: gemini(true, false),
            ..capped(0)
        };
        assert_eq!(
            bound(&source, &query(Some(0), vec![shape], ask)).input_floor,
            Some(0.0)
        );
    }

    #[rstest]
    #[case::the_cache_read_rate(
        priced(&[(Meter::TEXT_INPUT, BASE, 1.0), (Meter::CACHE_READ, BASE, 0.25), (Meter::TEXT_OUTPUT, BASE, 4.0)]),
        Some(10.0 + 90.0 * 0.25)
    )]
    #[case::the_higher_audio_cache_read_rate(
        priced(&[
            (Meter::TEXT_INPUT, BASE, 1.0),
            (Meter::CACHE_READ, BASE, 0.25),
            (AUDIO_CACHE_READ, BASE, 0.5),
            (Meter::TEXT_OUTPUT, BASE, 4.0),
        ]),
        Some(10.0 + 90.0 * 0.5)
    )]
    #[case::the_input_rate_without_a_cache_read_rate(
        priced(&[(Meter::TEXT_INPUT, BASE, 1.0), (Meter::TEXT_OUTPUT, BASE, 4.0)]),
        Some(10.0 + 90.0)
    )]
    fn a_cached_content_holds_the_rest_of_the_window_at(
        #[case] row_prices: TestRow,
        #[case] expected: Option<f64>,
    ) {
        let source = TestSource::with_row("cached", windowed(row_prices, 100));
        let ask = Ask {
            request: gemini(false, true),
            ..capped(0)
        };
        assert_eq!(
            bound(&source, &query(Some(10), vec![row("cached")], ask)).input_floor,
            expected
        );
    }

    #[rstest]
    fn a_cached_content_without_an_input_window_holds_no_cached_tokens() {
        let source = TestSource::with_row(
            "unwindowed",
            priced(&[
                (Meter::TEXT_INPUT, BASE, 1.0),
                (Meter::CACHE_READ, BASE, 0.25),
            ]),
        );
        let ask = Ask {
            request: gemini(false, true),
            ..capped(0)
        };
        assert_eq!(
            bound(&source, &query(Some(10), vec![row("unwindowed")], ask)).input_floor,
            Some(10.0)
        );
    }

    #[rstest]
    fn cached_tokens_without_any_rate_hold_nothing() {
        let source = TestSource::with_row("unpriced", priced(&[]));
        let shape = CostShape {
            flat: FlatSource::Group(GroupRates {
                max_input_tokens: Some(100),
                ..GroupRates::default()
            }),
            tiers: None,
            deployments: vec![row_ref("unpriced")],
        };
        let ask = Ask {
            request: gemini(false, true),
            ..capped(0)
        };
        assert_eq!(
            bound(&source, &query(Some(0), vec![shape], ask)).input_floor,
            None
        );
    }

    #[rstest]
    fn reasoning_raises_the_output_rate() {
        let source = TestSource::with_row(
            "reasoner",
            priced(&[
                (Meter::TEXT_INPUT, BASE, 1.0),
                (Meter::TEXT_OUTPUT, BASE, 2.0),
                (Meter::REASONING, BASE, 3.0),
            ]),
        );
        assert_eq!(
            bound(&source, &query(Some(10), vec![row("reasoner")], capped(5))).max,
            Some(10.0 + 15.0)
        );
    }

    fn tiered() -> TestRow {
        TestRow {
            prices: prices(&[
                (Meter::TEXT_INPUT, Standard, BASE, 1.0),
                (Meter::TEXT_OUTPUT, Standard, BASE, 4.0),
            ]),
            tiers: Some(prices(&[
                (Meter::TEXT_INPUT, Standard, BASE, 1.0),
                (Meter::TEXT_OUTPUT, Standard, BASE, 4.0),
                (Meter::TEXT_INPUT, Standard, over(256), 2.0),
                (AUDIO_INPUT, Standard, over(256), 3.0),
                (Meter::CACHE_READ, Standard, over(256), 0.5),
                (Meter::TEXT_OUTPUT, Standard, over(256), 8.0),
                (Meter::REASONING, Standard, over(256), 9.0),
            ])),
            facts: RowFacts {
                max_input_tokens: Some(400),
                ..RowFacts::default()
            },
            ..TestRow::default()
        }
    }

    #[rstest]
    #[case::counted_prompt_reads_its_tier(Some(300), Some(300.0 * 2.0 + 5.0 * 9.0), Some(600.0))]
    #[case::empty_prompt_reads_the_flat_rates(Some(0), Some(5.0 * 4.0), Some(0.0))]
    fn a_tiered_row_holds_at_the_tier_the_prompt_picks(
        #[case] counted: Option<i64>,
        #[case] max: Option<f64>,
        #[case] floor: Option<f64>,
    ) {
        let source = TestSource::with_row("tiered", tiered());
        let bound = bound(&source, &query(counted, vec![row("tiered")], capped(5)));
        assert_eq!((bound.max, bound.input_floor), (max, floor));
    }

    #[rstest]
    #[case::cached_tokens_pick_the_tier_at_its_cache_rate(gemini(false, true), Some(100.0 * 2.0 + 300.0 * 0.5))]
    #[case::inline_audio_reads_the_tier_audio_rate(gemini(true, false), Some(100.0 * 1.0))]
    #[case::audio_in_a_tier_that_prices_it(
        RequestCaps { inline_audio: true, cached_content: true, ..gemini(true, true) },
        Some(100.0 * 3.0 + 300.0 * 0.5)
    )]
    fn a_gemini_body_holds_its_tier_at(#[case] request: RequestCaps, #[case] floor: Option<f64>) {
        let source = TestSource::with_row("tiered", tiered());
        let ask = Ask {
            request,
            ..capped(0)
        };
        assert_eq!(
            bound(&source, &query(Some(100), vec![row("tiered")], ask)).input_floor,
            floor
        );
    }

    #[rstest]
    fn a_tier_without_a_cache_rate_reads_its_input_rate() {
        let source = TestSource::with_row(
            "tiered",
            TestRow {
                tiers: Some(prices(&[(Meter::TEXT_INPUT, Standard, BASE, 2.0)])),
                facts: RowFacts {
                    max_input_tokens: Some(100),
                    ..RowFacts::default()
                },
                ..TestRow::default()
            },
        );
        let ask = Ask {
            request: gemini(false, true),
            ..capped(0)
        };
        assert_eq!(
            bound(&source, &query(Some(10), vec![row("tiered")], ask)).input_floor,
            Some(10.0 * 2.0 + 90.0 * 2.0)
        );
    }

    #[rstest]
    #[case::per_image(RowMode::ImageGeneration, Some(2), Some(0.08))]
    #[case::no_n_is_one(RowMode::ImageEdit, None, Some(0.04))]
    #[case::chat_rows_price_tokens(RowMode::Chat, Some(2), Some(0.0))]
    fn an_image_generation_row_holds_per_image(
        #[case] mode: RowMode,
        #[case] n: Option<i64>,
        #[case] expected: Option<f64>,
    ) {
        let source = TestSource::with_row(
            "painter",
            TestRow {
                prices: prices(&[
                    (Meter::TEXT_INPUT, Standard, BASE, 0.0),
                    (Meter::TEXT_OUTPUT, Standard, BASE, 0.0),
                    (Meter::Output(OutputUnit::Image(None)), Standard, BASE, 0.04),
                ]),
                facts: RowFacts {
                    mode,
                    ..RowFacts::default()
                },
                ..TestRow::default()
            },
        );
        let ask = Ask {
            request: RequestCaps {
                n,
                ..RequestCaps::default()
            },
            ..capped(5)
        };
        assert_eq!(
            bound(&source, &query(Some(0), vec![row("painter")], ask)).max,
            expected
        );
    }
}
