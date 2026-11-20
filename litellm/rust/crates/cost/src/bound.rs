//! `bound`: what to hold against a budget before a call runs, a parity port of `estimate_request_max_cost`,
//! `estimate_request_input_cost` and their helpers in `proxy/spend_tracking/budget_reservation.py`. Policy `Legacy`
//! keeps today's heuristic exactly, including where it under-holds: it reads flat rates only, so bands, cache rates,
//! uplifts, discount and margin never reach the hold. A true worst case is a later, separate change

use crate::money::UnitPrice;
use crate::price::{Band, InputUnit, Meter, OutputUnit, Price, Prices, ServiceTier, When};
use crate::row::{LoweredRow, RowMode, RowRef, RowSource};

/// Output tokens held when neither the request nor the row caps them, `DEFAULT_MAX_OUTPUT_TOKENS_FALLBACK`
pub const DEFAULT_MAX_OUTPUT_TOKENS: i64 = 16_384;

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
    pub candidates: Vec<CostShape>,
}

/// Flat rates, optionally beside a deployment's `tiered_pricing` table
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
pub struct CostShape {
    pub flat: FlatSource,
    /// The deployment row whose `tiered_pricing` replaces the flat rates for a counted prompt
    #[cfg_attr(feature = "serde", serde(default))]
    pub tiers: Option<RowRef>,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize),
    serde(tag = "from", rename_all = "snake_case")
)]
pub enum FlatSource {
    /// The rates main reads through `get_model_info`, whose missing input and output rates read 0.0
    Row { row: RowRef },
    /// The group rates the router reports (`ModelGroupInfo`), which carry no reasoning or image rates
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

/// The request fields the hold reads
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize), serde(default))]
pub struct RequestCaps {
    /// The first of `max_completion_tokens`, `max_tokens`, `max_output_tokens` and `inferenceConfig.maxTokens`
    pub requested_output: Option<i64>,
    pub n: Option<i64>,
    pub best_of: Option<i64>,
    /// An embeddings, rerank or moderations route, which produces no output
    pub input_only: bool,
}

/// `max` is the hold, `input_floor` what a cancelled call has already spent. `None` when no candidate prices
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bound {
    pub max: Option<f64>,
    pub input_floor: Option<f64>,
}

/// The rates one cost shape is read at
struct Shape<'a> {
    input: Option<f64>,
    output: Option<f64>,
    reasoning: Option<f64>,
    max_input_tokens: Option<i64>,
    max_output_tokens: Option<i64>,
    image: Option<f64>,
    tiers: Option<&'a Prices>,
}

pub fn bound(source: &impl RowSource, query: &BoundQuery) -> Bound {
    let BoundPolicy::Legacy = query.policy;
    let per_model = |estimate: &dyn Fn(&Shape<'_>, Option<i64>) -> Option<f64>| {
        highest(query.models.iter().filter_map(|model| {
            highest(
                model
                    .candidates
                    .iter()
                    .filter_map(|shape| Shape::read(source, shape))
                    .filter_map(|shape| estimate(&shape, model.counted_input)),
            )
        }))
    };
    Bound {
        max: per_model(&|shape, counted| shape.max_cost(&query.request, counted)),
        input_floor: per_model(&|shape, counted| shape.input_cost(counted)),
    }
}

fn highest(estimates: impl Iterator<Item = f64>) -> Option<f64> {
    estimates.reduce(|kept, next| if next > kept { next } else { kept })
}

impl<'a> Shape<'a> {
    /// `None` where main's `_get_model_cost_infos` raises and holds nothing for the shape
    fn read<S: RowSource>(source: &'a S, shape: &CostShape) -> Option<Self> {
        let tiers = shape
            .tiers
            .as_ref()
            .and_then(|at| source.row(at))
            .and_then(|row| row.prices().tiers());
        Some(match &shape.flat {
            FlatSource::Group(rates) => Self {
                input: rates.input.map(UnitPrice::get),
                output: rates.output.map(UnitPrice::get),
                reasoning: None,
                max_input_tokens: rates.max_input_tokens,
                max_output_tokens: rates.max_output_tokens,
                image: None,
                tiers,
            },
            FlatSource::Row { row: at } => {
                let row = source.row(at)?;
                let flat = row.prices().flat();
                let facts = row.facts();
                let image_generation =
                    matches!(facts.mode, RowMode::ImageGeneration | RowMode::ImageEdit);
                Self {
                    input: Some(base(flat, Meter::TEXT_INPUT).unwrap_or(0.0)),
                    output: Some(base(flat, Meter::TEXT_OUTPUT).unwrap_or(0.0)),
                    reasoning: base(flat, Meter::REASONING),
                    max_input_tokens: facts
                        .max_input_tokens
                        .and_then(|tokens| i64::try_from(tokens).ok()),
                    max_output_tokens: facts
                        .max_output_tokens
                        .and_then(|tokens| i64::try_from(tokens).ok()),
                    image: image_generation.then(|| image_cost(flat)).flatten(),
                    tiers: tiers.or_else(|| row.prices().tiers()),
                }
            }
        })
    }

    /// `_max_cost_for_cost_info`
    fn max_cost(&self, request: &RequestCaps, counted: Option<i64>) -> Option<f64> {
        if let Some(per_image) = self.image {
            return Some(per_image * request.n.filter(|n| *n != 0).unwrap_or(1).max(1) as f64);
        }
        let input = counted.or(self.max_input_tokens)?;
        let output = if request.input_only {
            0
        } else {
            let ceiling = self
                .max_output_tokens
                .filter(|tokens| *tokens != 0)
                .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);
            request
                .requested_output
                .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS)
                .min(ceiling)
        };
        let multiplier = [request.n, request.best_of]
            .into_iter()
            .flatten()
            .fold(1, i64::max);
        let held_output = (output * multiplier) as f64;
        if let Some(tier) = self.tier(input) {
            let output_rate = tier
                .rate(&Meter::TEXT_OUTPUT)
                .max(tier.rate(&Meter::REASONING));
            return Some(input as f64 * tier.rate(&Meter::TEXT_INPUT) + held_output * output_rate);
        }
        let input_cost = match self.input {
            Some(rate) => 0.0 + input as f64 * rate,
            None if input > 0 => return None,
            None => 0.0,
        };
        let output_rate = self
            .output
            .unwrap_or(0.0)
            .max(self.reasoning.unwrap_or(0.0));
        match (self.output, self.reasoning) {
            (None, None) if output > 0 => None,
            (None, None) => Some(input_cost),
            _ => Some(input_cost + held_output * output_rate),
        }
    }

    /// `_input_cost_for_cost_info`
    fn input_cost(&self, counted: Option<i64>) -> Option<f64> {
        let input = counted.or(self.max_input_tokens)?;
        if let Some(tier) = self.tier(input) {
            return Some(input as f64 * tier.rate(&Meter::TEXT_INPUT));
        }
        Some(input as f64 * self.input?)
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

/// The tier of a `tiered_pricing` table a prompt picked, as the band it starts at
struct Tier<'a> {
    tiers: &'a Prices,
    band: Band,
}

impl Tier<'_> {
    /// `tier_rate`: the tier's own rate, 0.0 when it has none
    fn rate(&self, meter: &Meter) -> f64 {
        self.tiers
            .get(&Price::online(
                meter.clone(),
                When {
                    tier: ServiceTier::Standard,
                    band: self.band,
                },
            ))
            .map_or(0.0, UnitPrice::get)
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
    //! a router reports

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

    /// What the adapter reads off the request body and route
    fn caps(body: &Value, route: &str) -> RequestCaps {
        let requested = [
            &body["max_completion_tokens"],
            &body["max_tokens"],
            &body["max_output_tokens"],
            &body["inferenceConfig"]["maxTokens"],
        ];
        RequestCaps {
            requested_output: requested.into_iter().find_map(Value::as_i64),
            n: body["n"].as_i64(),
            best_of: body["best_of"].as_i64(),
            input_only: ["embeddings", "rerank", "moderations"]
                .iter()
                .any(|part| route.contains(part)),
        }
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
                },
            ),
            _ => (
                TestSource::with_row("row", fixture_row(row)),
                CostShape {
                    flat: FlatSource::Row {
                        row: row_ref("row"),
                    },
                    tiers: None,
                },
            ),
        };
        bound(
            &source,
            &BoundQuery {
                models: vec![BoundModel {
                    counted_input: case["counted"].as_i64(),
                    candidates: vec![shape],
                }],
                request: caps(&case["body"], case["route"].as_str().unwrap()),
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

    fn query(counted: Option<i64>, shapes: Vec<CostShape>, request: RequestCaps) -> BoundQuery {
        BoundQuery {
            models: vec![BoundModel {
                counted_input: counted,
                candidates: shapes,
            }],
            request,
            policy: BoundPolicy::Legacy,
        }
    }

    fn row(model: &str) -> CostShape {
        CostShape {
            flat: FlatSource::Row {
                row: row_ref(model),
            },
            tiers: None,
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
        }
    }

    fn capped(output: i64) -> RequestCaps {
        RequestCaps {
            requested_output: Some(output),
            ..RequestCaps::default()
        }
    }

    #[rstest]
    #[case::counted_input_and_requested_output(Some(10), capped(5), Some(10.0 + 5.0 * 4.0))]
    #[case::uncounted_reads_max_input(None, capped(5), Some(1_000.0 + 20.0))]
    #[case::output_capped_by_the_row(Some(10), capped(500), Some(10.0 + 100.0 * 4.0))]
    #[case::default_output_capped_by_the_row(Some(10), RequestCaps::default(), Some(10.0 + 400.0))]
    #[case::n_multiplies_output(
        Some(10),
        RequestCaps { requested_output: Some(5), n: Some(3), best_of: None, input_only: false },
        Some(10.0 + 60.0)
    )]
    #[case::best_of_multiplies_output_when_higher(
        Some(10),
        RequestCaps { requested_output: Some(5), n: Some(2), best_of: Some(3), input_only: false },
        Some(10.0 + 60.0)
    )]
    #[case::input_only_route(Some(10), RequestCaps { input_only: true, ..capped(5) }, Some(10.0))]
    fn the_hold_is_input_plus_capped_output(
        #[case] counted: Option<i64>,
        #[case] request: RequestCaps,
        #[case] expected: Option<f64>,
    ) {
        let source = TestSource::default();
        assert_eq!(
            bound(
                &source,
                &query(counted, vec![group(Some(1.0), Some(4.0))], request)
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
    fn the_hold_is_the_highest_shape_and_model() {
        let source =
            TestSource::with_row("cheap", text_row(1.0, 1.0)).and_row("dear", text_row(2.0, 8.0));
        let bound = bound(
            &source,
            &BoundQuery {
                models: vec![
                    BoundModel {
                        counted_input: Some(10),
                        candidates: vec![row("cheap"), row("missing")],
                    },
                    BoundModel {
                        counted_input: Some(10),
                        candidates: vec![row("dear")],
                    },
                ],
                request: capped(5),
                policy: BoundPolicy::Legacy,
            },
        );
        assert_eq!(bound.max, Some(20.0 + 40.0));
        assert_eq!(bound.input_floor, Some(20.0));
    }

    #[rstest]
    fn reasoning_raises_the_output_rate_and_bands_are_ignored() {
        let source = TestSource::with_row(
            "reasoner",
            TestRow {
                prices: prices(&[
                    (Meter::TEXT_INPUT, Standard, BASE, 1.0),
                    (Meter::TEXT_INPUT, Standard, over(5), 100.0),
                    (Meter::TEXT_OUTPUT, Standard, BASE, 2.0),
                    (Meter::REASONING, Standard, BASE, 3.0),
                ]),
                ..TestRow::default()
            },
        );
        assert_eq!(
            bound(&source, &query(Some(10), vec![row("reasoner")], capped(5))).max,
            Some(10.0 + 15.0)
        );
    }

    #[rstest]
    #[case::counted_prompt_reads_its_tier(Some(300), Some(300.0 * 2.0 + 5.0 * 9.0), Some(600.0))]
    #[case::empty_prompt_reads_the_flat_rates(Some(0), Some(5.0 * 4.0), Some(0.0))]
    fn a_tiered_row_holds_at_the_tier_the_prompt_picks(
        #[case] counted: Option<i64>,
        #[case] max: Option<f64>,
        #[case] floor: Option<f64>,
    ) {
        let tiered = TestRow {
            prices: prices(&[
                (Meter::TEXT_INPUT, Standard, BASE, 1.0),
                (Meter::TEXT_OUTPUT, Standard, BASE, 4.0),
            ]),
            tiers: Some(prices(&[
                (Meter::TEXT_INPUT, Standard, BASE, 1.0),
                (Meter::TEXT_OUTPUT, Standard, BASE, 4.0),
                (Meter::TEXT_INPUT, Standard, over(256), 2.0),
                (Meter::TEXT_OUTPUT, Standard, over(256), 8.0),
                (Meter::REASONING, Standard, over(256), 9.0),
            ])),
            ..TestRow::default()
        };
        let source = TestSource::with_row("tiered", tiered);
        let bound = bound(&source, &query(counted, vec![row("tiered")], capped(5)));
        assert_eq!((bound.max, bound.input_floor), (max, floor));
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
                prices: prices(&[(Meter::Output(OutputUnit::Image(None)), Standard, BASE, 0.04)]),
                facts: RowFacts {
                    mode,
                    ..RowFacts::default()
                },
                ..TestRow::default()
            },
        );
        let request = RequestCaps { n, ..capped(5) };
        assert_eq!(
            bound(&source, &query(Some(0), vec![row("painter")], request)).max,
            expected
        );
    }
}
