use std::collections::BTreeMap;

use crate::adjust::{Discounts, MarginRule, Margins};
use crate::call::CallContext;
use crate::money::UnitPrice;
use crate::price::{
    Adjustments, Band, Billing, CacheTtl, InputUnit, Meter, OffPeak, OffPeakRates, OutputUnit,
    Price, Prices, ServiceTier, TokenKind, When,
};
use crate::query::{
    AdjustmentPolicy, Basis, Candidate, Header, Modalities, PriceSource, Provider, TokenUsage,
    TokensQuery,
};
use crate::row::{LoweredRow, Provenance, RowFacts, RowPrices, RowRef, RowSource};
use crate::schedule::{Schedule, Window};

pub(crate) const BASE: Band = Band::Base;
pub(crate) const AUDIO_INPUT: Meter = Meter::Input(InputUnit::Token(TokenKind::Audio));
pub(crate) const IMAGE_OUTPUT: Meter = Meter::Output(OutputUnit::Token(TokenKind::Image));
pub(crate) const CACHE_WRITE_HOUR: Meter = Meter::CacheWrite {
    kind: TokenKind::Text,
    ttl: CacheTtl::OneHour,
};

pub(crate) fn over(tokens: u64) -> Band {
    Band::over_prompt_tokens(tokens)
}

pub(crate) fn prices(rates: &[(Meter, ServiceTier, Band, f64)]) -> Prices {
    prices_with(rates, None, Billing::default())
}

pub(crate) fn prices_with(
    rates: &[(Meter, ServiceTier, Band, f64)],
    off_peak: Option<OffPeak>,
    billing: Billing,
) -> Prices {
    Prices::new(
        rates
            .iter()
            .map(|(meter, tier, band, rate)| {
                (
                    Price::online(
                        meter.clone(),
                        When {
                            tier: *tier,
                            band: *band,
                        },
                    ),
                    UnitPrice::new(*rate).unwrap(),
                )
            })
            .collect(),
        off_peak,
        billing,
    )
}

pub(crate) fn off_peak_all_day(rates: OffPeakRates) -> Option<OffPeak> {
    OffPeak::new(
        Schedule {
            windows: vec![Window {
                hours: vec!["00:00-00:00".parse().unwrap()],
                weekdays: None,
            }],
            weekday_timezone: None,
        },
        rates,
    )
}

pub(crate) fn call() -> CallContext {
    tier_call(ServiceTier::Standard)
}

pub(crate) fn tier_call(tier: ServiceTier) -> CallContext {
    CallContext {
        tier,
        channel: crate::price::Channel::Online,
        residency: None,
        fedramp_endpoint: false,
        non_global_endpoint: false,
        speed: None,
        inference_geo: None,
        at: "2026-08-26T07:00:00Z".parse().unwrap(),
        overlay: None,
    }
}

pub(crate) fn quoted(
    prices: &Prices,
    call: &CallContext,
    prompt_tokens: u64,
    meter: &Meter,
) -> Option<(f64, Option<Meter>)> {
    prices
        .quote(call, prompt_tokens)
        .rate(meter)
        .map(|rate| (rate.unit_price.get(), rate.fallback_from))
}

pub(crate) fn basis() -> Basis {
    Basis {
        service_tier: ServiceTier::Standard,
        data_residency: None,
        vertex_location: None,
        at: call().at,
    }
}

/// One lowered row for `TestSource`. `tiers` makes it a `tiered_pricing` row over `prices`
#[derive(Clone, Debug, Default)]
pub(crate) struct TestRow {
    pub prices: Prices,
    pub tiers: Option<Prices>,
    pub adjustments: Adjustments,
    pub facts: RowFacts,
}

impl LoweredRow for TestRow {
    fn prices(&self) -> RowPrices<'_> {
        match &self.tiers {
            Some(tiers) => RowPrices::Tiered {
                tiers,
                empty_prompt: &self.prices,
            },
            None => RowPrices::Flat(&self.prices),
        }
    }

    fn adjustments(&self) -> &Adjustments {
        &self.adjustments
    }

    fn facts(&self) -> &RowFacts {
        &self.facts
    }
}

/// Rows keyed by the model name that resolves to them, plus the discount and margin config
#[derive(Clone, Debug, Default)]
pub(crate) struct TestSource {
    rows: BTreeMap<String, (RowRef, TestRow)>,
    discounts: Discounts,
    margins: Margins,
}

impl TestSource {
    pub(crate) fn with_row(model: &str, row: TestRow) -> Self {
        Self::default().and_row(model, row)
    }

    pub(crate) fn and_row(self, model: &str, row: TestRow) -> Self {
        let rows = self
            .rows
            .into_iter()
            .chain([(model.to_owned(), (row_ref(model), row))])
            .collect();
        Self { rows, ..self }
    }

    pub(crate) fn with_provenance(self, model: &str, provenance: Provenance) -> Self {
        let rows = self
            .rows
            .into_iter()
            .map(|(name, (at, row))| {
                let at = if name == model {
                    RowRef { provenance, ..at }
                } else {
                    at
                };
                (name, (at, row))
            })
            .collect();
        Self { rows, ..self }
    }

    pub(crate) fn with_discounts(self, rules: &[(&str, f64)]) -> Self {
        Self {
            discounts: Discounts::new(
                rules
                    .iter()
                    .map(|(key, fraction)| ((*key).to_owned(), *fraction)),
            ),
            ..self
        }
    }

    pub(crate) fn with_margins(self, rules: Vec<(String, MarginRule)>) -> Self {
        Self {
            margins: Margins::new(rules),
            ..self
        }
    }
}

impl RowSource for TestSource {
    type Row = TestRow;

    fn resolve(&self, candidate: &Candidate) -> Option<RowRef> {
        self.rows.get(&*candidate.model).map(|(at, _)| at.clone())
    }

    fn row(&self, at: &RowRef) -> Option<&TestRow> {
        self.rows
            .values()
            .find(|(key, _)| key.key == at.key)
            .map(|(_, row)| row)
    }

    fn discounts(&self) -> &Discounts {
        &self.discounts
    }

    fn margins(&self) -> &Margins {
        &self.margins
    }
}

pub(crate) fn row_ref(key: &str) -> RowRef {
    RowRef {
        key: key.into(),
        provenance: Provenance::Catalog,
    }
}

pub(crate) fn candidate(provider: Provider, model: &str) -> Candidate {
    Candidate {
        model: model.into(),
        provider,
    }
}

/// A row with a standard text input and output rate
pub(crate) fn text_row(input: f64, output: f64) -> TestRow {
    TestRow {
        prices: prices(&[
            (Meter::TEXT_INPUT, ServiceTier::Standard, BASE, input),
            (Meter::TEXT_OUTPUT, ServiceTier::Standard, BASE, output),
        ]),
        ..TestRow::default()
    }
}

/// Uncached text input and text output, the prompt being the input
pub(crate) fn usage(input: u64, output: u64) -> TokenUsage {
    TokenUsage {
        prompt_tokens: input,
        input: Modalities {
            text: input,
            ..Modalities::default()
        },
        output: Modalities {
            text: output,
            ..Modalities::default()
        },
        ..TokenUsage::default()
    }
}

/// A row captured in a parity fixture, lowered by hand for the handful of keys the fixtures hold
pub(crate) fn fixture_row(row: &serde_json::Value) -> TestRow {
    let fields = row.as_object().expect("a fixture row is an object");
    let rate = |key: &str| fields.get(key).and_then(serde_json::Value::as_f64);
    let listed: Vec<(Meter, ServiceTier, Band, f64)> = fields
        .iter()
        .filter_map(|(key, value)| Some((fixture_price(key)?, value.as_f64()?)))
        .map(|((meter, tier, band), rate)| (meter, tier, band, rate))
        .collect();
    let tiers = fields
        .get("tiered_pricing")
        .and_then(serde_json::Value::as_array)
        .map(|tiers| {
            let bands = std::iter::once(BASE).chain(
                tiers
                    .iter()
                    .map(|tier| over(tier["range"][1].as_f64().unwrap() as u64)),
            );
            let tiered: Vec<(Meter, ServiceTier, Band, f64)> = tiers
                .iter()
                .zip(bands)
                .flat_map(|(tier, band)| {
                    [Meter::TEXT_INPUT, Meter::TEXT_OUTPUT]
                        .into_iter()
                        .zip(["input_cost_per_token", "output_cost_per_token"])
                        .filter_map(move |(meter, key)| {
                            Some((meter, ServiceTier::Standard, band, tier[key].as_f64()?))
                        })
                })
                .collect();
            prices(&tiered)
        });
    let entry = fields
        .get("provider_specific_entry")
        .and_then(serde_json::Value::as_object);
    let uplift = |key: &str, uplift: crate::price::Uplift| {
        let factor = entry?.get(key)?.as_f64()?;
        Some((uplift, crate::money::Multiplier::new(factor)?))
    };
    TestRow {
        prices: prices(&listed),
        tiers,
        adjustments: Adjustments::new(
            [
                uplift(
                    "us",
                    crate::price::Uplift::InferenceGeo(crate::call::InferenceGeo::new("us")),
                ),
                uplift(
                    "fast",
                    crate::price::Uplift::Speed(crate::call::Speed::Fast),
                ),
                rate("regional_endpoint_uplift_multiplier")
                    .and_then(crate::money::Multiplier::new)
                    .map(|factor| (crate::price::Uplift::NonGlobalEndpoint, factor)),
            ]
            .into_iter()
            .flatten(),
        ),
        facts: RowFacts {
            mode: match fields.get("mode").and_then(serde_json::Value::as_str) {
                Some("image_generation") => crate::row::RowMode::ImageGeneration,
                Some("image_edit") => crate::row::RowMode::ImageEdit,
                _ => crate::row::RowMode::Chat,
            },
            max_input_tokens: rate("max_input_tokens").map(|tokens| tokens as u64),
            max_output_tokens: rate("max_output_tokens").map(|tokens| tokens as u64),
            ..RowFacts::default()
        },
    }
}

fn fixture_price(key: &str) -> Option<(Meter, ServiceTier, Band)> {
    let (key, tier) = key
        .strip_suffix("_priority")
        .map_or((key, ServiceTier::Standard), |key| {
            (key, ServiceTier::Priority)
        });
    let (key, band) = [
        ("_above_200k_tokens", 200_000),
        ("_above_100k_tokens", 100_000),
    ]
    .into_iter()
    .find_map(|(suffix, floor)| Some((key.strip_suffix(suffix)?, over(floor))))
    .unwrap_or((key, BASE));
    let meter = match key {
        "input_cost_per_token" => Meter::TEXT_INPUT,
        "output_cost_per_token" => Meter::TEXT_OUTPUT,
        "output_cost_per_reasoning_token" => Meter::REASONING,
        "cache_read_input_token_cost" => Meter::CACHE_READ,
        "cache_creation_input_token_cost" => Meter::CACHE_WRITE,
        "cache_creation_input_token_cost_above_1hr" => CACHE_WRITE_HOUR,
        "input_cost_per_image" => Meter::Input(InputUnit::Image),
        "output_cost_per_image" => Meter::Output(OutputUnit::Image(None)),
        "search_context_size_medium" => Meter::WebSearch(crate::price::SearchContextSize::Medium),
        "search_context_size_high" => Meter::WebSearch(crate::price::SearchContextSize::High),
        _ => return None,
    };
    Some((meter, tier, band))
}

/// A catalog-priced token query for one candidate, with discount and margin keyed on its provider
pub(crate) fn tokens_query(provider: Provider, model: &str, usage: TokenUsage) -> TokensQuery {
    TokensQuery {
        header: Header {
            adjustments: AdjustmentPolicy::Apply {
                provider: Some(provider.to_string().into()),
                model: None,
                region: None,
            },
            candidates: vec![candidate(provider, model)],
            source: PriceSource::Catalog,
            basis: basis(),
            reported: None,
        },
        usage,
        requested_model: None,
    }
}

/// A row of a per-unit parity fixture, lowered by hand for the flat keys those fixtures hold
pub(crate) fn units_row(fields: &serde_json::Value) -> TestRow {
    use crate::price::{Channel, Feature};
    let page = Meter::Feature(Feature::OcrPage);
    let annotation = Meter::Feature(Feature::AnnotationPage);
    let price = |key: &str| -> Option<(Meter, Channel)> {
        Some(match key {
            "input_cost_per_query" => (
                Meter::Input(InputUnit::Query { over_results: None }),
                Channel::Online,
            ),
            "input_cost_per_token" => (Meter::TEXT_INPUT, Channel::Online),
            "output_cost_per_token" => (Meter::TEXT_OUTPUT, Channel::Online),
            "ocr_cost_per_page" => (page.clone(), Channel::Online),
            "ocr_cost_per_page_batches" => (page.clone(), Channel::Batch),
            "annotation_cost_per_page" => (annotation.clone(), Channel::Online),
            "annotation_cost_per_page_batches" => (annotation.clone(), Channel::Batch),
            "ocr_cost_per_credit" => (Meter::Feature(Feature::OcrCredit), Channel::Online),
            "input_cost_per_character" => (Meter::Input(InputUnit::Character), Channel::Online),
            "output_cost_per_image" => (Meter::Output(OutputUnit::Image(None)), Channel::Online),
            "input_cost_per_second" => (Meter::Input(InputUnit::Second), Channel::Online),
            "output_cost_per_second" => (Meter::Output(OutputUnit::Second(None)), Channel::Online),
            "input_cost_per_audio_token" => (AUDIO_INPUT, Channel::Online),
            "output_cost_per_audio_token" => (
                Meter::Output(OutputUnit::Token(TokenKind::Audio)),
                Channel::Online,
            ),
            _ => return None,
        })
    };
    let rates = fields
        .as_object()
        .into_iter()
        .flatten()
        .filter_map(|(key, rate)| {
            let (meter, channel) = price(key)?;
            Some((
                Price {
                    meter,
                    channel,
                    when: When::default(),
                },
                UnitPrice::new(rate.as_f64()?)?,
            ))
        })
        .collect();
    TestRow {
        prices: Prices::new(rates, None, Billing::default()),
        ..TestRow::default()
    }
}

/// The case `name` of a parity fixture
pub(crate) fn fixture_case(fixture: &str, name: &str) -> serde_json::Value {
    let fixture: serde_json::Value = serde_json::from_str(fixture).unwrap();
    fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == name)
        .unwrap_or_else(|| panic!("no fixture case {name}"))
        .clone()
}

/// What a per-unit parity case expects: main's `(prompt, completion)`, or that main raised
pub(crate) fn expected_sides(case: &serde_json::Value) -> Option<(f64, f64)> {
    let expected = &case["expected"];
    Some((
        expected.get("prompt")?.as_f64()?,
        expected["completion"].as_f64()?,
    ))
}

/// A catalog-priced header for one candidate, with no discount or margin
pub(crate) fn header(provider: &str, model: &str) -> Header {
    Header {
        candidates: vec![candidate(
            provider.parse().expect("every provider name parses"),
            model,
        )],
        source: PriceSource::Catalog,
        basis: basis(),
        adjustments: AdjustmentPolicy::None,
        reported: None,
    }
}

pub(crate) fn priced(outcome: crate::receipt::PriceOutcome) -> crate::receipt::Receipt {
    match outcome {
        crate::receipt::PriceOutcome::Priced(receipt) => *receipt,
        crate::receipt::PriceOutcome::Unpriced(unpriced) => panic!("unpriced: {unpriced:?}"),
    }
}
