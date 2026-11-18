//! Model Catalog's side of `litellm-cost`: the rows one query is priced against, read from a snapshot before the
//! query runs. `PricingRows` is the `RowSource` the pricing core asks, and a lowered row answers as `LoweredRow` with
//! the `RowFacts` read off its fields. The core never sees the table, so a query can only reach the rows its caller
//! fetched here

use std::collections::BTreeSet;
use std::sync::Arc;

use litellm_cost::adjust::{Discounts, Margins};
use litellm_cost::call::{InferenceGeo, Speed};
use litellm_cost::price::{Adjustments, Uplift};
use litellm_cost::query::Candidate;
use litellm_cost::row::{
    Declared, LoweredRow as PricedRow, Provenance, ProviderEntry, RowFacts, RowMode, RowPrices,
    RowRef as PricedRef, RowSource,
};
use litellm_python_compat::number::{float_of, int_of};
use litellm_python_compat::truthy::json_truthy;
use serde_json::Value;

use crate::lower::BilledPrices;
use crate::snapshot::{CostConfig, LiveTable, LoweredRow, RowProvenance, RowRef, Snapshot};
use crate::wire::Fields;

/// The rows a query may read, each fetched once, and the discount and margin config of the snapshot
#[derive(Clone, Debug)]
pub struct PricingRows {
    named: Vec<(Candidate, PricedRef)>,
    rows: Vec<(PricedRef, Arc<LoweredRow>)>,
    config: Arc<CostConfig>,
}

impl<T: LiveTable> Snapshot<T> {
    /// The rows `named` resolved to, under the candidate main priced each with, plus `rows` read by key alone. A key
    /// the table no longer holds is left out, so a query on it answers as if the catalog had no row
    pub fn pricing_rows(
        &self,
        named: impl IntoIterator<Item = (Candidate, RowRef)>,
        rows: impl IntoIterator<Item = RowRef>,
    ) -> PricingRows {
        let named: Vec<(Candidate, RowRef)> = named.into_iter().collect();
        let fetched = named
            .iter()
            .map(|(_, row)| row.clone())
            .chain(rows)
            .filter_map(|row_ref| {
                let row = self.row(&row_ref)?;
                Some((priced_ref(&row_ref), row.lowered))
            });
        PricingRows {
            rows: fetched.collect(),
            named: named
                .into_iter()
                .map(|(candidate, row)| (candidate, priced_ref(&row)))
                .collect(),
            config: Arc::clone(&self.config),
        }
    }

    /// The row under exactly `key`, billed as the provider it names, for a caller that already resolved its key
    pub fn row_at(&self, key: &str) -> Option<RowRef> {
        let fields = self.fields(key)?;
        Some(RowRef {
            key: key.to_owned(),
            provider: fields
                .get("litellm_provider")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            provenance: RowProvenance::Catalog,
        })
    }
}

impl PricingRows {
    /// Adds a row the caller passed for this call alone, under `key`
    pub fn with_per_call(self, key: &str, fields: Fields, provider: &str) -> (Self, PricedRef) {
        let at = PricedRef {
            key: key.into(),
            provenance: Provenance::PerCall,
        };
        let row = Arc::new(LoweredRow::lower(Arc::new(fields), provider));
        let rows = self.rows.into_iter().chain([(at.clone(), row)]).collect();
        (Self { rows, ..self }, at)
    }
}

impl PricingRows {
    /// Adds a row built for this call alone under `key` and names it as `candidate`'s row, such as the fields of the
    /// capability rule `get_model_info` answers with when no name is a key
    pub fn with_named_per_call(
        self,
        candidate: Candidate,
        key: &str,
        fields: Fields,
        provider: &str,
    ) -> Self {
        let (rows, at) = self.with_per_call(key, fields, provider);
        let named = rows.named.into_iter().chain([(candidate, at)]).collect();
        Self { named, ..rows }
    }
}

impl RowSource for PricingRows {
    type Row = LoweredRow;

    fn resolve(&self, candidate: &Candidate) -> Option<PricedRef> {
        self.named
            .iter()
            .find(|(named, _)| named == candidate)
            .map(|(_, at)| at.clone())
    }

    fn row(&self, at: &PricedRef) -> Option<&LoweredRow> {
        self.rows
            .iter()
            .find(|(fetched, _)| fetched == at)
            .map(|(_, row)| &**row)
    }

    fn discounts(&self) -> &Discounts {
        &self.config.discounts
    }

    fn margins(&self) -> &Margins {
        &self.config.margins
    }
}

impl PricedRow for LoweredRow {
    fn prices(&self) -> RowPrices<'_> {
        match &self.prices {
            BilledPrices::Flat(prices) => RowPrices::Flat(prices),
            BilledPrices::Tiered {
                tiers,
                empty_prompt,
            } => RowPrices::Tiered {
                tiers,
                empty_prompt,
            },
        }
    }

    fn adjustments(&self) -> &Adjustments {
        &self.adjustments
    }

    fn facts(&self) -> &RowFacts {
        &self.facts
    }
}

/// The pricing core's name for a row the snapshot found
pub fn priced_ref(row: &RowRef) -> PricedRef {
    PricedRef {
        key: row.key.as_str().into(),
        provenance: match row.provenance {
            RowProvenance::Catalog => Provenance::Catalog,
            RowProvenance::Registered => Provenance::Registered,
        },
    }
}

/// What pricing reads off a row besides its prices, as main's `get_model_info` and the auth checks read it
pub(crate) fn row_facts(fields: &Fields) -> RowFacts {
    let count = |key: &str| {
        fields
            .get(key)
            .and_then(|value| int_of(value).ok())
            .and_then(|count| u64::try_from(count).ok())
    };
    RowFacts {
        mode: match fields.get("mode").and_then(Value::as_str) {
            None | Some("chat") => RowMode::Chat,
            Some("image_generation") => RowMode::ImageGeneration,
            Some("image_edit") => RowMode::ImageEdit,
            Some(_) => RowMode::Other,
        },
        max_input_tokens: count("max_input_tokens"),
        max_output_tokens: count("max_output_tokens"),
        provider_entry: provider_entry(fields.get("provider_specific_entry")),
        declared: declared(fields),
        prices_by_results: fields
            .get("tiered_pricing")
            .and_then(Value::as_array)
            .is_some_and(|tiers| !tiers.is_empty()),
        guardrail_units: fields
            .get("guardrail_cost_per_unit")
            .and_then(Value::as_object)
            .is_some_and(|rates| rates.values().all(|rate| float_of(rate).is_ok())),
    }
}

/// `get_provider_specific_geo_multiplier` and `get_provider_specific_speed_multiplier` read the entry with `float()`
fn provider_entry(entry: Option<&Value>) -> ProviderEntry {
    match entry.filter(|entry| json_truthy(entry)) {
        None => ProviderEntry::Readable,
        Some(Value::Object(uplifts)) => {
            let unreadable: BTreeSet<Uplift> = uplifts
                .iter()
                .filter(|(_, factor)| float_of(factor).is_err())
                .map(|(key, _)| match key.as_str() {
                    "fast" => Uplift::Speed(Speed::Fast),
                    geo => Uplift::InferenceGeo(InferenceGeo::new(geo)),
                })
                .collect();
            match unreadable.is_empty() {
                true => ProviderEntry::Readable,
                false => ProviderEntry::Unreadable(unreadable),
            }
        }
        Some(_) => ProviderEntry::NotAMapping,
    }
}

fn declared(fields: &Fields) -> Declared {
    let set = |key: &str| fields.get(key).is_some_and(|value| !value.is_null());
    let cost_per = || fields.iter().filter(|(key, _)| key.contains("cost_per"));
    Declared {
        token_rate_key: fields.contains_key("input_cost_per_token")
            || fields.contains_key("output_cost_per_token"),
        positive_cost_per_key: set("tiered_pricing")
            || cost_per().any(|(_, value)| match value {
                Value::Object(nested) => nested.values().any(is_positive_cost),
                value => is_positive_cost(value),
            }),
        cost_per_key: fields.contains_key("tiered_pricing") || cost_per().next().is_some(),
        ptu_flat_cost: set("ptu_count") && set("cost_per_ptu_per_hour"),
    }
}

/// `_is_positive_cost`: a number above zero, never a bool
fn is_positive_cost(value: &Value) -> bool {
    value.as_f64().is_some_and(|cost| cost > 0.0)
}

#[cfg(test)]
mod tests {
    use litellm_cost::adjust::MarginRule;
    use litellm_cost::query::{
        AdjustmentPolicy, Basis, Header, Modalities, PriceQuery, PriceSource, Provider, TokenUsage,
        TokensQuery,
    };
    use litellm_cost::receipt::{PriceOutcome, Unpriced};
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::snapshot::SnapshotInputs;
    use crate::store::Store;
    use crate::test_support::{GENERATION, TestRows, bundled_rows, bundled_rules, names};
    use crate::{CallIdentity, Tried};

    const MODEL: &str = "claude-opus-4-5";

    fn snapshot(discount: f64) -> Snapshot<Store<TestRows>> {
        Snapshot::new(
            GENERATION,
            CostConfig {
                discounts: Discounts::new([("anthropic".to_owned(), discount)]),
                margins: Margins::new([("global".to_owned(), MarginRule::Percent(0.0))]),
            },
            SnapshotInputs {
                table: Store::new(bundled_rows()),
                rules: Arc::new(bundled_rules()),
                names: Arc::new(names(&["anthropic"])),
            },
        )
    }

    fn query(source: PriceSource) -> PriceQuery {
        PriceQuery::Tokens(Box::new(TokensQuery {
            header: Header {
                candidates: vec![Candidate {
                    model: MODEL.into(),
                    provider: Provider::Anthropic,
                }],
                source,
                basis: Basis {
                    service_tier: Default::default(),
                    data_residency: None,
                    vertex_location: None,
                    at: jiff::Timestamp::UNIX_EPOCH,
                },
                adjustments: AdjustmentPolicy::Apply {
                    provider: Some("anthropic".into()),
                    model: None,
                    region: None,
                },
                reported: None,
            },
            usage: TokenUsage {
                prompt_tokens: 1000,
                input: Modalities {
                    text: 1000,
                    ..Modalities::default()
                },
                output: Modalities {
                    text: 100,
                    ..Modalities::default()
                },
                ..TokenUsage::default()
            },
            requested_model: None,
        }))
    }

    fn total(outcome: PriceOutcome) -> f64 {
        match outcome {
            PriceOutcome::Priced(receipt) => receipt.total,
            PriceOutcome::Unpriced(unpriced) => panic!("unpriced: {unpriced:?}"),
        }
    }

    fn rows(snapshot: &Snapshot<Store<TestRows>>) -> PricingRows {
        let found = snapshot.candidates(&CallIdentity {
            provider: "anthropic".to_owned(),
            request_model: Some(MODEL.to_owned()),
            response_model: Some(MODEL.to_owned()),
            ..CallIdentity::default()
        });
        snapshot.pricing_rows(
            found.rows.into_iter().map(|Tried { model, row }| {
                (
                    Candidate {
                        model: model.into(),
                        provider: Provider::Anthropic,
                    },
                    row,
                )
            }),
            [],
        )
    }

    #[rstest]
    #[case::no_discount(0.0)]
    #[case::a_tenth_off(0.1)]
    fn a_candidate_row_prices_at_its_catalog_rates_after_the_snapshot_discount(
        #[case] discount: f64,
    ) {
        let snapshot = snapshot(discount);
        let row = snapshot.fields(MODEL).unwrap();
        let rate = |key: &str| row[key].as_f64().unwrap();
        let listed = 1000.0 * rate("input_cost_per_token") + 100.0 * rate("output_cost_per_token");
        let rows = rows(&snapshot);
        let candidate = &rows.named[0].0;

        let priced = total(litellm_cost::price(&rows, &query(PriceSource::Catalog)));

        assert_eq!(&*candidate.model, "anthropic/claude-opus-4-5");
        assert!(
            (priced - listed * (1.0 - discount)).abs() < 1e-12,
            "{priced} vs {listed}"
        );
    }

    #[rstest]
    fn a_per_call_row_prices_instead_of_the_catalog_row() {
        let snapshot = snapshot(0.0);
        let (rows, at) = rows(&snapshot).with_per_call(
            "requested_model_info_override",
            fields(json!({"input_cost_per_token": 1.0, "output_cost_per_token": 2.0})),
            "anthropic",
        );

        let priced = total(litellm_cost::price(
            &rows,
            &query(PriceSource::Row {
                row: at,
                breakdown_by_name: false,
            }),
        ));

        assert_eq!(priced, 1200.0);
    }

    #[rstest]
    fn a_query_reaches_only_the_rows_its_caller_fetched() {
        let snapshot = snapshot(0.0);
        let rows = snapshot.pricing_rows([], []);

        assert!(matches!(
            litellm_cost::price(&rows, &query(PriceSource::Catalog)),
            PriceOutcome::Unpriced(Unpriced::NoRow { .. })
        ));
    }

    fn fields(row: Value) -> Fields {
        let Value::Object(fields) = row else {
            panic!("a row is an object")
        };
        fields
    }

    #[rstest]
    #[case::absent(json!({}), ProviderEntry::Readable)]
    #[case::empty(json!({"provider_specific_entry": {}}), ProviderEntry::Readable)]
    #[case::numbers(json!({"provider_specific_entry": {"us": 1.1, "fast": "6"}}), ProviderEntry::Readable)]
    #[case::a_list(json!({"provider_specific_entry": ["us"]}), ProviderEntry::NotAMapping)]
    #[case::a_word(
        json!({"provider_specific_entry": {"US": "high", "fast": null, "eu": 1.2}}),
        ProviderEntry::Unreadable(BTreeSet::from([
            Uplift::InferenceGeo(InferenceGeo::new("us")),
            Uplift::Speed(Speed::Fast),
        ]))
    )]
    fn provider_entry_marks_the_values_float_rejects(
        #[case] row: Value,
        #[case] expected: ProviderEntry,
    ) {
        assert_eq!(row_facts(&fields(row)).provider_entry, expected);
    }

    #[rstest]
    #[case::sparse_registration(json!({"id": "dep"}), Declared::default())]
    #[case::explicit_zero(
        json!({"input_cost_per_token": 0.0, "output_cost_per_token": null}),
        Declared { token_rate_key: true, cost_per_key: true, ..Declared::default() }
    )]
    #[case::bool_is_not_a_price(
        json!({"input_cost_per_second": true}),
        Declared { cost_per_key: true, ..Declared::default() }
    )]
    #[case::nested_positive(
        json!({"search_context_cost_per_query": {"search_context_size_low": 0.01}}),
        Declared { positive_cost_per_key: true, cost_per_key: true, ..Declared::default() }
    )]
    #[case::tiers(
        json!({"tiered_pricing": [{"input_cost_per_token": 1e-6}]}),
        Declared { positive_cost_per_key: true, cost_per_key: true, ..Declared::default() }
    )]
    #[case::ptu(
        json!({"ptu_count": 2, "cost_per_ptu_per_hour": 1.0, "input_cost_per_token": 0}),
        Declared { token_rate_key: true, positive_cost_per_key: true, cost_per_key: true, ptu_flat_cost: true }
    )]
    fn declared_reads_the_raw_keys_the_auth_checks_read(
        #[case] row: Value,
        #[case] expected: Declared,
    ) {
        assert_eq!(row_facts(&fields(row)).declared, expected);
    }

    #[rstest]
    #[case::chat(json!({"mode": "chat", "max_input_tokens": 1000, "max_output_tokens": "64"}), RowMode::Chat, Some(1000), Some(64))]
    #[case::no_mode(json!({"max_input_tokens": -1}), RowMode::Chat, None, None)]
    #[case::image(json!({"mode": "image_generation"}), RowMode::ImageGeneration, None, None)]
    #[case::embedding(json!({"mode": "embedding", "max_input_tokens": 8191.9}), RowMode::Other, Some(8191), None)]
    fn mode_and_limits_read_like_get_model_info(
        #[case] row: Value,
        #[case] mode: RowMode,
        #[case] max_input: Option<u64>,
        #[case] max_output: Option<u64>,
    ) {
        let facts = row_facts(&fields(row));

        assert_eq!(
            (facts.mode, facts.max_input_tokens, facts.max_output_tokens),
            (mode, max_input, max_output)
        );
    }

    #[rstest]
    #[case::result_tiers(json!({"tiered_pricing": [{"max_results_range": [0, 25], "input_cost_per_query": 0.005}]}), true)]
    #[case::token_tiers(json!({"tiered_pricing": [{"range": [0, 1000], "input_cost_per_token": 1e-6}]}), true)]
    #[case::no_tiers(json!({"tiered_pricing": []}), false)]
    #[case::flat_query_rate(json!({"input_cost_per_query": 0.005}), false)]
    fn a_row_prices_by_results_whenever_it_lists_tiers(#[case] row: Value, #[case] expected: bool) {
        assert_eq!(row_facts(&fields(row)).prices_by_results, expected);
    }

    #[rstest]
    #[case::numbers(json!({"guardrail_cost_per_unit": {"contentPolicyUnits": 0.00015, "wordPolicyUnits": 0}}), true)]
    #[case::numeric_text(json!({"guardrail_cost_per_unit": {"contentPolicyUnits": "1e-4"}}), true)]
    #[case::empty(json!({"guardrail_cost_per_unit": {}}), true)]
    #[case::not_a_mapping(json!({"guardrail_cost_per_unit": "malformed"}), false)]
    #[case::a_word_rate(json!({"guardrail_cost_per_unit": {"contentPolicyUnits": "cheap"}}), false)]
    #[case::absent(json!({"input_cost_per_token": 1e-6}), false)]
    fn a_row_prices_guardrails_when_its_units_validate_as_main_validates_them(
        #[case] row: Value,
        #[case] expected: bool,
    ) {
        assert_eq!(row_facts(&fields(row)).guardrail_units, expected);
    }

    #[rstest]
    fn a_named_per_call_row_resolves_as_its_candidates_row() {
        let candidate = Candidate {
            model: "rerank-v3.5".into(),
            provider: Provider::Other("cohere".to_owned()),
        };
        let rows = snapshot(0.0).pricing_rows([], []).with_named_per_call(
            candidate.clone(),
            "get_model_info",
            fields(json!({"input_cost_per_query": 0.002})),
            "cohere",
        );
        let at = rows
            .resolve(&candidate)
            .expect("the candidate names the row");

        assert_eq!(at.provenance, Provenance::PerCall);
        assert!(rows.row(&at).is_some());
    }
}

#[cfg(test)]
mod batch_parity {
    //! Parity with main's `batch_cost_calculator` over rows Model Catalog lowered. `batch_parity.json` was
    //! captured once from main's Python: each case holds the deployment row main priced, the token counts
    //! `parse_prompt_tokens_details` read off the line's usage, the residency and Vertex location, and main's
    //! `(prompt, completion)`

    use litellm_cost::query::{
        AdjustmentPolicy, Basis, BatchLineQuery, BatchUsage, Candidate, DataResidency, Header,
        PriceQuery, PriceSource, WritesByTtl,
    };
    use litellm_cost::receipt::PriceOutcome;
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::snapshot::SnapshotInputs;
    use crate::store::Store;
    use crate::test_support::{GENERATION, TestRows, bundled_rows, bundled_rules, names};

    const FIXTURE: &str = include_str!("batch_parity.json");

    fn object(row: Value) -> Fields {
        let Value::Object(fields) = row else {
            panic!("a row is an object")
        };
        fields
    }

    fn sides(row: Value, provider: &str, usage: BatchUsage, basis: Basis) -> (f64, f64) {
        let snapshot: Snapshot<Store<TestRows>> = Snapshot::new(
            GENERATION,
            CostConfig::default(),
            SnapshotInputs {
                table: Store::new(bundled_rows()),
                rules: Arc::new(bundled_rules()),
                names: Arc::new(names(&[])),
            },
        );
        let (rows, at) =
            snapshot
                .pricing_rows([], [])
                .with_per_call("deployment", object(row), provider);
        let query = PriceQuery::BatchLine(Box::new(BatchLineQuery {
            header: Header {
                candidates: vec![Candidate {
                    model: "parity-model".into(),
                    provider: provider.parse().expect("every provider parses"),
                }],
                source: PriceSource::Row {
                    row: at,
                    breakdown_by_name: false,
                },
                basis,
                adjustments: AdjustmentPolicy::None,
                reported: None,
            },
            usage,
        }));
        match litellm_cost::price(&rows, &query) {
            PriceOutcome::Priced(receipt) => (receipt.prompt.get(), receipt.completion.get()),
            PriceOutcome::Unpriced(unpriced) => panic!("unpriced: {unpriced:?}"),
        }
    }

    fn basis(data_residency: Option<DataResidency>, vertex_location: Option<&str>) -> Basis {
        Basis {
            service_tier: Default::default(),
            data_residency,
            vertex_location: vertex_location.map(Into::into),
            at: jiff::Timestamp::UNIX_EPOCH,
        }
    }

    #[rstest]
    #[case::batch_rates_only("batch_rates_only")]
    #[case::half_of_online_rates("half_of_online_rates")]
    #[case::half_price_with_cache_reads_and_writes("half_price_with_cache_reads_and_writes")]
    #[case::half_price_without_a_write_rate(
        "half_price_without_a_write_rate_bills_writes_at_half_input"
    )]
    #[case::half_price_cached_audio_and_image("half_price_cached_audio_and_image")]
    #[case::batch_cache_read_and_writes("batch_cache_read_and_writes")]
    #[case::batch_cache_read_online_fallback("batch_cache_read_falls_back_to_the_online_rate")]
    #[case::batch_cached_as_text("batch_without_any_cache_read_rate_bills_cached_as_text")]
    #[case::batch_modality_input_rates("batch_modality_input_rates")]
    #[case::batch_modality_band_named_only_online("batch_modality_band_named_only_online")]
    #[case::batch_modality_band_priced("batch_modality_band_priced")]
    #[case::batch_cached_modalities("batch_cached_modalities")]
    #[case::batch_cached_modalities_online("batch_cached_modalities_read_the_online_cache_rate")]
    #[case::batch_bands_crossed("batch_bands_crossed")]
    #[case::batch_band_floor_openai("batch_band_floor_is_out_for_openai")]
    #[case::batch_band_floor_xai("batch_band_floor_is_in_for_xai")]
    #[case::batch_cache_writes_by_ttl("batch_cache_writes_by_ttl")]
    #[case::batch_cache_writes_only_by_ttl_rate("batch_cache_writes_only_by_ttl_rate")]
    #[case::image_output_batch_rate("image_output_batch_rate")]
    #[case::image_output_half_online_rate("image_output_half_online_rate")]
    #[case::image_output_text_rate("image_output_defaults_to_the_text_rate")]
    #[case::output_half_online("output_half_online_without_batch_output")]
    #[case::eu_residency_uplift("eu_residency_uplift")]
    #[case::fedramp_uplift("fedramp_uplift")]
    #[case::vertex_regional_endpoint("vertex_regional_endpoint")]
    #[case::vertex_global_endpoint("vertex_global_endpoint")]
    #[case::residency_and_vertex("residency_and_vertex")]
    #[case::string_rates("string_rates")]
    #[case::no_rates("no_rates")]
    fn a_batch_line_costs_what_main_charged(#[case] name: &str) {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("no fixture case {name}"));
        let residency = serde_json::from_value::<Option<DataResidency>>(
            case["data_residency"]
                .as_str()
                .map_or(Value::Null, |written| json!(written.to_lowercase())),
        )
        .unwrap();
        let got = sides(
            case["row"].clone(),
            case["provider"].as_str().unwrap(),
            serde_json::from_value(case["usage"].clone()).unwrap(),
            basis(residency, case["vertex_location"].as_str()),
        );
        assert_eq!(
            got,
            (
                case["expected"]["prompt"].as_f64().unwrap(),
                case["expected"]["completion"].as_f64().unwrap()
            )
        );
    }

    fn line(prompt_tokens: u64, cache_hit_tokens: u64, cache_creation_tokens: u64) -> BatchUsage {
        BatchUsage {
            prompt_tokens,
            completion_tokens: 1,
            cache_hit_tokens,
            cache_creation_tokens,
            ..BatchUsage::default()
        }
    }

    /// Ported from the `get_batch_cost_rates` tests: each batch rate reads its own band and falls back to its flat
    /// rate
    #[rstest]
    #[case::string_rates_cross_their_band(
        json!({"input_cost_per_token_batches": "1e-06", "output_cost_per_token_batches": "4e-06",
               "input_cost_per_token_above_272k_tokens_batches": "2e-06",
               "output_cost_per_token_above_272k_tokens_batches": "6e-06"}),
        300_000, (300_000.0 * 2e-6, 6e-6)
    )]
    #[case::an_unparsable_band_reads_the_flat_rate(
        json!({"input_cost_per_token_batches": 1e-6, "output_cost_per_token_batches": 4e-6,
               "input_cost_per_token_above_272k_tokens_batches": "two",
               "output_cost_per_token_above_272k_tokens_batches": "six"}),
        300_000, (300_000.0 * 1e-6, 4e-6)
    )]
    #[case::each_side_crosses_its_own_band(
        json!({"input_cost_per_token_batches": 1e-6, "input_cost_per_token_above_272k_tokens_batches": 2e-6,
               "output_cost_per_token_batches": 4e-6, "output_cost_per_token_above_200k_tokens_batches": 5e-6}),
        250_000, (250_000.0 * 1e-6, 5e-6)
    )]
    #[case::the_band_floor_is_not_crossed(
        json!({"input_cost_per_token_batches": 1e-6, "input_cost_per_token_above_272k_tokens_batches": 2e-6,
               "output_cost_per_token_batches": 4e-6, "output_cost_per_token_above_200k_tokens_batches": 5e-6}),
        200_000, (200_000.0 * 1e-6, 4e-6)
    )]
    fn batch_token_rates_read_the_crossed_band(
        #[case] row: Value,
        #[case] prompt_tokens: u64,
        #[case] expected: (f64, f64),
    ) {
        assert_eq!(
            sides(row, "openai", line(prompt_tokens, 0, 0), basis(None, None)),
            expected
        );
    }

    /// Ported from the `get_batch_cost_rates` tests: cache reads and writes bill their batch band, and a row with no
    /// batch cache write rate bills its writes as uncached input
    #[rstest]
    #[case::cache_read_band(
        json!({"input_cost_per_token_batches": 1e-6, "cache_read_input_token_cost_batches": 1e-7,
               "cache_read_input_token_cost_above_272k_tokens_batches": 2e-7}),
        line(300_000, 100_000, 0), 200_000.0 * 1e-6 + 100_000.0 * 2e-7
    )]
    #[case::cache_write_band(
        json!({"input_cost_per_token_batches": 1e-7, "input_cost_per_token_above_272k_tokens_batches": 2e-7,
               "cache_creation_input_token_cost_batches": 1.25e-7,
               "cache_creation_input_token_cost_above_272k_tokens_batches": 2.5e-7}),
        line(300_000, 0, 100_000), 200_000.0 * 2e-7 + 100_000.0 * 2.5e-7
    )]
    #[case::the_cache_write_band_floor_is_not_crossed(
        json!({"input_cost_per_token_batches": 1e-7, "input_cost_per_token_above_272k_tokens_batches": 2e-7,
               "cache_creation_input_token_cost_batches": 1.25e-7,
               "cache_creation_input_token_cost_above_272k_tokens_batches": 2.5e-7}),
        line(272_000, 0, 100_000), 172_000.0 * 1e-7 + 100_000.0 * 1.25e-7
    )]
    #[case::cached_image_band(
        json!({"input_cost_per_token_batches": 2.5e-6, "cache_read_input_image_token_cost": 2e-6,
               "cache_read_input_image_token_cost_batches": 1e-6,
               "cache_read_input_image_token_cost_above_272k_tokens_batches": 1.5e-6}),
        BatchUsage { cached_image_tokens: 1_000, ..line(300_000, 1_000, 0) },
        299_000.0 * 2.5e-6 + 1_000.0 * 1.5e-6
    )]
    #[case::no_batch_cache_write_rate_bills_writes_as_input(
        json!({"input_cost_per_token_batches": 1e-7, "input_cost_per_token_above_272k_tokens_batches": 2e-7,
               "cache_creation_input_token_cost": 2.5e-7,
               "cache_creation_input_token_cost_above_272k_tokens": 5e-7}),
        line(300_000, 0, 100_000), 300_000.0 * 2e-7
    )]
    fn batch_cache_rates_read_the_crossed_band(
        #[case] row: Value,
        #[case] usage: BatchUsage,
        #[case] expected_prompt: f64,
    ) {
        assert_eq!(
            sides(row, "openai", usage, basis(None, None)).0,
            expected_prompt
        );
    }

    /// Ported from the `get_batch_cost_rates` 12 hour write tests: 12 hour writes bill their batch rate at the band
    /// the prompt crossed, and a row with no 12 hour batch rate bills them at the base batch write rate
    #[rstest]
    #[case::short_prompt(1_000, json!({"cache_creation_input_token_cost_above_12hr_batches": 2.5e-7,
        "cache_creation_input_token_cost_above_12hr_above_272k_tokens_batches": 5e-7}), 2.5e-7)]
    #[case::crossed_band(300_000, json!({"cache_creation_input_token_cost_above_12hr_batches": 2.5e-7,
        "cache_creation_input_token_cost_above_12hr_above_272k_tokens_batches": 5e-7}), 5e-7)]
    #[case::no_12h_batch_rate(300_000, json!({"cache_creation_input_token_cost_above_12hr": 2.5e-7,
        "cache_creation_input_token_cost_above_12hr_ultrafast": 5e-7}), 1.25e-7)]
    fn twelve_hour_writes_read_their_batch_rate(
        #[case] prompt_tokens: u64,
        #[case] twelve_hours: Value,
        #[case] expected_rate: f64,
    ) {
        let base = json!({"input_cost_per_token_batches": 1e-7, "cache_creation_input_token_cost_batches": 1.25e-7});
        let row = Value::Object(
            object(base)
                .into_iter()
                .chain(object(twelve_hours))
                .collect(),
        );
        let usage = BatchUsage {
            cache_creation_by_ttl: Some(WritesByTtl {
                twelve_hours: Some(1_000),
                ..WritesByTtl::default()
            }),
            ..line(prompt_tokens, 0, 1_000)
        };
        let uncached = (prompt_tokens - 1_000) as f64 * 1e-7;
        assert_eq!(
            sides(row, "openai", usage, basis(None, None)).0,
            uncached + (0.0 * 1.25e-7 + 1_000.0 * expected_rate)
        );
    }
}
