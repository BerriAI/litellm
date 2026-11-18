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
}
