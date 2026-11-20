//! What resolving and pricing a call reads: a host's cost table through `CatalogView`, and the `PricingContext` the
//! host keeps beside it. Neither knows how a host stores, versions or caches its table, so the Rust host's `Catalog`
//! and Python's live `litellm.model_cost` go through the same `CostLookup`

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use litellm_cost::price::Adjustments;
use serde_json::Value;

use crate::fallback::CapabilityRules;
use crate::lower::{BilledPrices, billed_prices};
use crate::wire::{EntryWire, Fields};

/// The fields that price a call without being a `*cost_per*token*` field, from main's `_NON_TOKEN_RATE_FIELDS`
const NON_TOKEN_RATE_FIELDS: [&str; 6] = [
    "cost_per_second",
    "input_cost_per_second",
    "output_cost_per_second",
    "input_cost_per_query",
    "input_cost_per_text_message",
    "tiered_pricing",
];

/// A host's cost table: exact keys as `key in litellm.model_cost`, names folded as `get_model_cost_key`, and the
/// fields of the row under a key
pub trait CatalogView {
    fn contains(&self, key: &str) -> bool;
    fn find_key(&self, name: &str) -> Option<String>;
    fn fields(&self, key: &str) -> Option<Arc<Fields>>;
    /// The fields `register_model` wrote onto the row under `key`, empty for a row the document installed
    fn registered_fields(&self, key: &str) -> BTreeSet<String>;

    /// The row under `key` lowered for billing as `provider`. A host that keeps lowerings overrides it
    fn lowered(&self, key: &str, provider: &str) -> Option<Arc<LoweredRow>> {
        Some(Arc::new(LoweredRow::lower(self.fields(key)?, provider)))
    }
}

/// Names only the host knows: every provider name, and the model names `get_model_info` renames before a lookup
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KnownNames {
    pub providers: HashSet<String>,
    pub renamed_models: HashSet<String>,
}

/// What resolving a call reads besides the table: the host's capability rules and the names only it knows
#[derive(Clone, Debug, Default)]
pub struct PricingContext {
    pub rules: CapabilityRules,
    pub names: KnownNames,
}

/// Where a row came from. A `Registered` row is a deployment's row as the router registered it under its id
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowProvenance {
    Catalog,
    Registered,
}

/// A row to price: its key in the table, the provider it is billed as, and where it came from
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RowRef {
    pub key: String,
    pub provider: String,
    pub provenance: RowProvenance,
}

/// One row read and lowered the way main bills it, with `litellm_provider` replaced by the billed provider
#[derive(Debug)]
pub struct LoweredRow {
    pub fields: Arc<Fields>,
    pub prices: BilledPrices,
    pub adjustments: Adjustments,
}

/// A row ready for pricing, with the key and provenance it was found by
#[derive(Clone, Debug)]
pub struct CatalogRow {
    pub row_ref: RowRef,
    pub lowered: Arc<LoweredRow>,
}

/// Port of main's `_cost_map_entry_prices_anything`
pub fn prices_anything(fields: &Fields) -> bool {
    fields
        .iter()
        .any(|(field, value)| !value.is_null() && is_rate_field(field))
}

fn is_rate_field(field: &str) -> bool {
    NON_TOKEN_RATE_FIELDS.contains(&field) || field.contains("cost_per") && field.contains("token")
}

impl LoweredRow {
    pub fn lower(fields: Arc<Fields>, provider: &str) -> Self {
        let billed: Fields = fields
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .chain([("litellm_provider".to_owned(), Value::from(provider))])
            .collect();
        let wire = EntryWire::read(&billed);
        Self {
            prices: billed_prices(&wire),
            adjustments: Adjustments::from(&wire),
            fields,
        }
    }
}

/// Resolving and pricing a call over one table and the context its host keeps
pub struct CostLookup<'a, V: ?Sized> {
    pub(crate) table: &'a V,
    pub(crate) context: &'a PricingContext,
}

impl<'a, V: CatalogView + ?Sized> CostLookup<'a, V> {
    pub fn new(table: &'a V, context: &'a PricingContext) -> Self {
        Self { table, context }
    }

    /// The row under `row_ref`, lowered for its provider. `None` when the table no longer has the key
    pub fn row(&self, row_ref: &RowRef) -> Option<CatalogRow> {
        Some(CatalogRow {
            lowered: self.table.lowered(&row_ref.key, &row_ref.provider)?,
            row_ref: row_ref.clone(),
        })
    }

    /// The row the router registered under deployment id `id`, exactly as stored, billed as `provider`
    pub fn deployment_row(&self, id: &str, provider: &str) -> Option<RowRef> {
        self.table.contains(id).then(|| RowRef {
            key: id.to_owned(),
            provider: provider.to_owned(),
            provenance: RowProvenance::Registered,
        })
    }
}

#[cfg(test)]
mod tests {
    use litellm_cost::call::InferenceGeo;
    use litellm_cost::money::Multiplier;
    use litellm_cost::price::Uplift;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::store::Store;
    use crate::test_support::{TestRow, TestRows, context, names};

    fn table(rows: &[(&str, Value)]) -> Store<TestRows> {
        Store::new(TestRows(
            rows.iter()
                .map(|(key, fields)| {
                    let Value::Object(fields) = fields.clone() else {
                        panic!("a row is an object")
                    };
                    ((*key).to_owned(), TestRow::new(fields, BTreeSet::new()))
                })
                .collect(),
        ))
    }

    #[rstest]
    fn a_row_is_lowered_with_its_geo_uplift() {
        let table = table(&[(
            "claude-opus-5-5",
            json!({"input_cost_per_token": 1e-6, "provider_specific_entry": {"us": 1.1}}),
        )]);
        let context = context(names(&[]));
        let opus = RowRef {
            key: "claude-opus-5-5".to_owned(),
            provider: "anthropic".to_owned(),
            provenance: RowProvenance::Catalog,
        };

        let row = CostLookup::new(&table, &context).row(&opus).unwrap();

        assert_eq!(
            row.lowered
                .adjustments
                .get(&Uplift::InferenceGeo(InferenceGeo::new("us")))
                .map(Multiplier::get),
            Some(1.1)
        );
        assert_eq!(row.row_ref, opus);
    }

    #[rstest]
    #[case::registered("dep-1", Some(RowProvenance::Registered))]
    #[case::not_registered("dep-2", None)]
    fn a_deployment_row_is_its_registered_id_row(
        #[case] id: &str,
        #[case] expected: Option<RowProvenance>,
    ) {
        let table = table(&[("dep-1", json!({"input_cost_per_token": 1e-6}))]);
        let context = context(names(&[]));

        let found = CostLookup::new(&table, &context).deployment_row(id, "anthropic");

        assert_eq!(found.as_ref().map(|row| row.provenance), expected);
        assert_eq!(found.map(|row| row.key), expected.map(|_| id.to_owned()));
    }

    #[rstest]
    #[case::token_rate(json!({"input_cost_per_token": 1e-6}), true)]
    #[case::unset_rate(json!({"input_cost_per_token": null}), false)]
    #[case::per_second(json!({"output_cost_per_second": 0.1}), true)]
    #[case::tiered(json!({"tiered_pricing": []}), true)]
    #[case::cache_rate_alone(json!({"cache_read_input_token_cost": 1e-7}), false)]
    #[case::no_rates(json!({"mode": "chat", "litellm_provider": "anthropic"}), false)]
    fn prices_anything_matches_main(#[case] row: Value, #[case] expected: bool) {
        let Value::Object(row) = row else {
            panic!("a row is an object")
        };
        assert_eq!(prices_anything(&row), expected);
    }
}
