//! One call's view of a host's live cost table, for pricing
//!
//! A `Snapshot` reads rows from the host's table through `LiveTable`, never copying the table. It is keyed on a
//! `Generation`, main's `ModelCostVersion` plus the version of the host's generalization rules, and on the discount
//! and margin config by value, and `Snapshots` hands the same snapshot to every call until either moves. Rows are lowered lazily and cached per row object, so
//! a row no write replaced keeps its lowering across generations

use std::collections::{BTreeSet, HashMap, HashSet};
use std::hash::Hash;
use std::sync::{Arc, Mutex, PoisonError};

use litellm_cost::adjust::{Discounts, Margins};
use litellm_cost::price::Adjustments;
use litellm_cost::row::RowFacts;
use serde_json::Value;

use crate::fallback::CapabilityRules;
use crate::lower::{BilledPrices, billed_prices};
use crate::pricing::row_facts;
use crate::store::{Rows, Store};
use crate::wire::{EntryWire, Fields};

/// Row objects whose reads stay cached. Past it the cache starts over, so replaced rows are not kept forever
const CACHED_ROWS: usize = 4096;

/// The fields that price a call without being a `*cost_per*token*` field, from main's `_NON_TOKEN_RATE_FIELDS`
const NON_TOKEN_RATE_FIELDS: [&str; 6] = [
    "cost_per_second",
    "input_cost_per_second",
    "output_cost_per_second",
    "input_cost_per_query",
    "input_cost_per_text_message",
    "tiered_pricing",
];

/// One row object of a host's table. Its identity changes exactly when a write installs a new row object
pub trait TableRow: Clone {
    type Identity: Eq + Hash + Clone;

    fn identity(&self) -> Self::Identity;
    fn fields(&self) -> Fields;
    /// The fields `register_model` wrote onto this row, empty for a row the document installed
    fn registered_fields(&self) -> BTreeSet<String>;
}

/// The live table: exact keys as `key in litellm.model_cost`, names folded as `get_model_cost_key`
pub trait LiveTable {
    type Row: TableRow;

    fn contains(&self, key: &str) -> bool;
    fn find_key(&self, name: &str) -> Option<String>;
    fn row(&self, key: &str) -> Option<Self::Row>;
}

impl<R> LiveTable for Store<R>
where
    R: Rows,
    <R::Row as ToOwned>::Owned: TableRow,
{
    type Row = <R::Row as ToOwned>::Owned;

    fn contains(&self, key: &str) -> bool {
        self.rows().contains(key)
    }

    fn find_key(&self, name: &str) -> Option<String> {
        Store::find_key(self, name)
    }

    fn row(&self, key: &str) -> Option<Self::Row> {
        self.get(key).map(std::borrow::Cow::into_owned)
    }
}

impl<T: LiveTable> LiveTable for Arc<T> {
    type Row = T::Row;

    fn contains(&self, key: &str) -> bool {
        (**self).contains(key)
    }

    fn find_key(&self, name: &str) -> Option<String> {
        (**self).find_key(name)
    }

    fn row(&self, key: &str) -> Option<Self::Row> {
        (**self).row(key)
    }
}

/// What a snapshot's table and rules are valid for: main's `ModelCostVersion` and the version of the host's
/// `fallback_generalizations` rules, which moves each time the host installs a rule list
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Generation {
    pub table: usize,
    pub size: usize,
    pub writes: u64,
    pub rules: u64,
}

/// `cost_discount_config` and `cost_margin_config` as the host reads them on this call. Main reads both live dicts on
/// every call, so a snapshot is reused only while they read the same, whether they were replaced or edited in place
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CostConfig {
    pub discounts: Discounts,
    pub margins: Margins,
}

/// Names only the host knows: every provider name, and the model names `get_model_info` renames before a lookup
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KnownNames {
    pub providers: HashSet<String>,
    pub renamed_models: HashSet<String>,
}

/// What the host hands a new snapshot
pub struct SnapshotInputs<T> {
    pub table: T,
    pub rules: Arc<CapabilityRules>,
    pub names: Arc<KnownNames>,
}

/// Where a row came from. A `Registered` row is a deployment's row as the router registered it under its id
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowProvenance {
    Catalog,
    Registered,
}

/// A row to price: its key in the live table, the provider it is billed as, and where it came from
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
    pub facts: RowFacts,
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

/// A call's view of the live table
pub struct Snapshot<T: LiveTable> {
    generation: Generation,
    pub(crate) table: T,
    pub(crate) rules: Arc<CapabilityRules>,
    pub(crate) names: Arc<KnownNames>,
    pub(crate) config: Arc<CostConfig>,
    memo: Arc<Memo<T::Row>>,
}

impl<T: LiveTable> Snapshot<T> {
    /// A snapshot with a cache of its own, for a host that prices one call
    pub fn new(generation: Generation, config: CostConfig, inputs: SnapshotInputs<T>) -> Self {
        Self::with_memo(generation, config, inputs, Arc::new(Memo::default()))
    }

    fn with_memo(
        generation: Generation,
        config: CostConfig,
        inputs: SnapshotInputs<T>,
        memo: Arc<Memo<T::Row>>,
    ) -> Self {
        Self {
            generation,
            table: inputs.table,
            rules: inputs.rules,
            names: inputs.names,
            config: Arc::new(config),
            memo,
        }
    }

    pub fn generation(&self) -> Generation {
        self.generation
    }

    pub fn discounts(&self) -> &Discounts {
        &self.config.discounts
    }

    pub fn margins(&self) -> &Margins {
        &self.config.margins
    }

    /// The row under `row_ref`, lowered for its provider. `None` when the table no longer has the key
    pub fn row(&self, row_ref: &RowRef) -> Option<CatalogRow> {
        let row = self.table.row(&row_ref.key)?;
        Some(CatalogRow {
            row_ref: row_ref.clone(),
            lowered: self.memo.lowered(&row, &row_ref.provider),
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

    pub(crate) fn fields(&self, key: &str) -> Option<Arc<Fields>> {
        Some(self.memo.fields(&self.table.row(key)?))
    }

    pub(crate) fn registered_fields(&self, key: &str) -> BTreeSet<String> {
        self.table
            .row(key)
            .map(|row| row.registered_fields())
            .unwrap_or_default()
    }
}

/// The snapshot for the current generation, and the row cache every generation shares
pub struct Snapshots<T: LiveTable> {
    current: Mutex<Option<Arc<Snapshot<T>>>>,
    memo: Arc<Memo<T::Row>>,
}

impl<T: LiveTable> Default for Snapshots<T> {
    fn default() -> Self {
        Self {
            current: Mutex::new(None),
            memo: Arc::new(Memo::default()),
        }
    }
}

impl<T: LiveTable> Snapshots<T> {
    /// The snapshot for `generation` and `config`, built from `read` only when either moved. `read` runs with no lock
    /// held
    pub fn at<E>(
        &self,
        generation: Generation,
        config: CostConfig,
        read: impl FnOnce() -> Result<SnapshotInputs<T>, E>,
    ) -> Result<Arc<Snapshot<T>>, E> {
        if let Some(current) = self
            .current
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .filter(|current| current.generation == generation && *current.config == config)
        {
            return Ok(Arc::clone(current));
        }
        let snapshot = Arc::new(Snapshot::with_memo(
            generation,
            config,
            read()?,
            Arc::clone(&self.memo),
        ));
        *self.current.lock().unwrap_or_else(PoisonError::into_inner) = Some(Arc::clone(&snapshot));
        Ok(snapshot)
    }

    /// Drops every cached row read, for a host that edited a row object in place
    pub fn forget_rows(&self) {
        *self.current.lock().unwrap_or_else(PoisonError::into_inner) = None;
        self.memo.clear();
    }
}

/// Reads and lowerings keyed by row object. Each entry holds its row so the identity cannot be reused while cached.
/// No lock is held while a row is read or lowered, since a host row read may wait on the host's own lock
struct Memo<Row: TableRow> {
    entries: Mutex<HashMap<Row::Identity, Cached<Row>>>,
}

struct Cached<Row> {
    _row: Row,
    fields: Arc<Fields>,
    lowered: HashMap<String, Arc<LoweredRow>>,
}

impl<Row: TableRow> Default for Memo<Row> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
}

impl<Row: TableRow> Memo<Row> {
    fn fields(&self, row: &Row) -> Arc<Fields> {
        let identity = row.identity();
        if let Some(cached) = self.entries().get(&identity) {
            return Arc::clone(&cached.fields);
        }
        let fields = Arc::new(row.fields());
        let mut entries = self.entries();
        if entries.len() >= CACHED_ROWS {
            entries.clear();
        }
        Arc::clone(
            &entries
                .entry(identity)
                .or_insert_with(|| Cached {
                    _row: row.clone(),
                    fields,
                    lowered: HashMap::new(),
                })
                .fields,
        )
    }

    fn lowered(&self, row: &Row, provider: &str) -> Arc<LoweredRow> {
        let identity = row.identity();
        if let Some(lowered) = self
            .entries()
            .get(&identity)
            .and_then(|cached| cached.lowered.get(provider))
        {
            return Arc::clone(lowered);
        }
        let fields = self.fields(row);
        let lowered = Arc::new(LoweredRow::lower(fields, provider));
        if let Some(cached) = self.entries().get_mut(&identity) {
            cached
                .lowered
                .insert(provider.to_owned(), Arc::clone(&lowered));
        }
        lowered
    }

    fn clear(&self) {
        self.entries().clear();
    }

    fn entries(&self) -> std::sync::MutexGuard<'_, HashMap<Row::Identity, Cached<Row>>> {
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl LoweredRow {
    pub(crate) fn lower(fields: Arc<Fields>, provider: &str) -> Self {
        let billed: Fields = fields
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .chain([("litellm_provider".to_owned(), Value::from(provider))])
            .collect();
        let wire = EntryWire::read(&billed);
        Self {
            prices: billed_prices(&wire),
            adjustments: Adjustments::from(&wire),
            facts: row_facts(&fields),
            fields,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::RwLock;

    use indexmap::IndexMap;
    use litellm_cost::adjust::AdjustmentKey;
    use litellm_cost::call::InferenceGeo;
    use litellm_cost::money::Multiplier;
    use litellm_cost::price::Uplift;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::test_support::{GENERATION, bundled_rules};

    /// A row object a test can edit in place, as Python code can edit a `litellm.model_cost` row
    #[derive(Clone)]
    struct EditableRow(Arc<RwLock<Fields>>);

    impl TableRow for EditableRow {
        type Identity = usize;

        fn identity(&self) -> usize {
            Arc::as_ptr(&self.0) as usize
        }

        fn fields(&self) -> Fields {
            self.0.read().unwrap().clone()
        }

        fn registered_fields(&self) -> BTreeSet<String> {
            BTreeSet::new()
        }
    }

    #[derive(Default)]
    struct Table(RwLock<IndexMap<String, EditableRow>>);

    impl Table {
        fn write(&self, key: &str, fields: Value) -> EditableRow {
            let Value::Object(fields) = fields else {
                panic!("a row is an object")
            };
            let row = EditableRow(Arc::new(RwLock::new(fields)));
            self.0.write().unwrap().insert(key.to_owned(), row.clone());
            row
        }
    }

    impl LiveTable for Table {
        type Row = EditableRow;

        fn contains(&self, key: &str) -> bool {
            self.0.read().unwrap().contains_key(key)
        }

        fn find_key(&self, name: &str) -> Option<String> {
            self.contains(name).then(|| name.to_owned())
        }

        fn row(&self, key: &str) -> Option<EditableRow> {
            self.0.read().unwrap().get(key).cloned()
        }
    }

    fn inputs(table: &Arc<Table>) -> Result<SnapshotInputs<Arc<Table>>, ()> {
        Ok(SnapshotInputs {
            table: Arc::clone(table),
            rules: Arc::new(bundled_rules()),
            names: Arc::new(KnownNames::default()),
        })
    }

    fn opus() -> RowRef {
        RowRef {
            key: "claude-opus-5-5".to_owned(),
            provider: "anthropic".to_owned(),
            provenance: RowProvenance::Catalog,
        }
    }

    fn input_rate(snapshot: &Snapshot<Arc<Table>>) -> Option<f64> {
        snapshot.row(&opus())?.lowered.fields["input_cost_per_token"].as_f64()
    }

    #[rstest]
    fn one_generation_reads_its_inputs_once() {
        let table = Arc::new(Table::default());
        let snapshots = Snapshots::default();
        let reads = Cell::new(0);
        let read = || {
            reads.set(reads.get() + 1);
            inputs(&table)
        };

        let first = snapshots
            .at(GENERATION, CostConfig::default(), read)
            .unwrap();
        let second = snapshots
            .at(GENERATION, CostConfig::default(), read)
            .unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(reads.get(), 1);
    }

    #[rstest]
    #[case::table_write(Generation { writes: 1, ..GENERATION })]
    #[case::new_table(Generation { table: 2, ..GENERATION })]
    #[case::rules_replaced(Generation { rules: 7, ..GENERATION })]
    fn a_moved_generation_is_a_new_snapshot(#[case] moved: Generation) {
        let table = Arc::new(Table::default());
        let snapshots = Snapshots::default();

        let before = snapshots
            .at(GENERATION, CostConfig::default(), || inputs(&table))
            .unwrap();
        let after = snapshots
            .at(moved, CostConfig::default(), || inputs(&table))
            .unwrap();

        assert!(!Arc::ptr_eq(&before, &after));
        assert_eq!(after.generation(), moved);
    }

    fn discount(fraction: f64) -> CostConfig {
        CostConfig {
            discounts: Discounts::new([("anthropic".to_owned(), fraction)]),
            margins: Margins::default(),
        }
    }

    /// Main reads the config dicts live on every call, so a snapshot answers with the config the host read on this
    /// call however the dict changed: replaced (even by a new dict at the old one's address) or edited in place
    #[rstest]
    fn a_snapshot_prices_with_the_config_read_on_this_call() {
        let table = Arc::new(Table::default());
        let snapshots = Snapshots::default();
        let reads = Cell::new(0);
        let read = || {
            reads.set(reads.get() + 1);
            inputs(&table)
        };
        let anthropic = || AdjustmentKey {
            provider: Some("anthropic"),
            ..AdjustmentKey::default()
        };

        let applied: Vec<Option<f64>> = [0.1, 0.1, 0.5, 0.1]
            .into_iter()
            .map(|fraction| {
                let snapshot = snapshots.at(GENERATION, discount(fraction), read).unwrap();
                snapshot.discounts().resolve(&anthropic())
            })
            .collect();

        assert_eq!(applied, [Some(0.1), Some(0.1), Some(0.5), Some(0.1)]);
        assert_eq!(reads.get(), 3);
    }

    #[rstest]
    fn a_row_a_write_replaced_is_read_from_its_new_object() {
        let table = Arc::new(Table::default());
        let snapshots = Snapshots::default();
        table.write("claude-opus-5-5", json!({"input_cost_per_token": 1e-6}));
        let before = snapshots
            .at(GENERATION, CostConfig::default(), || inputs(&table))
            .unwrap();
        assert_eq!(input_rate(&before), Some(1e-6));

        table.write("claude-opus-5-5", json!({"input_cost_per_token": 2e-6}));
        let after = snapshots
            .at(
                Generation {
                    writes: 1,
                    ..GENERATION
                },
                CostConfig::default(),
                || inputs(&table),
            )
            .unwrap();

        assert_eq!(input_rate(&after), Some(2e-6));
    }

    #[rstest]
    fn an_in_place_edit_is_read_once_the_host_forgets_rows() {
        let table = Arc::new(Table::default());
        let snapshots = Snapshots::default();
        let row = table.write("claude-opus-5-5", json!({"input_cost_per_token": 1e-6}));
        let before = snapshots
            .at(GENERATION, CostConfig::default(), || inputs(&table))
            .unwrap();
        assert_eq!(input_rate(&before), Some(1e-6));
        row.0
            .write()
            .unwrap()
            .insert("input_cost_per_token".to_owned(), json!(3e-6));
        let moved = Generation {
            writes: 1,
            ..GENERATION
        };

        let unseen = input_rate(
            &snapshots
                .at(moved, CostConfig::default(), || inputs(&table))
                .unwrap(),
        );
        snapshots.forget_rows();
        let seen = input_rate(
            &snapshots
                .at(moved, CostConfig::default(), || inputs(&table))
                .unwrap(),
        );

        assert_eq!(unseen, Some(1e-6));
        assert_eq!(seen, Some(3e-6));
    }

    #[rstest]
    fn a_row_is_lowered_with_its_geo_uplift() {
        let table = Arc::new(Table::default());
        table.write(
            "claude-opus-5-5",
            json!({"input_cost_per_token": 1e-6, "provider_specific_entry": {"us": 1.1}}),
        );
        let snapshot = Snapshot::new(GENERATION, CostConfig::default(), inputs(&table).unwrap());

        let row = snapshot.row(&opus()).unwrap();

        assert_eq!(
            row.lowered
                .adjustments
                .get(&Uplift::InferenceGeo(InferenceGeo::new("us")))
                .map(Multiplier::get),
            Some(1.1)
        );
        assert_eq!(row.row_ref, opus());
    }

    #[rstest]
    #[case::registered("dep-1", Some(RowProvenance::Registered))]
    #[case::not_registered("dep-2", None)]
    fn a_deployment_row_is_its_registered_id_row(
        #[case] id: &str,
        #[case] expected: Option<RowProvenance>,
    ) {
        let table = Arc::new(Table::default());
        table.write("dep-1", json!({"input_cost_per_token": 1e-6}));
        let snapshot = Snapshot::new(GENERATION, CostConfig::default(), inputs(&table).unwrap());

        let found = snapshot.deployment_row(id, "anthropic");

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
