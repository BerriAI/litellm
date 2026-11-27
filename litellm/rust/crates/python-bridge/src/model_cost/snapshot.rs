//! The pricing view of `litellm.model_cost`. Everything here exists because the table is a Python dict that any
//! Python code can replace or edit: the generation a snapshot is valid for, the swap when it moves, and lowerings
//! kept by what a row reads rather than by which object holds it

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use litellm_cost::adjust::CostConfig;
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{CatalogView, CostLookup, LoweredRow, PricingContext, Rows, Store};

use super::live::LiveDict;

/// Lowerings kept at once. Past it the cache starts over, so rows no table holds any more are not kept forever
const LOWERED_ROWS: usize = 4096;

/// What a snapshot's table and rules are valid for: main's `ModelCostVersion` and the version of the host's
/// `fallback_generalizations` rules, which moves each time the host installs a rule list
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Generation {
    pub(crate) table: usize,
    pub(crate) size: usize,
    pub(crate) writes: u64,
    pub(crate) rules: u64,
}

/// The published table at one generation, with the rules, names and cost config read from Python for it
pub(crate) struct CostSnapshot {
    generation: Generation,
    table: Arc<Store<LiveDict>>,
    context: PricingContext,
    lowered: Arc<LoweredRows>,
}

impl CostSnapshot {
    pub(crate) fn config(&self) -> &CostConfig {
        &self.context.config
    }

    /// `f` over one pricing call's view. Each row is read from Python once per call, so a write while the call runs
    /// cannot give it two versions of one row
    pub(crate) fn lookup<R>(&self, f: impl FnOnce(&CostLookup<'_, CallView<'_>>) -> R) -> R {
        let view = CallView {
            table: &self.table,
            lowered: &self.lowered,
            read: Mutex::new(HashMap::new()),
        };
        f(&CostLookup::new(&view, &self.context))
    }
}

/// The snapshot for the current generation, and the lowerings every generation shares
#[derive(Default)]
pub(crate) struct Snapshots {
    current: Mutex<Option<Arc<CostSnapshot>>>,
    lowered: Arc<LoweredRows>,
}

impl Snapshots {
    /// The snapshot for `generation` and `config`, built from `read` only when either moved. Main reads the discount
    /// and margin dicts live on every call, so `config` is what the host read on this call and is compared by value,
    /// whether the dicts were replaced or edited in place. `read` runs with no lock held
    pub(crate) fn at<E>(
        &self,
        generation: Generation,
        config: CostConfig,
        read: impl FnOnce(CostConfig) -> Result<(Arc<Store<LiveDict>>, PricingContext), E>,
    ) -> Result<Arc<CostSnapshot>, E> {
        if let Some(current) = lock(&self.current)
            .as_ref()
            .filter(|current| current.generation == generation && *current.context.config == config)
        {
            return Ok(Arc::clone(current));
        }
        let (table, context) = read(config)?;
        let snapshot = Arc::new(CostSnapshot {
            generation,
            table,
            context,
            lowered: Arc::clone(&self.lowered),
        });
        *lock(&self.current) = Some(Arc::clone(&snapshot));
        Ok(snapshot)
    }

    /// Drops the current snapshot, so the next call reads the rules, names and config again
    pub(crate) fn forget(&self) {
        *lock(&self.current) = None;
    }
}

/// Lowerings by row key and billed provider. A lowering is reused only while the row reads exactly the fields it was
/// lowered from, so a row Python replaced or edited in place is lowered again. No lock is held while a row is lowered
#[derive(Default)]
struct LoweredRows(Mutex<HashMap<(String, String), Arc<LoweredRow>>>);

impl LoweredRows {
    fn lowered(&self, key: &str, provider: &str, fields: Arc<Fields>) -> Arc<LoweredRow> {
        let at = (key.to_owned(), provider.to_owned());
        if let Some(kept) = lock(&self.0).get(&at).filter(|kept| kept.fields == fields) {
            return Arc::clone(kept);
        }
        let lowered = Arc::new(LoweredRow::lower(fields, provider));
        let mut kept = lock(&self.0);
        if kept.len() >= LOWERED_ROWS {
            kept.clear();
        }
        kept.insert(at, Arc::clone(&lowered));
        lowered
    }
}

/// One pricing call's view of the published table
pub(crate) struct CallView<'a> {
    table: &'a Store<LiveDict>,
    lowered: &'a LoweredRows,
    read: Mutex<HashMap<String, Option<Arc<Fields>>>>,
}

impl CatalogView for CallView<'_> {
    fn contains(&self, key: &str) -> bool {
        self.table.rows().contains(key)
    }

    fn find_key(&self, name: &str) -> Option<String> {
        self.table.find_key(name)
    }

    fn fields(&self, key: &str) -> Option<Arc<Fields>> {
        if let Some(read) = lock(&self.read).get(key) {
            return read.clone();
        }
        let read = self.table.get(key).map(|row| Arc::new(row.fields()));
        lock(&self.read)
            .entry(key.to_owned())
            .or_insert(read)
            .clone()
    }

    fn registered_fields(&self, key: &str) -> BTreeSet<String> {
        self.table
            .get(key)
            .map(|row| row.registered_fields())
            .unwrap_or_default()
    }

    fn lowered(&self, key: &str, provider: &str) -> Option<Arc<LoweredRow>> {
        Some(self.lowered.lowered(key, provider, self.fields(key)?))
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use litellm_cost::adjust::{AdjustmentKey, Discounts, Margins};
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use rstest::rstest;

    use super::*;

    const GENERATION: Generation = Generation {
        table: 1,
        size: 1,
        writes: 0,
        rules: 0,
    };

    fn discount(fraction: f64) -> CostConfig {
        CostConfig {
            discounts: Discounts::new([("anthropic".to_owned(), fraction)]),
            margins: Margins::default(),
        }
    }

    /// Asks one `Snapshots` for each `(generation, config)` in turn over an empty table, and gives back each snapshot
    /// it handed out with how many times it read the table, rules and names
    fn snapshots_at(asked: &[(Generation, CostConfig)]) -> (Vec<Arc<CostSnapshot>>, usize) {
        Python::initialize();
        Python::attach(|py| {
            let table = Arc::new(Store::new(LiveDict::new(&PyDict::new(py))));
            let snapshots = Snapshots::default();
            let reads = Cell::new(0);
            let read = |config| {
                reads.set(reads.get() + 1);
                Ok::<_, ()>((
                    Arc::clone(&table),
                    PricingContext {
                        config: Arc::new(config),
                        ..PricingContext::default()
                    },
                ))
            };
            let handed: Vec<Arc<CostSnapshot>> = asked
                .iter()
                .map(|(generation, config)| {
                    snapshots.at(*generation, config.clone(), read).unwrap()
                })
                .collect();
            (handed, reads.get())
        })
    }

    #[rstest]
    fn one_generation_reads_its_inputs_once() {
        let (handed, reads) = snapshots_at(&[
            (GENERATION, CostConfig::default()),
            (GENERATION, CostConfig::default()),
        ]);

        assert!(Arc::ptr_eq(&handed[0], &handed[1]));
        assert_eq!(reads, 1);
    }

    #[rstest]
    #[case::table_write(Generation { writes: 1, ..GENERATION })]
    #[case::new_table(Generation { table: 2, ..GENERATION })]
    #[case::rules_replaced(Generation { rules: 7, ..GENERATION })]
    fn a_moved_generation_is_a_new_snapshot(#[case] moved: Generation) {
        let (handed, reads) = snapshots_at(&[
            (GENERATION, CostConfig::default()),
            (moved, CostConfig::default()),
        ]);

        assert!(!Arc::ptr_eq(&handed[0], &handed[1]));
        assert_eq!(handed[1].generation, moved);
        assert_eq!(reads, 2);
    }

    /// Main reads the config dicts live on every call, so a snapshot answers with the config the host read on this
    /// call however the dict changed: replaced (even by a new dict at the old one's address) or edited in place
    #[rstest]
    fn a_snapshot_prices_with_the_config_read_on_this_call() {
        let anthropic = AdjustmentKey {
            provider: Some("anthropic"),
            ..AdjustmentKey::default()
        };
        let asked: Vec<(Generation, CostConfig)> = [0.1, 0.1, 0.5, 0.1]
            .into_iter()
            .map(|fraction| (GENERATION, discount(fraction)))
            .collect();

        let (handed, reads) = snapshots_at(&asked);

        let applied: Vec<Option<f64>> = handed
            .iter()
            .map(|snapshot| snapshot.config().discounts.resolve(&anthropic))
            .collect();
        assert_eq!(applied, [Some(0.1), Some(0.1), Some(0.5), Some(0.1)]);
        assert_eq!(reads, 3);
    }
}
