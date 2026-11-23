//! The pricing view of `litellm.model_cost`. Everything here exists because the table is a Python dict that any
//! Python code can replace or edit: the generation a snapshot is valid for, the swap when it moves, and lowerings
//! kept by what a row reads rather than by which object holds it
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the cost bridge stacked on this change prices through it"
    )
)]

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{CatalogView, CostLookup, LoweredRow, PricingContext, Rows, Store};

use super::live::LiveDict;

/// Lowerings kept at once. Past it the cache starts over, so rows no table holds any more are not kept forever
const LOWERED_ROWS: usize = 4096;

/// What a snapshot's table, rules and names are valid for: main's `ModelCostVersion` and the identity of the discount
/// and margin config dicts, which have no version and are only ever replaced whole
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Generation {
    pub(crate) table: usize,
    pub(crate) size: usize,
    pub(crate) writes: u64,
    pub(crate) discount: usize,
    pub(crate) margin: usize,
}

/// The published table at one generation, with the rules and names read from Python for it
pub(crate) struct CostSnapshot {
    generation: Generation,
    table: Arc<Store<LiveDict>>,
    context: PricingContext,
    lowered: Arc<LoweredRows>,
}

impl CostSnapshot {
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
    /// The snapshot for `generation`, built from `read` only when the generation moved. `read` runs with no lock held
    pub(crate) fn at<E>(
        &self,
        generation: Generation,
        read: impl FnOnce() -> Result<(Arc<Store<LiveDict>>, PricingContext), E>,
    ) -> Result<Arc<CostSnapshot>, E> {
        if let Some(current) = lock(&self.current)
            .as_ref()
            .filter(|current| current.generation == generation)
        {
            return Ok(Arc::clone(current));
        }
        let (table, context) = read()?;
        let snapshot = Arc::new(CostSnapshot {
            generation,
            table,
            context,
            lowered: Arc::clone(&self.lowered),
        });
        *lock(&self.current) = Some(Arc::clone(&snapshot));
        Ok(snapshot)
    }

    /// Drops the current snapshot, so the next call reads the rules and names again
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
