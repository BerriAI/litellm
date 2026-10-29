mod name_index;

use std::borrow::Cow;
use std::fmt;
use std::sync::{PoisonError, RwLock};

use name_index::NameIndex;

/// Where a `Store`'s rows live. The store rebuilds its index whenever `stamp` changes, so a source must change its
/// stamp whenever a key or a key's group changes, or call `Store::invalidate` when it cannot
pub trait Rows {
    type Row: ToOwned + ?Sized;
    type Stamp: PartialEq;

    /// The name two spellings of one key share, which `Store::find_key` matches on
    fn fold(name: &str) -> String;
    fn stamp(&self) -> Self::Stamp;
    fn contains(&self, key: &str) -> bool;
    fn row(&self, key: &str) -> Option<Cow<'_, Self::Row>>;
    /// Every key in row order with the group it is listed under
    fn scan(&self, visit: &mut dyn FnMut(&str, Option<&str>));
}

/// An in-memory store for three queries (row by exact key, actual key for a name by its folded form, keys in a group)
/// that owns the only index behind them
pub struct Store<R: Rows> {
    rows: R,
    index: RwLock<Option<Built<R::Stamp>>>,
}

struct Built<S> {
    stamp: S,
    names: NameIndex,
}

impl<R: Rows> Store<R> {
    pub fn new(rows: R) -> Self {
        Self {
            rows,
            index: RwLock::new(None),
        }
    }

    pub fn rows(&self) -> &R {
        &self.rows
    }

    /// Changes the rows in place. The index is rebuilt on the next query
    pub fn edit<T>(&mut self, edit: impl FnOnce(&mut R) -> T) -> T {
        let edited = edit(&mut self.rows);
        *self.index.get_mut().unwrap_or_else(PoisonError::into_inner) = None;
        edited
    }

    pub fn get(&self, key: &str) -> Option<Cow<'_, R::Row>> {
        self.rows.row(key)
    }

    /// The key itself when present, otherwise the actual key that folds to the same name. Never a key the rows lack
    pub fn find_key(&self, name: &str) -> Option<String> {
        if self.rows.contains(name) {
            return Some(name.to_owned());
        }
        let folded = R::fold(name);
        let hit = self.query(|names| names.key_for(&folded).map(str::to_owned))?;
        if self.rows.contains(&hit) {
            return Some(hit);
        }
        self.invalidate();
        self.query(|names| names.key_for(&folded).map(str::to_owned))
            .filter(|key| self.rows.contains(key))
    }

    /// Keys whose group is exactly `group`, in row order. Never a key the rows lack
    pub fn keys_in(&self, group: &str) -> Vec<String> {
        self.query(|names| names.keys_in(group).to_vec())
            .into_iter()
            .filter(|key| self.rows.contains(key))
            .collect()
    }

    /// Drops the index, for a source whose stamp cannot see every key change
    pub fn invalidate(&self) {
        *self.index.write().unwrap_or_else(PoisonError::into_inner) = None;
    }

    fn query<T>(&self, read: impl Fn(&NameIndex) -> T) -> T {
        let stamp = self.rows.stamp();
        if let Some(built) = self
            .index
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .filter(|built| built.stamp == stamp)
        {
            return read(&built.names);
        }
        let names = NameIndex::scan(&self.rows);
        let answer = read(&names);
        *self.index.write().unwrap_or_else(PoisonError::into_inner) = Some(Built { stamp, names });
        answer
    }
}

impl<R: Rows + Clone> Clone for Store<R> {
    fn clone(&self) -> Self {
        Self::new(self.rows.clone())
    }
}

impl<R: Rows + fmt::Debug> fmt::Debug for Store<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Store")
            .field("rows", &self.rows)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use indexmap::IndexMap;
    use rstest::{fixture, rstest};

    use super::*;

    const BUNDLED: &[(&str, Option<&str>)] = &[
        ("gpt-4o", Some("openai")),
        ("o3-pro", Some("openai")),
        ("azure/gpt-4o", Some("azure")),
        ("claude-sonnet-4-5", Some("anthropic")),
        ("gemini-2.5-pro", Some("vertex_ai-language-models")),
        (
            "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct",
            Some("fireworks_ai"),
        ),
        (
            "anyscale/meta-llama/Meta-Llama-3-8B-Instruct",
            Some("anyscale"),
        ),
        ("sample_free_row", None),
    ];

    /// `litellm.model_cost` as Python sees it. The stamp is what python-bridge can observe for free, the dict's
    /// identity and its length, so a write that keeps both hides a key change from the store
    struct ModelCost {
        dict: RefCell<IndexMap<String, Option<String>>>,
        identity: Cell<u64>,
        index_changes: Cell<u64>,
        honest_stamp: bool,
        scans: Cell<usize>,
    }

    impl ModelCost {
        fn new(rows: &[(&str, Option<&str>)], honest_stamp: bool) -> Self {
            Self {
                dict: RefCell::new(
                    rows.iter()
                        .map(|(key, provider)| ((*key).to_owned(), provider.map(str::to_owned)))
                        .collect(),
                ),
                identity: Cell::new(0),
                index_changes: Cell::new(0),
                honest_stamp,
                scans: Cell::new(0),
            }
        }

        /// `model_cost[key] = {"litellm_provider": provider, ...}`
        fn set(&self, key: &str, provider: Option<&str>) {
            let previous = self
                .dict
                .borrow_mut()
                .insert(key.to_owned(), provider.map(str::to_owned));
            if previous.as_ref().map(Option::as_deref) != Some(provider) {
                self.index_changes.set(self.index_changes.get() + 1);
            }
        }

        /// `del model_cost[key]`
        fn del(&self, key: &str) {
            if self.dict.borrow_mut().shift_remove(key).is_some() {
                self.index_changes.set(self.index_changes.get() + 1);
            }
        }

        /// `litellm.model_cost = {...}`, a new dict object
        fn rebind(&self, rows: &[(&str, Option<&str>)]) {
            *self.dict.borrow_mut() = rows
                .iter()
                .map(|(key, provider)| ((*key).to_owned(), provider.map(str::to_owned)))
                .collect();
            self.identity.set(self.identity.get() + 1);
            self.index_changes.set(self.index_changes.get() + 1);
        }
    }

    /// A row is its `litellm_provider`, which is also the group the store lists its key under
    impl Rows for &ModelCost {
        type Row = Option<String>;
        type Stamp = (u64, u64);

        fn fold(name: &str) -> String {
            name.to_lowercase()
        }

        fn stamp(&self) -> Self::Stamp {
            if self.honest_stamp {
                (self.identity.get(), self.index_changes.get())
            } else {
                (self.identity.get(), self.dict.borrow().len() as u64)
            }
        }

        fn contains(&self, key: &str) -> bool {
            self.dict.borrow().contains_key(key)
        }

        fn row(&self, key: &str) -> Option<Cow<'_, Option<String>>> {
            Some(Cow::Owned(self.dict.borrow().get(key)?.clone()))
        }

        fn scan(&self, visit: &mut dyn FnMut(&str, Option<&str>)) {
            self.scans.set(self.scans.get() + 1);
            for (key, provider) in self.dict.borrow().iter() {
                visit(key, provider.as_deref());
            }
        }
    }

    #[fixture]
    fn model_cost() -> ModelCost {
        ModelCost::new(BUNDLED, false)
    }

    /// Rows only `Store::edit` can change, so the stamp never moves
    #[derive(Debug)]
    struct Owned(IndexMap<String, Option<String>>);

    impl Rows for Owned {
        type Row = Option<String>;
        type Stamp = ();

        fn fold(name: &str) -> String {
            name.to_lowercase()
        }

        fn stamp(&self) -> Self::Stamp {}

        fn contains(&self, key: &str) -> bool {
            self.0.contains_key(key)
        }

        fn row(&self, key: &str) -> Option<Cow<'_, Option<String>>> {
            self.0.get(key).map(Cow::Borrowed)
        }

        fn scan(&self, visit: &mut dyn FnMut(&str, Option<&str>)) {
            for (key, provider) in &self.0 {
                visit(key, provider.as_deref());
            }
        }
    }

    fn owned(rows: &[(&str, Option<&str>)]) -> Store<Owned> {
        Store::new(Owned(
            rows.iter()
                .map(|(key, provider)| ((*key).to_owned(), provider.map(str::to_owned)))
                .collect(),
        ))
    }

    #[rstest]
    #[case::bundled_key_as_written("gpt-4o", Some("gpt-4o"))]
    #[case::user_typed_upper_case("GPT-4o", Some("gpt-4o"))]
    #[case::provider_prefixed("Azure/GPT-4o", Some("azure/gpt-4o"))]
    #[case::fireworks_path_in_another_case(
        "fireworks_ai/accounts/fireworks/models/Llama-v3p1-8B-Instruct",
        Some("fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct")
    )]
    #[case::mixed_case_bundled_key(
        "anyscale/meta-llama/meta-llama-3-8b-instruct",
        Some("anyscale/meta-llama/Meta-Llama-3-8B-Instruct")
    )]
    #[case::unmapped_model("my-finetune", None)]
    #[case::prefix_is_not_a_match("gpt-4", None)]
    fn find_key_maps_a_requested_model_to_its_key(
        model_cost: ModelCost,
        #[case] model: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(Store::new(&model_cost).find_key(model).as_deref(), expected);
    }

    #[rstest]
    #[case::the_bundled_spelling("gpt-4o", "gpt-4o")]
    #[case::the_registered_spelling("GPT-4o", "GPT-4o")]
    #[case::another_spelling_takes_the_later_key("Gpt-4O", "GPT-4o")]
    fn a_registered_case_variant_is_its_own_row(
        model_cost: ModelCost,
        #[case] model: &str,
        #[case] expected: &str,
    ) {
        model_cost.set("GPT-4o", Some("openai"));
        assert_eq!(
            Store::new(&model_cost).find_key(model).as_deref(),
            Some(expected)
        );
    }

    #[rstest]
    #[case::openai_in_map_order("openai", &["gpt-4o", "o3-pro"])]
    #[case::azure("azure", &["azure/gpt-4o"])]
    #[case::raw_vertex_family("vertex_ai-language-models", &["gemini-2.5-pro"])]
    #[case::provider_alias_is_python_s_job("vertex_ai", &[])]
    #[case::matched_as_written("OpenAI", &[])]
    #[case::unknown_provider("acme", &[])]
    fn keys_in_lists_the_keys_a_raw_provider_names(
        model_cost: ModelCost,
        #[case] provider: &str,
        #[case] expected: &[&str],
    ) {
        assert_eq!(Store::new(&model_cost).keys_in(provider), expected);
    }

    #[rstest]
    fn a_model_written_straight_into_the_dict_is_found_in_any_case(model_cost: ModelCost) {
        let store = Store::new(&model_cost);
        assert_eq!(store.find_key("mycorp/custom-model"), None);

        model_cost.set("MyCorp/Custom-Model", Some("openai"));

        assert_eq!(
            store.find_key("mycorp/custom-model").as_deref(),
            Some("MyCorp/Custom-Model")
        );
        assert_eq!(
            store.keys_in("openai"),
            ["gpt-4o", "o3-pro", "MyCorp/Custom-Model"]
        );
    }

    #[rstest]
    fn a_price_edit_reuses_the_index(model_cost: ModelCost) {
        let store = Store::new(&model_cost);
        assert_eq!(store.find_key("GPT-4o").as_deref(), Some("gpt-4o"));

        model_cost.set("gpt-4o", Some("openai"));
        model_cost.set("o3-pro", Some("openai"));
        model_cost.set("gpt-4o", Some("openai"));

        assert_eq!(store.find_key("O3-PRO").as_deref(), Some("o3-pro"));
        assert_eq!(store.keys_in("openai"), ["gpt-4o", "o3-pro"]);
        assert_eq!(model_cost.scans.get(), 1);
    }

    #[rstest]
    fn a_reload_that_rebinds_the_dict_is_seen_even_at_the_same_length(model_cost: ModelCost) {
        let store = Store::new(&model_cost);
        assert_eq!(store.find_key("GPT-4o").as_deref(), Some("gpt-4o"));

        let mut reloaded = BUNDLED.to_vec();
        reloaded[0] = ("gpt-4o-2024-11-20", Some("openai"));
        model_cost.rebind(&reloaded);

        assert_eq!(store.find_key("GPT-4o"), None);
        assert_eq!(
            store.find_key("GPT-4o-2024-11-20").as_deref(),
            Some("gpt-4o-2024-11-20")
        );
    }

    #[rstest]
    fn a_deleted_model_is_never_returned(model_cost: ModelCost) {
        let store = Store::new(&model_cost);
        assert_eq!(store.find_key("O3-PRO").as_deref(), Some("o3-pro"));

        model_cost.del("o3-pro");

        assert_eq!(store.find_key("O3-PRO"), None);
        assert_eq!(store.keys_in("openai"), ["gpt-4o"]);
    }

    #[rstest]
    fn a_respelled_key_at_the_same_length_is_found_after_one_rebuild(model_cost: ModelCost) {
        let store = Store::new(&model_cost);
        assert_eq!(store.find_key("o3-PRO").as_deref(), Some("o3-pro"));

        model_cost.del("o3-pro");
        model_cost.set("O3-Pro", Some("openai"));

        assert_eq!(store.find_key("o3-PRO").as_deref(), Some("O3-Pro"));
        assert_eq!(model_cost.scans.get(), 2);
    }

    #[rstest]
    fn a_swap_at_the_same_length_hides_the_new_key_until_invalidate(model_cost: ModelCost) {
        let store = Store::new(&model_cost);
        assert_eq!(store.find_key("O3-PRO").as_deref(), Some("o3-pro"));

        model_cost.del("o3-pro");
        model_cost.set("MyCorp/Custom-Model", Some("openai"));

        assert_eq!(store.find_key("mycorp/custom-model"), None);
        assert_eq!(store.keys_in("openai"), ["gpt-4o"]);

        store.invalidate();

        assert_eq!(
            store.find_key("mycorp/custom-model").as_deref(),
            Some("MyCorp/Custom-Model")
        );
        assert_eq!(store.keys_in("openai"), ["gpt-4o", "MyCorp/Custom-Model"]);
    }

    #[rstest]
    fn a_provider_rewrite_on_an_existing_key_needs_invalidate(model_cost: ModelCost) {
        let store = Store::new(&model_cost);
        assert_eq!(store.keys_in("openai"), ["gpt-4o", "o3-pro"]);

        model_cost.set("o3-pro", Some("azure"));

        assert_eq!(store.keys_in("openai"), ["gpt-4o", "o3-pro"]);

        store.invalidate();

        assert_eq!(store.keys_in("openai"), ["gpt-4o"]);
        assert_eq!(store.keys_in("azure"), ["o3-pro", "azure/gpt-4o"]);
    }

    #[rstest]
    fn an_edit_on_owned_rows_is_seen_by_the_next_query() {
        let mut store = owned(BUNDLED);
        assert_eq!(store.find_key("GPT-4o").as_deref(), Some("gpt-4o"));

        store.edit(|rows| {
            rows.0.shift_remove("gpt-4o");
            rows.0
                .insert("MyCorp/Custom-Model".to_owned(), Some("openai".to_owned()));
        });

        assert_eq!(store.find_key("GPT-4o"), None);
        assert_eq!(
            store.find_key("mycorp/custom-model").as_deref(),
            Some("MyCorp/Custom-Model")
        );
        assert_eq!(store.keys_in("openai"), ["o3-pro", "MyCorp/Custom-Model"]);
    }

    #[rstest]
    fn get_reads_the_row_by_exact_key_only() {
        let store = owned(BUNDLED);
        assert_eq!(
            store.get("azure/gpt-4o").as_deref(),
            Some(&Some("azure".to_owned()))
        );
        assert_eq!(store.get("Azure/GPT-4o"), None);
    }
}
