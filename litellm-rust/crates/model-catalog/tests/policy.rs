use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;

use indexmap::IndexMap;
use litellm_model_catalog::wire::Fields;
use litellm_model_catalog::{
    BoundMap, Catalog, Error, Held, Installs, IntegrityLimits, Outcome, Pending, Slot,
};
use rstest::rstest;
use serde_json::{Value, json};

const FOREIGN: &str = "<foreign>";
const NOT_A_ROW: &str = "<not a row>";

fn bundled_document() -> Value {
    json!({
        "sample_spec": {"litellm_provider": "one of the providers", "input_cost_per_token": 0.0},
        "gpt-4o": {"litellm_provider": "openai", "input_cost_per_token": 1.0, "max_tokens": 16384},
        "retired": {"litellm_provider": "openai", "input_cost_per_token": 1.0},
        "claude": {"litellm_provider": "anthropic", "input_cost_per_token": 3.0, "aliases": ["claude-latest"]}
    })
}

fn parse(document: &Value) -> Catalog {
    Catalog::parse(document.to_string().as_bytes())
        .unwrap()
        .value
}

/// What python-bridge's `parse` hands to install for a fetched body
fn fetched(document: &Value, bundled: &Catalog) -> Arc<Catalog> {
    let loaded = Catalog::parse(document.to_string().as_bytes()).unwrap();
    Arc::new(
        loaded
            .value
            .with_rejected_replaced(&loaded.errors, bundled)
            .0,
    )
}

#[derive(Clone, Debug, PartialEq)]
enum Stored {
    Json(Value),
    Foreign,
}

#[derive(Clone, Debug)]
enum Entry {
    Row(usize),
    NotARow,
}

/// A Python dict of rows, where two keys can hold the same row object
#[derive(Clone, Debug, Default)]
struct Dict {
    keys: IndexMap<String, Entry>,
    rows: Vec<IndexMap<String, Stored>>,
}

struct Session<'a> {
    dict: &'a mut Dict,
    groups: HashMap<String, usize>,
}

impl Dict {
    fn row_mut(&mut self, key: &str) -> &mut IndexMap<String, Stored> {
        match self.keys[key] {
            Entry::Row(index) => &mut self.rows[index],
            Entry::NotARow => panic!("{key} is not a row"),
        }
    }

    fn view(&self, key: &str) -> Value {
        match self.keys.get(key) {
            None => Value::Null,
            Some(Entry::NotARow) => json!(NOT_A_ROW),
            Some(Entry::Row(index)) => self.rows[*index]
                .iter()
                .map(|(field, stored)| {
                    let value = match stored {
                        Stored::Json(value) => value.clone(),
                        Stored::Foreign => json!(FOREIGN),
                    };
                    (field.clone(), value)
                })
                .collect(),
        }
    }

    fn shares_row(&self, a: &str, b: &str) -> bool {
        matches!(
            (&self.keys[a], &self.keys[b]),
            (Entry::Row(a), Entry::Row(b)) if a == b
        )
    }
}

impl BoundMap for Session<'_> {
    type Error = Infallible;

    fn slot(&self, key: &str) -> Result<Slot, Infallible> {
        Ok(match self.dict.keys.get(key) {
            None => Slot::Missing,
            Some(Entry::NotARow) => Slot::NotARow,
            Some(Entry::Row(_)) => Slot::Row,
        })
    }

    fn held(&self, key: &str, field: &str) -> Result<Held, Infallible> {
        let Entry::Row(index) = self.dict.keys[key] else {
            panic!("{key} is not a row");
        };
        Ok(match self.dict.rows[index].get(field) {
            None => Held::Absent,
            Some(Stored::Json(value)) => Held::Json(value.clone()),
            Some(Stored::Foreign) => Held::Foreign,
        })
    }

    fn fields(&self, key: &str) -> Result<Vec<(String, Held)>, Infallible> {
        let Entry::Row(index) = self.dict.keys[key] else {
            panic!("{key} is not a row");
        };
        Ok(self.dict.rows[index]
            .iter()
            .map(|(field, stored)| {
                let held = match stored {
                    Stored::Json(value) => Held::Json(value.clone()),
                    Stored::Foreign => Held::Foreign,
                };
                (field.clone(), held)
            })
            .collect())
    }

    fn row_is_empty(&self, key: &str) -> Result<bool, Infallible> {
        Ok(matches!(self.dict.keys[key], Entry::Row(index) if self.dict.rows[index].is_empty()))
    }

    fn set_field(&mut self, key: &str, field: &str, value: &Value) -> Result<(), Infallible> {
        self.dict
            .row_mut(key)
            .insert(field.to_owned(), Stored::Json(value.clone()));
        Ok(())
    }

    fn del_field(&mut self, key: &str, field: &str) -> Result<(), Infallible> {
        self.dict.row_mut(key).shift_remove(field);
        Ok(())
    }

    fn insert_row(&mut self, key: &str, group: &str, fields: &Fields) -> Result<(), Infallible> {
        let index = match self.groups.get(group) {
            Some(index) => *index,
            None => {
                self.dict.rows.push(
                    fields
                        .iter()
                        .map(|(field, value)| (field.clone(), Stored::Json(value.clone())))
                        .collect(),
                );
                let index = self.dict.rows.len() - 1;
                self.groups.insert(group.to_owned(), index);
                index
            }
        };
        self.dict.keys.insert(key.to_owned(), Entry::Row(index));
        Ok(())
    }

    fn del_key(&mut self, key: &str) -> Result<(), Infallible> {
        self.dict.keys.shift_remove(key);
        Ok(())
    }

    fn retain_keys(&mut self, keep: &dyn Fn(&str) -> bool) -> Result<usize, Infallible> {
        let before = self.dict.keys.len();
        self.dict.keys.retain(|key, _| keep(key));
        Ok(before - self.dict.keys.len())
    }
}

/// What the user does to `litellm.model_cost` between two installs
#[derive(Clone, Debug)]
enum Edit {
    Set(&'static str, &'static str, Value),
    Unset(&'static str, &'static str),
    SetForeign(&'static str, &'static str),
    Pop(&'static str),
    Insert(&'static str, Value),
    ReplaceWithNonRow(&'static str),
    /// `litellm.model_cost = dict(litellm.model_cost)`, the copy landing at `dict_id`
    RebindToCopy {
        dict_id: usize,
    },
}

/// One Python process: the bundled map installed at import, then the user and later installs
struct World {
    bundled: Arc<Catalog>,
    installs: Installs,
    dict: Dict,
    dict_id: usize,
}

impl World {
    fn new() -> Self {
        let bundled = Arc::new(parse(&bundled_document()));
        let mut world = Self {
            installs: Installs::new(Arc::clone(&bundled)),
            bundled: Arc::clone(&bundled),
            dict: Dict::default(),
            dict_id: 1,
        };
        world.install_catalog(bundled);
        world
    }

    fn begin(&mut self, document: &Value) -> Pending {
        let catalog = fetched(document, &self.bundled);
        self.installs.begin(catalog, self.dict_id)
    }

    fn write(&mut self, pending: Pending) -> Outcome {
        let composed = pending.compose();
        let mut session = Session {
            dict: &mut self.dict,
            groups: HashMap::new(),
        };
        let Ok(outcome) = pending.write(&mut session, &composed);
        outcome
    }

    fn install_catalog(&mut self, catalog: Arc<Catalog>) {
        let pending = self.installs.begin(catalog, self.dict_id);
        let outcome = self.write(pending);
        self.installs.finish(outcome.record);
    }

    fn install(&mut self, document: &Value) {
        let catalog = fetched(document, &self.bundled);
        self.install_catalog(catalog);
    }

    fn edit(&mut self, edit: &Edit) {
        match edit {
            Edit::Set(key, field, value) => {
                self.dict
                    .row_mut(key)
                    .insert((*field).to_owned(), Stored::Json(value.clone()));
            }
            Edit::Unset(key, field) => {
                self.dict.row_mut(key).shift_remove(*field);
            }
            Edit::SetForeign(key, field) => {
                self.dict
                    .row_mut(key)
                    .insert((*field).to_owned(), Stored::Foreign);
            }
            Edit::Pop(key) => {
                self.dict.keys.shift_remove(*key);
            }
            Edit::Insert(key, row) => {
                self.dict.rows.push(
                    row.as_object()
                        .unwrap()
                        .iter()
                        .map(|(field, value)| (field.clone(), Stored::Json(value.clone())))
                        .collect(),
                );
                let index = self.dict.rows.len() - 1;
                self.dict.keys.insert((*key).to_owned(), Entry::Row(index));
            }
            Edit::ReplaceWithNonRow(key) => {
                self.dict.keys.insert((*key).to_owned(), Entry::NotARow);
            }
            Edit::RebindToCopy { dict_id } => {
                self.dict = self.dict.clone();
                self.dict_id = *dict_id;
            }
        }
    }

    fn row(&self, key: &str) -> Value {
        self.dict.view(key)
    }
}

fn gpt_4o(price: f64) -> Value {
    json!({"gpt-4o": {"litellm_provider": "openai", "input_cost_per_token": price}})
}

fn gpt_4o_row(price: f64, max_tokens: u32) -> Value {
    json!({"gpt-4o": {"litellm_provider": "openai", "input_cost_per_token": price, "max_tokens": max_tokens}})
}

/// The behavior users expect. A fetched map replaces bundled at row level, bundled is only the fallback for a bad
/// row or a bad document, and the user's own writes survive every install. Cases the code does not meet yet fail
mod expected {
    use super::*;

    #[rstest]
    #[case::fetched_row_is_served_as_written(
        gpt_4o(2.0),
        "gpt-4o",
        json!({"litellm_provider": "openai", "input_cost_per_token": 2.0})
    )]
    #[case::null_field_is_not_served(
        json!({"gpt-4o": {"litellm_provider": "openai", "max_tokens": null}}),
        "gpt-4o",
        json!({"litellm_provider": "openai"})
    )]
    #[case::key_the_fetched_map_leaves_out_is_not_served(gpt_4o(2.0), "retired", Value::Null)]
    #[case::new_key_is_served(
        json!({"new": {"litellm_provider": "openai", "input_cost_per_token": 5.0}}),
        "new",
        json!({"litellm_provider": "openai", "input_cost_per_token": 5.0})
    )]
    #[case::alias_is_served_without_the_aliases_field(
        json!({"claude": {"litellm_provider": "anthropic", "input_cost_per_token": 4.0, "aliases": ["claude-latest"]}}),
        "claude-latest",
        json!({"litellm_provider": "anthropic", "input_cost_per_token": 4.0})
    )]
    #[case::sample_spec_is_never_served(
        json!({"sample_spec": {"input_cost_per_token": 0.0}, "gpt-4o": {"litellm_provider": "openai"}}),
        "sample_spec",
        Value::Null
    )]
    #[case::rejected_row_is_served_from_bundled(
        gpt_4o(-1.0),
        "gpt-4o",
        json!({"litellm_provider": "openai", "input_cost_per_token": 1.0, "max_tokens": 16384})
    )]
    #[case::rejected_row_with_no_bundled_row_is_left_out(
        json!({"internal/model": {"litellm_provider": "openai", "input_cost_per_token": "free"}}),
        "internal/model",
        Value::Null
    )]
    #[case::custom_map_serves_only_its_own_rows(
        json!({"internal/model": {"litellm_provider": "openai", "input_cost_per_token": 5.0}}),
        "gpt-4o",
        Value::Null
    )]
    fn layering(#[case] fetched: Value, #[case] key: &str, #[case] expected: Value) {
        let mut world = World::new();
        world.install(&fetched);
        assert_eq!(world.row(key), expected);
    }

    #[rstest]
    fn an_alias_and_its_owner_share_one_row_object() {
        let mut world = World::new();
        world.install(
            &json!({"claude": {"litellm_provider": "anthropic", "aliases": ["claude-latest"]}}),
        );
        assert!(world.dict.shares_row("claude", "claude-latest"));
    }

    #[rstest]
    fn a_documents_own_rows_are_what_install_serves() {
        let bundled = Arc::new(parse(&bundled_document()));
        let document = fetched(&gpt_4o(2.0), &bundled);
        let mut installs = Installs::new(bundled);
        let composed = installs.begin(Arc::clone(&document), 1).compose();
        let own: IndexMap<&str, &Fields> = document.served().collect();
        let served: IndexMap<&str, &Fields> = composed
            .iter()
            .map(|(key, fields)| (key.as_str(), fields))
            .collect();
        assert_eq!(own, served);
    }

    #[rstest]
    #[case::below_the_minimum_model_count(10, 0.0, Some("below_minimum"))]
    #[case::shrunk_against_the_bundled_count(1, 0.9, Some("shrunk"))]
    #[case::within_both_limits(1, 0.3, None)]
    fn whole_map_checks(
        #[case] min_model_count: usize,
        #[case] max_shrink_ratio: f64,
        #[case] refused: Option<&str>,
    ) {
        let bundled = parse(&bundled_document());
        let kind = parse(&gpt_4o(2.0))
            .check_size(
                bundled.model_count(),
                IntegrityLimits::new(min_model_count, max_shrink_ratio).unwrap(),
            )
            .err()
            .map(|located| match located.error {
                Error::BelowMinimum { .. } => "below_minimum",
                Error::Shrunk { .. } => "shrunk",
                other => panic!("unexpected {other}"),
            });
        assert_eq!(kind, refused);
    }

    #[rstest]
    #[case::user_value_is_kept_and_untouched_fields_follow(
        Edit::Set("gpt-4o", "input_cost_per_token", json!(9.0)),
        gpt_4o_row(2.0, 32000),
        "gpt-4o",
        json!({"litellm_provider": "openai", "input_cost_per_token": 9.0, "max_tokens": 32000})
    )]
    #[case::user_deleted_field_stays_deleted(
        Edit::Unset("gpt-4o", "max_tokens"),
        gpt_4o_row(1.0, 32000),
        "gpt-4o",
        json!({"litellm_provider": "openai", "input_cost_per_token": 1.0})
    )]
    #[case::user_deleted_key_is_not_added_back(
        Edit::Pop("gpt-4o"),
        gpt_4o_row(2.0, 16384),
        "gpt-4o",
        Value::Null
    )]
    #[case::user_replaced_row_keeps_only_the_users_fields(
        Edit::Insert("gpt-4o", json!({"input_cost_per_token": 9.0})),
        gpt_4o_row(2.0, 32000),
        "gpt-4o",
        json!({"input_cost_per_token": 9.0})
    )]
    #[case::user_key_no_layer_has_is_untouched(
        Edit::Insert("mine", json!({"input_cost_per_token": 9.0})),
        gpt_4o_row(2.0, 16384),
        "mine",
        json!({"input_cost_per_token": 9.0})
    )]
    #[case::user_key_a_later_fetch_brings_keeps_the_users_fields(
        Edit::Insert("new", json!({"input_cost_per_token": 9.0})),
        json!({"new": {"litellm_provider": "openai", "input_cost_per_token": 2.0, "max_tokens": 5}}),
        "new",
        json!({"input_cost_per_token": 9.0, "litellm_provider": "openai", "max_tokens": 5})
    )]
    #[case::key_the_fetched_map_drops_keeps_only_the_users_fields(
        Edit::Set("retired", "note", json!("mine")),
        gpt_4o_row(2.0, 16384),
        "retired",
        json!({"note": "mine"})
    )]
    #[case::foreign_value_is_kept(
        Edit::SetForeign("gpt-4o", "input_cost_per_token"),
        gpt_4o_row(2.0, 16384),
        "gpt-4o",
        json!({"litellm_provider": "openai", "input_cost_per_token": FOREIGN, "max_tokens": 16384})
    )]
    #[case::non_row_value_is_kept(Edit::ReplaceWithNonRow("gpt-4o"), gpt_4o_row(2.0, 16384), "gpt-4o", json!(NOT_A_ROW))]
    #[case::user_write_equal_to_litellms_value_follows_later_fetches(
        Edit::Set("gpt-4o", "input_cost_per_token", json!(1.0)),
        gpt_4o_row(2.0, 16384),
        "gpt-4o",
        json!({"litellm_provider": "openai", "input_cost_per_token": 2.0, "max_tokens": 16384})
    )]
    #[case::rebind_is_written_in_full(
        Edit::RebindToCopy { dict_id: 2 },
        gpt_4o_row(2.0, 16384),
        "gpt-4o",
        json!({"litellm_provider": "openai", "input_cost_per_token": 2.0, "max_tokens": 16384})
    )]
    fn direct_writes(
        #[case] edit: Edit,
        #[case] fetched: Value,
        #[case] key: &str,
        #[case] expected: Value,
    ) {
        let mut world = World::new();
        world.edit(&edit);
        world.install(&fetched);
        assert_eq!(world.row(key), expected);
    }

    #[rstest]
    fn rebind_drops_direct_writes() {
        let mut world = World::new();
        world.edit(&Edit::Set("gpt-4o", "input_cost_per_token", json!(9.0)));
        world.edit(&Edit::Insert("mine", json!({"input_cost_per_token": 9.0})));
        world.edit(&Edit::RebindToCopy { dict_id: 2 });
        world.install(&gpt_4o(2.0));
        assert_eq!(world.row("gpt-4o")["input_cost_per_token"], json!(2.0));
        assert_eq!(world.row("mine"), Value::Null);
    }

    #[rstest]
    fn overlapping_installs_serve_the_last_one_begun_and_keep_following_fetches() {
        let mut world = World::new();
        let first = world.begin(&gpt_4o(2.0));
        let second = world.begin(&gpt_4o(3.0));
        let first_outcome = world.write(first);
        world.installs.finish(first_outcome.record);
        let second_outcome = world.write(second);
        world.installs.finish(second_outcome.record);
        assert_eq!(world.row("gpt-4o")["input_cost_per_token"], json!(3.0));

        world.install(&gpt_4o(4.0));
        assert_eq!(world.row("gpt-4o")["input_cost_per_token"], json!(4.0));
    }
}

/// Not decided yet. Each case pins what the code does today, so deciding one changes its expected value
mod open {
    use super::*;

    #[rstest]
    fn a_rebind_to_a_dict_at_the_old_address_keeps_direct_writes() {
        let mut world = World::new();
        world.edit(&Edit::Set("gpt-4o", "input_cost_per_token", json!(9.0)));
        world.edit(&Edit::RebindToCopy { dict_id: 1 });
        world.install(&gpt_4o(2.0));
        assert_eq!(world.row("gpt-4o")["input_cost_per_token"], json!(9.0));
    }
}
