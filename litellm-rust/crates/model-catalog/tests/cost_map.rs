use std::collections::BTreeMap;
use std::sync::LazyLock;

use litellm_model_catalog::{
    BUNDLED_REGISTRY, CostMap, ErrorKind, IntegrityLimits, ModelRow, Provenance, Row, RowShape,
    git_blob_id,
};
use proptest::prelude::*;
use rstest::rstest;
use serde_json::{Map, Value, json};

fn any_size() -> IntegrityLimits {
    IntegrityLimits::new(0, 0.0).unwrap()
}

fn fields(value: Value) -> Map<String, Value> {
    let Value::Object(fields) = value else {
        panic!("an object")
    };
    fields
}

static BUNDLED: LazyLock<CostMap> = LazyLock::new(CostMap::bundled);

fn installed(document: &Value) -> CostMap {
    let mut cost_map = BUNDLED.clone();
    cost_map
        .accept(document.to_string().as_bytes(), None, any_size())
        .unwrap();
    cost_map
}

/// What Python's dict would hold: key order, each value, and which keys share one object
fn served(rows: &[Row<'_>]) -> Vec<(String, Value)> {
    let by_key: BTreeMap<&str, &RowShape<'_>> =
        rows.iter().map(|row| (row.key, &row.shape)).collect();
    rows.iter()
        .map(|row| {
            let value = match &row.shape {
                RowShape::AliasOf(owner) => match by_key[owner] {
                    RowShape::Model(model) => {
                        json!({"alias_of": owner, "row": model.fields.as_ref()})
                    }
                    shape => panic!("an alias of {shape:?}"),
                },
                RowShape::Model(model) => Value::Object(model.fields.clone().into_owned()),
                RowShape::Section(value) => (*value).clone(),
            };
            (row.key.to_owned(), value)
        })
        .collect()
}

fn model<'a>(rows: &'a [Row<'a>], key: &str) -> &'a ModelRow<'a> {
    match &rows.iter().find(|row| row.key == key).unwrap().shape {
        RowShape::Model(model) => model,
        shape => panic!("{key} is {shape:?}"),
    }
}

#[rstest]
fn loaded_serves_python_key_order_without_fallback_rules_or_alias_lists() {
    let cost_map = installed(&json!({
        "sample_spec": {"explanation": "example"},
        "fallback_generalizations": {"rules": []},
        "owner": {"litellm_provider": "p", "aliases": ["short", "other"], "x": 1},
        "plain": {"litellm_provider": "p", "aliases": null},
    }));
    let rows = cost_map.loaded();
    assert_eq!(
        served(&rows),
        vec![
            ("sample_spec".to_owned(), json!({"explanation": "example"})),
            ("owner".to_owned(), json!({"litellm_provider": "p", "x": 1})),
            (
                "plain".to_owned(),
                json!({"litellm_provider": "p", "aliases": null})
            ),
            (
                "short".to_owned(),
                json!({"alias_of": "owner", "row": {"litellm_provider": "p", "x": 1}})
            ),
            (
                "other".to_owned(),
                json!({"alias_of": "owner", "row": {"litellm_provider": "p", "x": 1}})
            ),
        ]
    );
    assert_eq!(model(&rows, "owner").names, ["owner", "short", "other"]);
    assert!(model(&rows, "plain").names.is_empty());
}

#[rstest]
#[case::shrunk(IntegrityLimits::new(0, 0.5).unwrap(), ErrorKind::Shrunk)]
#[case::below_minimum(IntegrityLimits::new(2, 0.0).unwrap(), ErrorKind::BelowMinimum)]
fn a_rejected_document_changes_nothing(#[case] limits: IntegrityLimits, #[case] kind: ErrorKind) {
    let mut cost_map = installed(&json!({"kept": {"litellm_provider": "p"}}));
    cost_map.register("registered", &fields(json!({"x": 1})));
    let before = served(&cost_map.materialize());
    let provenance = cost_map.provenance().clone();

    let error = cost_map
        .accept(
            br#"{"replacement":{"litellm_provider":"p"}}"#,
            Some("etag".into()),
            limits,
        )
        .unwrap_err();

    assert_eq!(error.kind(), kind);
    assert_eq!(served(&cost_map.materialize()), before);
    assert_eq!(cost_map.provenance(), &provenance);
}

#[rstest]
#[case::not_json(b"{".as_slice(), ErrorKind::Json)]
#[case::empty(b"{}".as_slice(), ErrorKind::Empty)]
fn an_unparsable_document_is_rejected(#[case] body: &[u8], #[case] kind: ErrorKind) {
    let mut cost_map = CostMap::bundled();
    assert_eq!(
        cost_map.accept(body, None, any_size()).unwrap_err().kind(),
        kind
    );
    assert_eq!(
        cost_map.provenance().revision,
        git_blob_id(BUNDLED_REGISTRY)
    );
}

#[rstest]
fn provenance_is_the_accepted_bytes_and_etag() {
    let body = br#"{"m":{"litellm_provider":"p"}}"#;
    let mut cost_map = CostMap::bundled();
    cost_map
        .accept(body, Some("\"v1\"".into()), any_size())
        .unwrap();
    assert_eq!(
        cost_map.provenance(),
        &Provenance {
            revision: "eaa747f210a1527d5ea7138be0f768c47578ca68".to_owned(),
            etag: Some("\"v1\"".into()),
        }
    );
    assert_eq!(
        cost_map.accept_bundled().revision,
        git_blob_id(BUNDLED_REGISTRY)
    );
}

#[rstest]
fn registrations_survive_an_accept_above_the_new_document() {
    let mut cost_map =
        installed(&json!({"m": {"litellm_provider": "p", "price": 1, "kept": true}}));
    cost_map.register("m", &fields(json!({"price": 5})));
    cost_map.landed("m", "m");

    cost_map
        .accept(
            br#"{"m":{"litellm_provider":"p","price":2,"new":true}}"#,
            None,
            any_size(),
        )
        .unwrap();

    let rows = cost_map.materialize();
    let row = model(&rows, "m");
    assert_eq!(
        Value::Object(row.fields.clone().into_owned()),
        json!({"litellm_provider": "p", "price": 5, "new": true})
    );
    assert_eq!(
        row.registered_fields.iter().copied().collect::<Vec<_>>(),
        ["price"]
    );
}

#[rstest]
fn two_registrations_for_one_key_merge_in_call_order() {
    let mut cost_map = installed(&json!({"m": {"litellm_provider": "p"}}));
    cost_map.register("m", &fields(json!({"a": 1, "b": 1})));
    cost_map.register("m", &fields(json!({"b": 2, "c": null})));
    cost_map.landed("m", "m");
    assert_eq!(
        Value::Object(cost_map.registered_row("m").unwrap().into_owned()),
        json!({"litellm_provider": "p", "a": 1, "b": 2})
    );
}

#[rstest]
fn a_registration_for_a_key_the_document_later_adds_still_wins() {
    let mut cost_map = installed(&json!({"m": {"litellm_provider": "p"}}));
    cost_map.register(
        "custom",
        &fields(json!({"litellm_provider": "mine", "price": 9})),
    );
    cost_map.landed("custom", "custom");
    let before = served(&cost_map.materialize());
    assert_eq!(
        before.last(),
        Some(&(
            "custom".to_owned(),
            json!({"litellm_provider": "mine", "price": 9})
        ))
    );

    cost_map
        .accept(
            br#"{"custom":{"litellm_provider":"p","price":1,"other":2}}"#,
            None,
            any_size(),
        )
        .unwrap();

    assert_eq!(
        served(&cost_map.materialize()),
        vec![(
            "custom".to_owned(),
            json!({"litellm_provider": "mine", "price": 9, "other": 2})
        )]
    );
}

#[rstest]
fn a_registration_landed_on_an_alias_applies_to_the_whole_group() {
    let mut cost_map =
        installed(&json!({"owner": {"litellm_provider": "p", "aliases": ["short"], "price": 1}}));
    cost_map.register("p/short", &fields(json!({"price": 3})));
    cost_map.landed("short", "p/short");
    let rows = cost_map.materialize();
    assert_eq!(
        served(&rows),
        vec![
            (
                "owner".to_owned(),
                json!({"litellm_provider": "p", "price": 3})
            ),
            (
                "short".to_owned(),
                json!({"alias_of": "owner", "row": {"litellm_provider": "p", "price": 3}})
            ),
        ]
    );
    assert_eq!(
        model(&rows, "owner")
            .registered_fields
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        ["price"]
    );
    assert!(
        cost_map
            .materialize()
            .iter()
            .all(|row| row.key != "p/short")
    );
}

#[rstest]
fn registered_view_reads_the_document_under_registrations_without_rows_written_elsewhere() {
    let mut cost_map =
        installed(&json!({"a": {"litellm_provider": "p", "x": 1}, "b": {"litellm_provider": "p"}}));
    cost_map.register("openai/a", &fields(json!({"x": 2})));
    cost_map.landed("a", "openai/a");
    assert_eq!(
        Value::Object(cost_map.registered_row("a").unwrap().into_owned()),
        json!({"litellm_provider": "p", "x": 2})
    );
    assert_eq!(
        Value::Object(cost_map.registered_row("openai/a").unwrap().into_owned()),
        json!({"x": 2})
    );
    assert_eq!(cost_map.registered_row("missing"), None);
    assert_eq!(cost_map.registered_keys(), ["a", "b", "openai/a"]);
}

fn document_and_registrations() -> impl Strategy<Value = (BTreeMap<String, i64>, Vec<(String, i64)>)>
{
    let key = prop::sample::select(vec!["a", "b", "c", "d", "e"]).prop_map(str::to_owned);
    (
        prop::collection::btree_map(key.clone(), any::<i64>(), 1..5),
        prop::collection::vec((key, any::<i64>()), 0..8),
    )
}

proptest! {
    /// Keys are the union of document and registrations, and each row is the latest registration over the document
    #[test]
    fn materialize_serves_the_overlay_over_the_base((document, registrations) in document_and_registrations()) {
        let body: Map<String, Value> = document
            .iter()
            .map(|(key, price)| (key.clone(), json!({"litellm_provider": "p", "price": price})))
            .collect();
        let mut cost_map = installed(&Value::Object(body));
        for (key, price) in &registrations {
            cost_map.register(key, &fields(json!({"price": price})));
            cost_map.landed(key, key);
        }

        let latest: BTreeMap<&str, i64> = registrations.iter().map(|(key, price)| (key.as_str(), *price)).collect();
        let rows = cost_map.materialize();
        let served: BTreeMap<&str, Option<i64>> = rows
            .iter()
            .map(|row| match &row.shape {
                RowShape::Model(model) => (row.key, model.fields.get("price").and_then(Value::as_i64)),
                shape => panic!("unexpected {shape:?}"),
            })
            .collect();
        let expected: BTreeMap<&str, Option<i64>> = document
            .iter()
            .map(|(key, price)| (key.as_str(), Some(latest.get(key.as_str()).copied().unwrap_or(*price))))
            .chain(latest.iter().filter(|(key, _)| !document.contains_key(**key)).map(|(key, price)| (*key, Some(*price))))
            .collect();
        prop_assert_eq!(rows.len(), served.len());
        prop_assert_eq!(served, expected);
    }
}
