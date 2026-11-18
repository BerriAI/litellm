use std::collections::{BTreeMap, BTreeSet};

use litellm_redis_resolve::{Asked, Error, Handling, Kwarg, interpret, kwargs};
use rstest::rstest;
use serde::Deserialize;
use serde_json::json;
use strum::IntoEnumIterator;

#[derive(Deserialize)]
struct Catalog {
    meta: Meta,
    args: BTreeMap<String, Arg>,
}

#[derive(Deserialize)]
struct Meta {
    redis_py_locked: String,
}

#[derive(Deserialize)]
struct Arg {
    environment: bool,
    redis_py: BTreeMap<String, Vec<String>>,
}

// the signatures of the clients the proxy builds, where an unknown kwarg raises TypeError
const PROXY_SIGNATURES: &[&str] = &["async", "cluster", "connection"];

fn rejected() -> impl Iterator<Item = (Kwarg, Asked)> {
    Kwarg::iter().filter_map(|kwarg| match kwarg.setting().handling {
        Handling::Reject { asked, .. } => Some((kwarg, asked)),
        Handling::Read | Handling::Warn { .. } => None,
    })
}

fn catalog() -> Catalog {
    serde_json::from_str(include_str!("../redis-py-kwargs.json")).unwrap()
}

fn host_with(kwarg: &str, value: serde_json::Value) -> litellm_redis_resolve::Kwargs {
    kwargs([
        ("host".to_owned(), json!("redis")),
        (kwarg.to_owned(), value),
    ])
}

#[rstest]
fn every_catalogued_kwarg_is_a_kwarg_and_nothing_else_is() {
    let kwargs: BTreeSet<&str> = Kwarg::iter().map(Kwarg::name).collect();

    let catalog = catalog();
    let catalogued: BTreeSet<&str> = catalog.args.keys().map(String::as_str).collect();
    assert_eq!(kwargs, catalogued);
}

#[rstest]
fn settings_read_redis_environment_names_exactly_where_python_does() {
    let catalog = catalog();

    let differing: Vec<&str> = Kwarg::iter()
        .filter(|kwarg| {
            catalog
                .args
                .get(kwarg.name())
                .is_some_and(|arg| arg.environment != kwarg.setting().environment)
        })
        .map(Kwarg::name)
        .collect();
    assert!(differing.is_empty(), "{differing:?}");
}

#[rstest]
fn every_warned_setting_is_reported() {
    let warned = Kwarg::iter()
        .filter(|kwarg| matches!(kwarg.setting().handling, Handling::Warn { met: None, .. }));

    for kwarg in warned {
        let resolved = interpret(&host_with(kwarg.name(), json!("set")), None, None).unwrap();

        let reported: Vec<&str> = resolved
            .ignored
            .iter()
            .map(|ignored| ignored.setting.as_str())
            .collect();
        assert_eq!(reported, vec![kwarg.name()]);
    }
}

#[rstest]
fn every_rejected_setting_fails_when_asked_for_and_not_otherwise() {
    for (kwarg, asked) in rejected() {
        let (asking, not_asking) = match asked {
            Asked::WhenTrue => (json!(true), json!(false)),
            Asked::WhenSet => (json!(["VERIFY_CRL_CHECK_LEAF"]), json!([])),
        };

        let error = interpret(&host_with(kwarg.name(), asking), None, None).unwrap_err();
        assert!(
            matches!(&error, Error::Unenforceable { setting: source, .. } if source == kwarg.name()),
            "{error:?}"
        );
        assert!(interpret(&host_with(kwarg.name(), not_asking), None, None).is_ok());
    }
}

#[rstest]
fn python_already_refuses_every_rejected_setting() {
    let catalog = catalog();

    let accepted: Vec<&str> = rejected()
        .map(|(kwarg, _)| kwarg.name())
        .filter(|kwarg| {
            catalog.args[*kwarg]
                .redis_py
                .get(&catalog.meta.redis_py_locked)
                .is_some_and(|signatures| {
                    signatures
                        .iter()
                        .any(|signature| PROXY_SIGNATURES.contains(&signature.as_str()))
                })
        })
        .collect();
    assert!(accepted.is_empty(), "{accepted:?}");
}
