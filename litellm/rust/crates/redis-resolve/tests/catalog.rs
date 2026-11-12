use std::collections::{BTreeMap, BTreeSet};

use litellm_redis_resolve::{Asked, Error, Handling, SETTINGS, Setting, interpret, kwargs};
use rstest::rstest;
use serde::Deserialize;
use serde_json::json;

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

fn rejected() -> impl Iterator<Item = (&'static Setting, Asked)> {
    SETTINGS
        .iter()
        .filter_map(|setting| match setting.handling {
            Handling::Reject { asked, .. } => Some((setting, asked)),
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
fn every_catalogued_kwarg_has_exactly_one_settings_row() {
    let rows: Vec<&str> = SETTINGS.iter().map(|setting| setting.kwarg).collect();
    let unique: BTreeSet<&str> = rows.iter().copied().collect();
    assert_eq!(rows.len(), unique.len(), "a kwarg has two SETTINGS rows");

    let catalog = catalog();
    let catalogued: BTreeSet<&str> = catalog.args.keys().map(String::as_str).collect();
    assert_eq!(unique, catalogued);
}

#[rstest]
fn settings_read_redis_environment_names_exactly_where_python_does() {
    let catalog = catalog();

    let differing: Vec<&str> = SETTINGS
        .iter()
        .filter(|setting| {
            catalog
                .args
                .get(setting.kwarg)
                .is_some_and(|arg| arg.environment != setting.environment)
        })
        .map(|setting| setting.kwarg)
        .collect();
    assert!(differing.is_empty(), "{differing:?}");
}

#[rstest]
fn every_warned_setting_is_reported() {
    let warned = SETTINGS
        .iter()
        .filter(|setting| matches!(setting.handling, Handling::Warn { met: None, .. }));

    for setting in warned {
        let resolved = interpret(&host_with(setting.kwarg, json!("set")), None, None).unwrap();

        let reported: Vec<&str> = resolved
            .ignored
            .iter()
            .map(|ignored| ignored.setting.as_str())
            .collect();
        assert_eq!(reported, vec![setting.kwarg]);
    }
}

#[rstest]
fn every_rejected_setting_fails_when_asked_for_and_not_otherwise() {
    for (setting, asked) in rejected() {
        let (asking, not_asking) = match asked {
            Asked::WhenTrue => (json!(true), json!(false)),
            Asked::WhenSet => (json!(["VERIFY_CRL_CHECK_LEAF"]), json!([])),
        };

        let error = interpret(&host_with(setting.kwarg, asking), None, None).unwrap_err();
        assert!(
            matches!(&error, Error::Unenforceable { setting: source, .. } if source == setting.kwarg),
            "{error:?}"
        );
        assert!(interpret(&host_with(setting.kwarg, not_asking), None, None).is_ok());
    }
}

#[rstest]
fn python_already_refuses_every_rejected_setting() {
    let catalog = catalog();

    let accepted: Vec<&str> = rejected()
        .map(|(setting, _)| setting.kwarg)
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
