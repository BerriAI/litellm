//! The settings rows of `LiteLLM_Config` layered over the config document, the way
//! `ProxyConfig._update_config_from_db` and `SettingsStore` in `litellm/proxy/config_resolvers/`
//! layer them.
//!
//! The overlay works on the document after its includes are merged and before any `os.environ/`
//! reference is resolved, which is where the proxy runs it. That way an `os.environ/` reference a
//! stored value holds resolves like one in the file, and the stored `environment_variables` land in
//! the document's own `environment_variables`, which `config-load` puts in front of the environment
//! every reference is read from. Nothing here touches the process environment.
//!
//! The rules, per section:
//! - The file wins per key. A stored value only fills a key the file does not declare, and a key
//!   the file declares, even as `null`, keeps the file's value. A stored `null` counts as absent
//! - `general_settings` takes only the keys the proxy manages from that row, and never the keys it
//!   reads from the `LiteLLM_UISettings` table instead. Those keys come from the `ui_settings` column
//!   of that table, which the file still wins over
//! - `litellm_settings` takes only the keys a stored row changes in the proxy
//! - `general_settings.pass_through_endpoints` is the stored endpoints followed by each file
//!   endpoint whose `path` no stored endpoint has
//! - `environment_variables` drops `LITELLM_ENABLE_MCP_STDIO` in any case, and adds each stored
//!   name upper-cased beside the name as stored, as the proxy exports both

mod keys;

use std::collections::BTreeMap;

use litellm_secret_value::SecretValue;
use serde::Deserialize;
use serde_yaml_ng::Mapping;

use crate::{Object, Value};

/// A variable the proxy never takes from the database, only from its own environment:
/// `MCP_STDIO_ENABLED_ENV_VAR` in `litellm/proxy/mcp_server/stdio_gate.py`.
const MCP_STDIO_ENABLED: &str = "LITELLM_ENABLE_MCP_STDIO";
const PASS_THROUGH_ENDPOINTS: &str = "pass_through_endpoints";

/// A section of the config a `LiteLLM_Config` row of the same name layers over.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, strum::IntoStaticStr, strum::VariantArray,
)]
pub enum Section {
    #[strum(serialize = "environment_variables")]
    EnvironmentVariables,
    #[strum(serialize = "general_settings")]
    GeneralSettings,
    #[strum(serialize = "litellm_settings")]
    LitellmSettings,
    #[strum(serialize = "router_settings")]
    RouterSettings,
}

/// The `param_value` of each settings row of `LiteLLM_Config`, decrypted by the caller, and the
/// `LiteLLM_UISettings` row. A missing row, a `null` one or one that is not a JSON object is an
/// empty section. A stored variable the caller could not decrypt, or that is `null`, is left out of
/// `environment_variables`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct SettingsRows {
    pub general_settings: Object,
    pub router_settings: Object,
    pub litellm_settings: Object,
    pub environment_variables: BTreeMap<String, SecretValue>,
    pub ui_settings: UiSettings,
}

/// The `ui_settings` column of the `LiteLLM_UISettings` row, read as
/// `sync_ui_settings_to_general_settings` in
/// `litellm/proxy/ui_crud_endpoints/proxy_setting_endpoints.py` reads it. The column holds either
/// the settings object or, as some writers save it, that object's JSON text as a JSON string. A
/// missing row, or a value that is neither, is no settings.
#[derive(Clone, Debug, Default)]
pub struct UiSettings(Object);

impl UiSettings {
    /// The settings in `column`, the column's JSON text as the database returns it.
    pub fn from_json(column: &str) -> Self {
        serde_json::from_str(column).unwrap_or_default()
    }
}

impl std::ops::Deref for UiSettings {
    type Target = Object;

    fn deref(&self) -> &Object {
        &self.0
    }
}

impl<'de> Deserialize<'de> for UiSettings {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Stored {
            Object(Object),
            Text(String),
            Other(serde::de::IgnoredAny),
        }
        Ok(match Stored::deserialize(deserializer)? {
            Stored::Object(settings) => Self(settings),
            Stored::Text(text) => serde_json::from_str(&text).map(Self).unwrap_or_default(),
            Stored::Other(_) => Self::default(),
        })
    }
}

/// A key the file sets to something other than the stored value, so the stored value is never
/// used. The proxy logs each one with `config_ownership_message`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Shadowed {
    pub section: Section,
    pub key: String,
}

/// The document with the stored settings layered in, and the stored keys the file shadows.
#[derive(Clone)]
pub struct Overlaid {
    pub document: Value,
    pub shadowed: Box<[Shadowed]>,
}

/// `document` holds decrypted stored values and the master key, so `Debug` leaves it out.
impl std::fmt::Debug for Overlaid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Overlaid")
            .field("document", &"<redacted>")
            .field("shadowed", &self.shadowed)
            .finish()
    }
}

/// `document` with `rows` layered under it. Call it only when the proxy would read the database,
/// as [`super::switches::store_model_in_db`] says.
pub fn overlay(document: Value, rows: &SettingsRows) -> Overlaid {
    let Value::Mapping(mut document) = document else {
        return Overlaid {
            document,
            shadowed: Box::default(),
        };
    };
    let mut shadowed = Vec::new();
    for &section in <Section as strum::VariantArray>::VARIANTS {
        let name = Value::String(<&str>::from(section).to_owned());
        let file = document.remove(&name);
        let (merged, keys) = layered(section, file, rows.stored(section));
        shadowed.extend(keys.into_iter().map(|key| Shadowed { section, key }));
        if let Some(merged) = merged {
            document.insert(name, merged);
        }
    }
    shadowed.sort();
    Overlaid {
        document: Value::Mapping(document),
        shadowed: shadowed.into(),
    }
}

impl SettingsRows {
    /// The non-null stored values the proxy applies to `section`.
    fn stored(&self, section: Section) -> Mapping {
        match section {
            Section::GeneralSettings => {
                let ui = |key: &str| keys::UI_SETTINGS.binary_search(&key).is_ok();
                let mut stored = applied(&self.general_settings, |key| {
                    keys::MANAGED_GENERAL_SETTINGS.binary_search(&key).is_ok() && !ui(key)
                });
                stored.extend(applied(&self.ui_settings.0, ui));
                stored
            }
            Section::RouterSettings => applied(&self.router_settings, |_| true),
            Section::LitellmSettings => applied(&self.litellm_settings, |key| {
                keys::STORED_LITELLM_SETTINGS.binary_search(&key).is_ok()
            }),
            Section::EnvironmentVariables => self.environment(),
        }
    }

    fn environment(&self) -> Mapping {
        let stored = self
            .environment_variables
            .iter()
            .filter(|(name, _)| name.to_uppercase() != MCP_STDIO_ENABLED);
        let upper = stored
            .clone()
            .map(|(name, value)| (name.to_uppercase(), value));
        stored
            .map(|(name, value)| (name.clone(), value))
            .chain(upper)
            .map(|(name, value)| {
                (
                    Value::String(name),
                    Value::String(value.expose().to_owned()),
                )
            })
            .collect()
    }
}

fn applied(row: &Object, applies: impl Fn(&str) -> bool) -> Mapping {
    row.iter()
        .filter(|(key, value)| !value.is_null() && applies(key))
        .map(|(key, value)| (Value::String(key.clone()), value.clone()))
        .collect()
}

/// The file's `section` with `stored` under it, `None` when the document should not have the
/// section, and the keys the file shadows.
fn layered(
    section: Section,
    file: Option<Value>,
    mut stored: Mapping,
) -> (Option<Value>, Vec<String>) {
    let mut merged = match file {
        Some(Value::Mapping(own)) => own,
        file if stored.is_empty() => return (file, Vec::new()),
        _ => Mapping::new(),
    };
    if section == Section::GeneralSettings {
        let key = Value::String(PASS_THROUGH_ENDPOINTS.into());
        let declared = merged.remove(&key).filter(|value| !value.is_null());
        let endpoints = stored.remove(&key);
        if declared.is_some() || endpoints.is_some() {
            merged.insert(key, pass_through_endpoints(endpoints, declared));
        }
    }
    let mut shadowed = Vec::new();
    for (key, value) in stored {
        match merged.get(&key) {
            Some(own) if *own != value => shadowed.extend(key.as_str().map(str::to_owned)),
            Some(_) => {}
            None => {
                merged.insert(key, value);
            }
        }
    }
    (Some(Value::Mapping(merged)), shadowed)
}

/// `_pass_through_endpoints_beside_db`: every stored endpoint, then each file endpoint whose
/// `path` no stored endpoint has. A value that is not a list counts as no endpoints.
fn pass_through_endpoints(stored: Option<Value>, declared: Option<Value>) -> Value {
    let stored = endpoints(stored);
    let paths: Vec<_> = stored.iter().filter_map(path).cloned().collect();
    let beside = endpoints(declared)
        .into_iter()
        .filter(|endpoint| path(endpoint).is_none_or(|path| !paths.contains(path)));
    Value::Sequence(stored.iter().cloned().chain(beside).collect())
}

fn endpoints(value: Option<Value>) -> Vec<Value> {
    match value {
        Some(Value::Sequence(endpoints)) => endpoints,
        _ => Vec::new(),
    }
}

/// An endpoint's `path` as Python's `endpoint.get("path")` reads it, `None` for an endpoint that
/// is not a mapping.
fn path(endpoint: &Value) -> Option<&Value> {
    const NULL: &Value = &Value::Null;
    match endpoint {
        Value::Mapping(endpoint) => Some(endpoint.get(PASS_THROUGH_PATH).unwrap_or(NULL)),
        _ => None,
    }
}

const PASS_THROUGH_PATH: &str = "path";

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde::Deserialize;

    use super::{Section, SettingsRows, Shadowed, UiSettings, keys, overlay};
    use crate::Value;

    fn yaml(text: &str) -> Value {
        litellm_yaml::from_str(text).expect("test YAML parses")
    }

    fn rows(json: &str) -> SettingsRows {
        serde_json::from_str(json).expect("test rows parse")
    }

    #[test]
    fn debug_output_leaves_the_overlaid_document_out() {
        let overlaid = overlay(
            yaml("general_settings: {master_key: sk-file-secret}"),
            &rows(r#"{"general_settings": {"master_key": "sk-stored-secret"}}"#),
        );
        let printed = format!("{overlaid:?}");
        assert!(!printed.contains("secret"), "{printed}");
        assert!(printed.contains("master_key"), "{printed}");
    }

    fn shadowed(keys: &[(Section, &str)]) -> Box<[Shadowed]> {
        keys.iter()
            .map(|&(section, key)| Shadowed {
                section,
                key: key.to_owned(),
            })
            .collect()
    }

    #[rstest]
    #[case::stored_value_fills_a_key_the_file_leaves_unset(
        "general_settings: {max_parallel_requests: 1}",
        r#"{"general_settings": {"global_max_parallel_requests": 5}}"#,
        "general_settings: {max_parallel_requests: 1, global_max_parallel_requests: 5}",
        &[]
    )]
    #[case::file_value_wins_and_shadows_a_different_stored_value(
        "router_settings: {num_retries: 1}",
        r#"{"router_settings": {"num_retries": 3, "timeout": 30}}"#,
        "router_settings: {num_retries: 1, timeout: 30}",
        &[(Section::RouterSettings, "num_retries")]
    )]
    #[case::equal_stored_value_is_not_shadowed(
        "router_settings: {num_retries: 3}",
        r#"{"router_settings": {"num_retries": 3}}"#,
        "router_settings: {num_retries: 3}",
        &[]
    )]
    #[case::file_null_still_owns_the_key(
        "general_settings: {completion_model: null}",
        r#"{"general_settings": {"completion_model": "db-model"}}"#,
        "general_settings: {completion_model: null}",
        &[(Section::GeneralSettings, "completion_model")]
    )]
    #[case::stored_null_counts_as_absent(
        "general_settings: {}",
        r#"{"general_settings": {"completion_model": null}}"#,
        "general_settings: {}",
        &[]
    )]
    #[case::stored_null_adds_no_section(
        "model_list: []",
        r#"{"router_settings": {"num_retries": null}}"#,
        "model_list: []",
        &[]
    )]
    #[case::general_settings_key_the_proxy_does_not_manage_is_dropped(
        "general_settings: {}",
        r#"{"general_settings": {"not_a_general_setting": 1}}"#,
        "general_settings: {}",
        &[]
    )]
    #[case::general_settings_key_from_the_ui_settings_table_is_dropped(
        "general_settings: {}",
        r#"{"general_settings": {"forward_client_headers_to_llm_api": true}}"#,
        "general_settings: {}",
        &[]
    )]
    #[case::ui_settings_value_fills_general_settings(
        "general_settings: {master_key: sk-file}",
        r#"{"ui_settings": {"forward_client_headers_to_llm_api": true}}"#,
        "general_settings: {master_key: sk-file, forward_client_headers_to_llm_api: true}",
        &[]
    )]
    #[case::ui_settings_saved_as_json_text_is_read(
        "general_settings: {}",
        r#"{"ui_settings": "{\"forward_client_headers_to_llm_api\": true}"}"#,
        "general_settings: {forward_client_headers_to_llm_api: true}",
        &[]
    )]
    #[case::ui_settings_value_wins_over_the_general_settings_row(
        "general_settings: {}",
        r#"{"general_settings": {"forward_client_headers_to_llm_api": false}, "ui_settings": {"forward_client_headers_to_llm_api": true}}"#,
        "general_settings: {forward_client_headers_to_llm_api: true}",
        &[]
    )]
    #[case::file_wins_over_ui_settings_and_shadows_it(
        "general_settings: {forward_client_headers_to_llm_api: false}",
        r#"{"ui_settings": {"forward_client_headers_to_llm_api": true}}"#,
        "general_settings: {forward_client_headers_to_llm_api: false}",
        &[(Section::GeneralSettings, "forward_client_headers_to_llm_api")]
    )]
    #[case::ui_settings_key_that_is_no_runtime_flag_is_dropped(
        "general_settings: {}",
        r#"{"ui_settings": {"enable_chat_ui": true, "master_key": "sk-ui"}}"#,
        "general_settings: {}",
        &[]
    )]
    #[case::ui_settings_null_counts_as_absent(
        "general_settings: {}",
        r#"{"ui_settings": {"forward_client_headers_to_llm_api": null}}"#,
        "general_settings: {}",
        &[]
    )]
    #[case::unreadable_ui_settings_text_is_no_settings(
        "general_settings: {}",
        r#"{"ui_settings": "{not json"}"#,
        "general_settings: {}",
        &[]
    )]
    #[case::ui_settings_that_are_not_an_object_are_no_settings(
        "model_list: []",
        r#"{"ui_settings": ["forward_client_headers_to_llm_api"]}"#,
        "model_list: []",
        &[]
    )]
    #[case::ui_settings_create_the_general_settings_section(
        "model_list: []",
        r#"{"ui_settings": {"disable_custom_api_keys": true}}"#,
        "{model_list: [], general_settings: {disable_custom_api_keys: true}}",
        &[]
    )]
    #[case::litellm_settings_key_a_row_cannot_change_is_dropped(
        "litellm_settings: {}",
        r#"{"litellm_settings": {"drop_params": true, "cache": true}}"#,
        "litellm_settings: {}",
        &[]
    )]
    #[case::litellm_settings_overridable_key_and_callbacks_are_applied(
        "litellm_settings: {}",
        r#"{"litellm_settings": {"turn_off_message_logging": true, "callbacks": ["langfuse"]}}"#,
        "litellm_settings: {turn_off_message_logging: true, callbacks: [langfuse]}",
        &[]
    )]
    #[case::shadow_is_reported_only_for_a_litellm_settings_key_a_row_can_change(
        "litellm_settings: {drop_params: false, turn_off_message_logging: false}",
        r#"{"litellm_settings": {"drop_params": true, "turn_off_message_logging": true}}"#,
        "litellm_settings: {drop_params: false, turn_off_message_logging: false}",
        &[(Section::LitellmSettings, "turn_off_message_logging")]
    )]
    #[case::pass_through_endpoints_are_stored_first_then_file_paths_not_stored(
        "general_settings: {pass_through_endpoints: [{path: /a, target: file}, {path: /b, target: file}]}",
        r#"{"general_settings": {"pass_through_endpoints": [{"path": "/b", "target": "db"}, {"path": "/c", "target": "db"}]}}"#,
        "general_settings: {pass_through_endpoints: [{path: /b, target: db}, {path: /c, target: db}, {path: /a, target: file}]}",
        &[]
    )]
    #[case::file_endpoint_without_path_is_dropped_beside_a_stored_one_without_path(
        "general_settings: {pass_through_endpoints: [{target: file}, other]}",
        r#"{"general_settings": {"pass_through_endpoints": [{"target": "db"}]}}"#,
        "general_settings: {pass_through_endpoints: [{target: db}, other]}",
        &[]
    )]
    #[case::stored_pass_through_endpoints_create_the_section(
        "model_list: []",
        r#"{"general_settings": {"pass_through_endpoints": [{"path": "/x"}]}}"#,
        "{model_list: [], general_settings: {pass_through_endpoints: [{path: /x}]}}",
        &[]
    )]
    #[case::file_null_pass_through_endpoints_without_stored_ones_is_dropped(
        "general_settings: {pass_through_endpoints: null, master_key: sk-file}",
        "{}",
        "general_settings: {master_key: sk-file}",
        &[]
    )]
    #[case::stored_pass_through_endpoints_that_are_not_a_list_count_as_none(
        "general_settings: {pass_through_endpoints: [{path: /a}]}",
        r#"{"general_settings": {"pass_through_endpoints": "nope"}}"#,
        "general_settings: {pass_through_endpoints: [{path: /a}]}",
        &[]
    )]
    #[case::stored_variables_are_added_upper_cased_and_the_mcp_stdio_flag_is_dropped(
        "environment_variables: {}",
        r#"{"environment_variables": {"lower_name": "v", "litellm_enable_mcp_stdio": "true", "LITELLM_ENABLE_MCP_STDIO": "true"}}"#,
        "environment_variables: {lower_name: v, LOWER_NAME: v}",
        &[]
    )]
    #[case::file_variable_wins_over_a_stored_one(
        "environment_variables: {SHARED: file}",
        r#"{"environment_variables": {"SHARED": "db", "REFERENCE": "os.environ/OTHER"}}"#,
        "environment_variables: {SHARED: file, REFERENCE: os.environ/OTHER}",
        &[(Section::EnvironmentVariables, "SHARED")]
    )]
    #[case::section_the_file_leaves_null_takes_the_stored_values(
        "router_settings: null",
        r#"{"router_settings": {"num_retries": 2}}"#,
        "router_settings: {num_retries: 2}",
        &[]
    )]
    #[case::section_the_file_leaves_null_stays_null_without_rows(
        "router_settings: null",
        "{}",
        "router_settings: null",
        &[]
    )]
    #[case::sections_without_rows_are_left_alone(
        "{model_list: [{model_name: m}], include: [other.yaml]}",
        r#"{"router_settings": {"num_retries": 2}}"#,
        "{model_list: [{model_name: m}], include: [other.yaml], router_settings: {num_retries: 2}}",
        &[]
    )]
    fn overlay_layers_stored_settings_under_the_file(
        #[case] file: &str,
        #[case] stored: &str,
        #[case] expected: &str,
        #[case] expected_shadowed: &[(Section, &str)],
    ) {
        let overlaid = overlay(yaml(file), &rows(stored));

        assert_eq!(overlaid.document, yaml(expected));
        assert_eq!(overlaid.shadowed, shadowed(expected_shadowed));
    }

    #[rstest]
    #[case::object(r#"{"forward_client_headers_to_llm_api": true}"#)]
    #[case::json_text(r#""{\"forward_client_headers_to_llm_api\": true}""#)]
    fn ui_settings_column_reads_as_the_settings_object(#[case] column: &str) {
        let rows = SettingsRows {
            ui_settings: UiSettings::from_json(column),
            ..SettingsRows::default()
        };

        let overlaid = overlay(yaml("general_settings: {}"), &rows);

        assert_eq!(
            overlaid.document,
            yaml("general_settings: {forward_client_headers_to_llm_api: true}")
        );
    }

    #[derive(Deserialize)]
    struct Fixture {
        tables: Tables,
        cases: Vec<FixtureCase>,
    }

    #[derive(Deserialize)]
    struct Tables {
        managed_general_settings: Vec<String>,
        ui_settings: Vec<String>,
        stored_litellm_settings: Vec<String>,
    }

    #[derive(Deserialize)]
    struct FixtureCase {
        name: String,
        file: Value,
        rows: SettingsRows,
        merged: Value,
        shadowed: Vec<(String, String)>,
    }

    /// Written once by `scripts/generate_db_settings_fixtures.py` from the Python proxy.
    fn fixture() -> Fixture {
        serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/generated/db_settings.json"
        )))
        .expect("generated/db_settings.json parses")
    }

    #[rstest]
    #[case::managed_general_settings(keys::MANAGED_GENERAL_SETTINGS, |tables: Tables| tables.managed_general_settings)]
    #[case::ui_settings(keys::UI_SETTINGS, |tables: Tables| tables.ui_settings)]
    #[case::stored_litellm_settings(keys::STORED_LITELLM_SETTINGS, |tables: Tables| tables.stored_litellm_settings)]
    fn key_table_matches_python(#[case] table: &[&str], #[case] python: fn(Tables) -> Vec<String>) {
        assert_eq!(table, python(fixture().tables).as_slice());
    }

    #[rstest]
    #[case::general_settings_file_wins_and_db_fills("general_settings_file_wins_and_db_fills")]
    #[case::router_and_litellm_settings("router_and_litellm_settings")]
    #[case::pass_through_endpoints_union("pass_through_endpoints_union")]
    #[case::db_only_sections("db_only_sections")]
    #[case::environment_variables("environment_variables")]
    #[case::ui_settings_object("ui_settings_object")]
    #[case::ui_settings_json_text("ui_settings_json_text")]
    #[case::ui_settings_unreadable("ui_settings_unreadable")]
    #[case::ui_settings_not_an_object("ui_settings_not_an_object")]
    #[case::no_rows("no_rows")]
    fn overlay_matches_python(#[case] name: &str) {
        let case = fixture()
            .cases
            .into_iter()
            .find(|case| case.name == name)
            .expect("fixture has the case");

        let overlaid = overlay(case.file, &case.rows);

        assert_eq!(overlaid.document, case.merged);
        let shadowed: Vec<_> = overlaid
            .shadowed
            .iter()
            .map(|shadowed| {
                (
                    <&str>::from(shadowed.section).to_owned(),
                    shadowed.key.clone(),
                )
            })
            .collect();
        assert_eq!(shadowed, case.shadowed);
    }
}
