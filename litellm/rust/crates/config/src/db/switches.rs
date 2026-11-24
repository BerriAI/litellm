//! Whether the proxy reads config from its database at all, and which kinds of stored object it
//! loads when it does.

use litellm_env::Lookup;
use litellm_python_compat::flags::{lower_is_true, str_to_bool};

use crate::{GeneralSettings, Value, db::settings::SettingsRows};

/// The variable that turns on `store_model_in_db` whatever the config says.
pub const STORE_MODEL_IN_DB: &str = "STORE_MODEL_IN_DB";
const STORE_MODEL_IN_DB_KEY: &str = "store_model_in_db";
const SUPPORTED_DB_OBJECTS: &str = "supported_db_objects";
const FIPS_MODE: &str = "fips_mode";

/// `general_settings.fips_mode` as `fips_mode_setting` in `litellm/proxy/common_utils/fips.py`
/// reads it: off when the key is absent, a bool as written, text through `str_to_bool`, and
/// anything else, an empty key or an unset `os.environ/` reference included, not a boolean, which
/// makes the proxy refuse to start. Only the config file sets it: it is no `ConfigGeneralSettings`
/// field, so the stored `general_settings` row cannot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FipsMode {
    Off,
    On,
    NotABoolean,
}

impl FipsMode {
    /// The mode `general` sets, read after its `os.environ/` references are resolved.
    pub fn of(general: &GeneralSettings) -> Self {
        let setting = match general.additional_fields.get(FIPS_MODE) {
            None => Some(false),
            Some(Value::Bool(on)) => Some(*on),
            Some(Value::String(text)) => str_to_bool(text),
            Some(_) => None,
        };
        match setting {
            Some(false) => Self::Off,
            Some(true) => Self::On,
            None => Self::NotABoolean,
        }
    }
}

/// Whether the proxy layers the database over the config file, as it decides at startup in
/// `initialize_scheduled_background_jobs`: on when the config file's `general_settings.store_model_in_db` is
/// `true`, when `STORE_MODEL_IN_DB` reads as true in `environment`, or when the stored
/// `general_settings` row holds `true` or text that lower-cases to `"true"`. Any one of the three
/// turns it on, so the stored row turns it on even under a file that sets it to `false`.
///
/// `file` is the setting of the config loaded without the database, with its own
/// `environment_variables` in front of `environment` as `config-load` builds it.
pub fn store_model_in_db(
    file: Option<bool>,
    environment: &dyn Lookup,
    rows: &SettingsRows,
) -> bool {
    file == Some(true)
        || environment.enabled(STORE_MODEL_IN_DB) == Some(true)
        || match rows.general_settings.get(STORE_MODEL_IN_DB_KEY) {
            Some(Value::Bool(on)) => *on,
            Some(Value::String(text)) => lower_is_true(text),
            _ => false,
        }
}

/// A kind of object the proxy can load from its database, spelled as
/// `general_settings.supported_db_objects` names it. Each is one `should_load_db_object` call in
/// `litellm/proxy/proxy_server.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr, strum::VariantArray)]
pub enum DbObject {
    #[strum(serialize = "models")]
    Models,
    #[strum(serialize = "mcp")]
    Mcp,
    #[strum(serialize = "guardrails")]
    Guardrails,
    #[strum(serialize = "policies")]
    Policies,
    #[strum(serialize = "vector_stores")]
    VectorStores,
    #[strum(serialize = "vector_store_indexes")]
    VectorStoreIndexes,
    #[strum(serialize = "agents")]
    Agents,
    #[strum(serialize = "pass_through_endpoints")]
    PassThroughEndpoints,
    #[strum(serialize = "prompts")]
    Prompts,
    #[strum(serialize = "search_tools")]
    SearchTools,
    #[strum(serialize = "tools")]
    Tools,
    #[strum(serialize = "anthropic_beta_headers")]
    AnthropicBetaHeaders,
    #[strum(serialize = "sso_settings")]
    SsoSettings,
    #[strum(serialize = "cache_settings")]
    CacheSettings,
    #[strum(serialize = "semantic_filter_settings")]
    SemanticFilterSettings,
    #[strum(serialize = "websearch_interception_settings")]
    WebsearchInterceptionSettings,
    #[strum(serialize = "config_overrides")]
    ConfigOverrides,
    #[strum(serialize = "model_cost_map")]
    ModelCostMap,
}

/// `general_settings.supported_db_objects`: which kinds of stored object the proxy loads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SupportedDbObjects {
    /// Unset or `null`, so every kind loads.
    All,
    /// Set to something other than a list, which the proxy warns about and then loads every kind.
    NotAList,
    /// The names listed. An entry that is not text names no kind.
    Only(Box<[String]>),
}

impl SupportedDbObjects {
    /// The setting in `settings`, which is the config with the database already layered in, since
    /// the proxy reads it from its live `general_settings`.
    pub fn of(settings: &GeneralSettings) -> Self {
        match settings.additional_fields.get(SUPPORTED_DB_OBJECTS) {
            None | Some(Value::Null) => Self::All,
            Some(Value::Sequence(names)) => Self::Only(
                names
                    .iter()
                    .filter_map(|name| name.as_str().map(str::to_owned))
                    .collect(),
            ),
            Some(_) => Self::NotAList,
        }
    }

    /// `should_load_db_object(object)`.
    pub fn loads(&self, object: DbObject) -> bool {
        match self {
            Self::All | Self::NotAList => true,
            Self::Only(names) => names.iter().any(|name| name == <&str>::from(object)),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{DbObject, FipsMode, STORE_MODEL_IN_DB, SupportedDbObjects, store_model_in_db};
    use crate::{Config, db::settings::SettingsRows};

    fn rows(json: &str) -> SettingsRows {
        serde_json::from_str(json).expect("test rows parse")
    }

    #[rstest]
    #[case::all_unset(None, None, "{}", false)]
    #[case::file_on(Some(true), None, "{}", true)]
    #[case::file_off(Some(false), None, "{}", false)]
    #[case::variable_true_in_any_case_and_padding(None, Some(" True "), "{}", true)]
    #[case::variable_one_is_not_true(None, Some("1"), "{}", false)]
    #[case::variable_false_does_not_turn_the_file_off(Some(true), Some("false"), "{}", true)]
    #[case::stored_true(
        None,
        None,
        r#"{"general_settings": {"store_model_in_db": true}}"#,
        true
    )]
    #[case::stored_text_true_in_any_case(
        None,
        None,
        r#"{"general_settings": {"store_model_in_db": "TRUE"}}"#,
        true
    )]
    #[case::stored_text_yes_is_not_true(
        None,
        None,
        r#"{"general_settings": {"store_model_in_db": "yes"}}"#,
        false
    )]
    #[case::stored_true_overrides_a_file_off(
        Some(false),
        None,
        r#"{"general_settings": {"store_model_in_db": true}}"#,
        true
    )]
    #[case::stored_false(
        None,
        None,
        r#"{"general_settings": {"store_model_in_db": false}}"#,
        false
    )]
    fn store_model_in_db_is_on_when_any_source_is(
        #[case] file: Option<bool>,
        #[case] variable: Option<&str>,
        #[case] stored: &str,
        #[case] expected: bool,
    ) {
        let environment = |name: &str| {
            variable
                .filter(|_| name == STORE_MODEL_IN_DB)
                .map(str::to_owned)
        };

        assert_eq!(
            store_model_in_db(file, &environment, &rows(stored)),
            expected
        );
    }

    #[rstest]
    #[case::unset_loads_everything("", DbObject::Mcp, true)]
    #[case::null_loads_everything("supported_db_objects: null", DbObject::Mcp, true)]
    #[case::listed_kind_loads("supported_db_objects: [models]", DbObject::Models, true)]
    #[case::unlisted_kind_does_not_load("supported_db_objects: [models]", DbObject::Mcp, false)]
    #[case::empty_list_loads_nothing("supported_db_objects: []", DbObject::Models, false)]
    #[case::entry_that_is_not_text_names_nothing(
        "supported_db_objects: [1, mcp]",
        DbObject::Mcp,
        true
    )]
    #[case::not_a_list_loads_everything("supported_db_objects: models", DbObject::Mcp, true)]
    fn supported_db_objects_gates_each_kind(
        #[case] general_settings: &str,
        #[case] object: DbObject,
        #[case] expected: bool,
    ) {
        let config = Config::from_yaml(&format!("general_settings: {{{general_settings}}}"))
            .expect("test config parses");

        assert_eq!(
            SupportedDbObjects::of(&config.general_settings).loads(object),
            expected
        );
    }

    #[rstest]
    #[case::absent("{}", FipsMode::Off)]
    #[case::true_bool("{fips_mode: true}", FipsMode::On)]
    #[case::false_bool("{fips_mode: false}", FipsMode::Off)]
    #[case::text_any_case_padded("{fips_mode: ' TRUE '}", FipsMode::On)]
    #[case::false_text("{fips_mode: 'False'}", FipsMode::Off)]
    #[case::other_text("{fips_mode: 'yes'}", FipsMode::NotABoolean)]
    #[case::empty_key("{fips_mode: null}", FipsMode::NotABoolean)]
    #[case::number("{fips_mode: 1}", FipsMode::NotABoolean)]
    fn fips_mode_reads_as_python_reads_it(#[case] general: &str, #[case] expected: FipsMode) {
        let general: crate::GeneralSettings = serde_yaml_ng::from_str(general).unwrap();

        assert_eq!(FipsMode::of(&general), expected);
    }

    #[rstest]
    fn a_stored_row_cannot_set_fips_mode() {
        let rows: crate::db::settings::SettingsRows =
            serde_json::from_str(r#"{"general_settings": {"fips_mode": true}}"#).unwrap();
        let overlaid = crate::db::settings::overlay(
            serde_yaml_ng::from_str("general_settings: {}").unwrap(),
            &rows,
        );
        let config = crate::Config::from_value(overlaid.document).unwrap();

        assert_eq!(FipsMode::of(&config.general_settings), FipsMode::Off);
    }
}
