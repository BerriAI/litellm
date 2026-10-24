use litellm_auth_types::SecretValue;
use serde::Deserialize;

use crate::Spelled;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TracingStoreKind {
    Clickhouse,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClickHouseStoreSettings {
    #[serde(rename = "type")]
    pub kind: TracingStoreKind,
    pub url: Option<SecretValue>,
    pub database: Option<String>,
    pub retention_days: Option<Spelled<f64>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum TracingStoreSettings {
    ClickHouse(ClickHouseStoreSettings),
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct TracingSettings {
    pub store: Option<TracingStoreSettings>,
}

#[cfg(test)]
mod tests {
    use crate::{Config, Spelled, TracingStoreSettings};

    #[test]
    fn tracing_settings_are_typed_and_redact_the_url() {
        let config = Config::from_yaml(
        "general_settings:\n  tracing:\n    store:\n      type: clickhouse\n      url: https://writer:SENTINEL@example.com\n      database: analytics\n      retention_days: 7\n",
    )
    .unwrap();
        let tracing = config.general_settings.tracing.as_ref().unwrap();
        let Some(TracingStoreSettings::ClickHouse(store)) = tracing.store.as_ref() else {
            panic!("expected ClickHouse tracing store")
        };
        assert_eq!(
            store.url.as_ref().unwrap().expose(),
            "https://writer:SENTINEL@example.com"
        );
        assert_eq!(store.database.as_deref(), Some("analytics"));
        assert_eq!(store.retention_days, Some(Spelled::Value(7.0)));
        assert!(!format!("{config:?}").contains("SENTINEL"));
    }

    #[test]
    fn tracing_settings_accept_environment_references() {
        let config = Config::from_yaml(
        "general_settings:\n  tracing:\n    store:\n      type: clickhouse\n      url: os.environ/CLICKHOUSE_URL\n      retention_days: os.environ/RETENTION_DAYS\n",
    )
    .unwrap();
        let Some(TracingStoreSettings::ClickHouse(store)) =
            config.general_settings.tracing.unwrap().store
        else {
            panic!("expected ClickHouse tracing store")
        };
        assert_eq!(
            store.retention_days,
            Some(Spelled::Text("os.environ/RETENTION_DAYS".to_owned()))
        );
    }

    #[test]
    fn tracing_settings_reject_string_store() {
        assert!(
            Config::from_yaml("general_settings:\n  tracing:\n    store: clickhouse\n").is_err()
        );
    }

    #[test]
    fn tracing_settings_reject_removed_reader_configuration() {
        assert!(Config::from_yaml("general_settings:\n  tracing:\n    store:\n      type: clickhouse\n      reader_url: http://localhost:8123\n").is_err());
    }
}
