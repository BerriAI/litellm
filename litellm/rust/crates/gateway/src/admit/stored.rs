//! What the proxy's database adds to the config the native app was built from. With
//! `store_model_in_db` on, from the variable, the config or the database's own `general_settings`
//! row, the proxy serves its database's models, settings, guardrails and policy attachments on top
//! of the config, and the native app reads none of them. A request goes to Python only over what it
//! reaches: a model group a database model joins, or a stored setting, guardrail or policy that
//! reaches every request. The database is read when the admit is built and again on the proxy's
//! own reload interval, so a change made through the UI reaches the admit as soon as it reaches
//! Python.

use std::{
    collections::BTreeSet,
    sync::{Arc, Weak},
    time::Duration,
};

use arc_swap::ArcSwap;
use litellm_config::Config;
use litellm_env::Lookup;
use litellm_python_compat::flags::str_to_bool;
use litellm_repositories::{StoredConfig, StoredConfigRepository, StoredSection};

const STORE_MODEL_IN_DB: &str = "STORE_MODEL_IN_DB";
const ENVIRONMENT_VARIABLES: &str = "environment_variables";

/// The `general_settings` keys a stored row can set that no request the native app serves reads:
/// the switch itself, the proxy's own pool and reload timing, health checks that do not route,
/// retention schedules and UI surfaces. Every other stored key reaches the request or the native
/// app reads its own copy of it from the config, so it goes to Python.
const INERT_GENERAL_SETTINGS: [&str; 26] = [
    "store_model_in_db",
    "proxy_config_reload_interval_seconds",
    "health_check_interval",
    "health_check_concurrency",
    "enable_public_model_hub",
    "mcp_allowed_hosts",
    "mcp_allowed_origins",
    "email_settings",
    "database_connection_pool_limit",
    "database_connection_timeout",
    "database_connect_timeout",
    "database_socket_timeout",
    "database_max_idle_connection_lifetime",
    "database_connection_pool_timeout",
    "database_statement_timeout",
    "database_lock_timeout",
    "database_idle_in_transaction_session_timeout",
    "database_disable_prepared_statements",
    "maximum_spend_logs_retention_period",
    "maximum_autorouter_session_retention_period",
    "maximum_health_check_retention_period",
    "maximum_daily_tag_spend_retention_period",
    "maximum_spend_logs_cleanup_cron",
    "maximum_spend_logs_retention_interval",
    "maximum_spend_logs_size",
    "maximum_spend_logs_cleanup_target_size",
];

/// The last read of the database.
#[derive(Debug)]
enum Held {
    /// The native app has no database to read, so only the config and the variable say whether
    /// the proxy serves from one.
    NoDatabase,
    /// The database was never read. A later read that fails keeps the last one that did not.
    /// Unread, only the config and the variable can switch `store_model_in_db` on, as for a proxy
    /// that could not read its database when it started.
    Unread,
    Read(StoredConfig),
}

/// What the proxy's database holds, as of its last read.
pub(super) struct Stored {
    /// What turns `store_model_in_db` on without the database: the variable or the config key.
    switched: Option<&'static str>,
    /// The variables `environment_variables` in the config sets, which the proxy reads over a
    /// stored one of the same name.
    owned_variables: BTreeSet<String>,
    held: Arc<ArcSwap<Held>>,
}

impl Stored {
    /// What `config` and its `environment` switch on, with the database `reader` read once now
    /// and again every `proxy_config_reload_interval_seconds` for as long as this lives.
    pub(super) async fn read(
        config: &Config,
        environment: &dyn Lookup,
        reader: Option<Arc<StoredConfigRepository>>,
    ) -> Self {
        let held = match &reader {
            None => Held::NoDatabase,
            Some(reader) => read_once(reader).await.unwrap_or(Held::Unread),
        };
        let stored = Self {
            held: Arc::new(ArcSwap::from_pointee(held)),
            ..Self::switched_by(config, environment)
        };
        if let Some(reader) = reader {
            reread(
                reader,
                Arc::downgrade(&stored.held),
                interval(config.general_settings.proxy_config_reload_interval_seconds),
            );
        }
        stored
    }

    fn switched_by(config: &Config, environment: &dyn Lookup) -> Self {
        let by_variable = environment
            .get(STORE_MODEL_IN_DB)
            .and_then(|value| str_to_bool(&value))
            == Some(true);
        let by_config = config.general_settings.store_model_in_db == Some(true);
        Self {
            switched: match (by_variable, by_config) {
                (true, _) => Some(STORE_MODEL_IN_DB),
                (false, true) => Some("general_settings.store_model_in_db"),
                (false, false) => None,
            },
            owned_variables: config.environment_variables.keys().cloned().collect(),
            held: Arc::new(ArcSwap::from_pointee(Held::NoDatabase)),
        }
    }

    /// Switched off, with no database.
    pub(super) fn none() -> Self {
        Self::switched_by(&Config::default(), &|_: &str| None::<String>)
    }

    #[cfg(test)]
    fn holding(config: &Config, stored: Option<StoredConfig>) -> Self {
        let held = stored.map_or(Held::Unread, Held::Read);
        Self {
            held: Arc::new(ArcSwap::from_pointee(held)),
            ..Self::switched_by(config, &|_: &str| None::<String>)
        }
    }

    /// What the database holds that reaches every request, or the switch when the native app
    /// cannot read the database the proxy serves from.
    pub(super) fn settings(&self) -> Option<Box<str>> {
        match &**self.held.load() {
            Held::NoDatabase => self.switched.map(Box::from),
            Held::Unread => self.switched.map(|_| "database_unread".into()),
            Held::Read(stored) if self.serves(stored) => self.reaching_every_request(stored),
            Held::Read(_) => None,
        }
    }

    /// The database model that joins `group`, by its name or as a wildcard that matches it.
    pub(super) fn model(&self, group: &str) -> Option<Box<str>> {
        match &**self.held.load() {
            Held::Read(stored) if self.serves(stored) => stored
                .models
                .iter()
                .find(|name| {
                    *name == group
                        || (name.contains('*') && litellm_fnmatch::fnmatchcase(group, name))
                })
                .map(|name| format!("database model_list {name}").into()),
            _ => None,
        }
    }

    fn serves(&self, stored: &StoredConfig) -> bool {
        self.switched.is_some() || stored.sections.iter().any(|row| row.store_model_in_db)
    }

    fn reaching_every_request(&self, stored: &StoredConfig) -> Option<Box<str>> {
        stored
            .sections
            .iter()
            .find_map(|row| self.stored_key(row))
            .or_else(|| {
                stored
                    .default_on_guardrails
                    .first()
                    .map(|name| format!("database guardrails {name}").into())
            })
            .or_else(|| {
                stored
                    .attached_policies
                    .first()
                    .map(|name| format!("database policy_attachments {name}").into())
            })
    }

    /// The first key of `row` the proxy reads from the database and a request reaches.
    fn stored_key(&self, row: &StoredSection) -> Option<Box<str>> {
        let section = row.section.as_str();
        row.keys
            .iter()
            .find(|key| match section {
                "general_settings" => !INERT_GENERAL_SETTINGS.contains(&key.as_str()),
                ENVIRONMENT_VARIABLES => !self.owned_variables.contains(*key),
                _ => true,
            })
            .map(|key| format!("database {section}.{key}").into())
    }
}

/// Python reloads on its default when the interval is not positive.
fn interval(seconds: u64) -> Duration {
    const DEFAULT: u64 = 30;
    Duration::from_secs(if seconds == 0 { DEFAULT } else { seconds })
}

async fn read_once(reader: &StoredConfigRepository) -> Option<Held> {
    match reader.read().await {
        Ok(stored) => Some(Held::Read(stored)),
        Err(error) => {
            tracing::warn!(
                error = %litellm_error::Report::new(&error),
                "could not read what the proxy serves from its database, so the gateway keeps its last read"
            );
            None
        }
    }
}

/// Reads the database every `every` into `held` until the admit holding it is dropped.
fn reread(reader: Arc<StoredConfigRepository>, held: Weak<ArcSwap<Held>>, every: Duration) {
    tokio::spawn(async move {
        let mut ticks = tokio::time::interval(every);
        ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticks.tick().await;
        loop {
            ticks.tick().await;
            let Some(read) = read_once(&reader).await else {
                continue;
            };
            let Some(held) = held.upgrade() else {
                return;
            };
            held.store(Arc::new(read));
        }
    });
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn section(section: &str, keys: &[&str], store_model_in_db: bool) -> StoredSection {
        StoredSection {
            section: section.to_owned(),
            keys: keys.iter().map(|key| (*key).to_owned()).collect(),
            store_model_in_db,
        }
    }

    fn switched_on() -> StoredSection {
        section("general_settings", &["store_model_in_db"], true)
    }

    const CONFIG: &str =
        "model_list: []\nenvironment_variables:\n  OPENAI_API_KEY: os.environ/OPENAI_API_KEY\n";

    fn stored(stored: Option<StoredConfig>) -> Stored {
        Stored::holding(&Config::from_yaml(CONFIG).unwrap(), stored)
    }

    #[rstest]
    #[case::unread(None, None)]
    #[case::switched_off_in_the_database(
        Some(StoredConfig {
            sections: vec![section("router_settings", &["fallbacks"], false)],
            models: vec!["claude".to_owned()],
            ..StoredConfig::default()
        }),
        None
    )]
    #[case::switched_on_with_nothing_more(
        Some(StoredConfig { sections: vec![switched_on()], ..StoredConfig::default() }),
        None
    )]
    #[case::with_models_alone(
        Some(StoredConfig {
            sections: vec![switched_on()],
            models: vec!["claude".to_owned()],
            ..StoredConfig::default()
        }),
        None
    )]
    #[case::with_inert_general_settings(
        Some(StoredConfig {
            sections: vec![section(
                "general_settings",
                &["database_connection_pool_limit", "maximum_spend_logs_retention_period", "store_model_in_db"],
                true,
            )],
            ..StoredConfig::default()
        }),
        None
    )]
    #[case::with_a_general_setting(
        Some(StoredConfig {
            sections: vec![section("general_settings", &["alerting", "store_model_in_db"], true)],
            ..StoredConfig::default()
        }),
        Some("database general_settings.alerting")
    )]
    #[case::with_a_router_setting(
        Some(StoredConfig {
            sections: vec![switched_on(), section("router_settings", &["fallbacks"], false)],
            ..StoredConfig::default()
        }),
        Some("database router_settings.fallbacks")
    )]
    #[case::with_a_variable_the_config_sets(
        Some(StoredConfig {
            sections: vec![switched_on(), section("environment_variables", &["OPENAI_API_KEY"], false)],
            ..StoredConfig::default()
        }),
        None
    )]
    #[case::with_a_variable_of_its_own(
        Some(StoredConfig {
            sections: vec![switched_on(), section("environment_variables", &["ANTHROPIC_API_KEY"], false)],
            ..StoredConfig::default()
        }),
        Some("database environment_variables.ANTHROPIC_API_KEY")
    )]
    #[case::with_a_default_on_guardrail(
        Some(StoredConfig {
            sections: vec![switched_on()],
            default_on_guardrails: vec!["pii".to_owned()],
            ..StoredConfig::default()
        }),
        Some("database guardrails pii")
    )]
    #[case::with_a_policy_attachment(
        Some(StoredConfig {
            sections: vec![switched_on()],
            attached_policies: vec!["audit".to_owned()],
            ..StoredConfig::default()
        }),
        Some("database policy_attachments audit")
    )]
    fn every_request_goes_to_python_over_what_the_database_holds_for_it(
        #[case] held: Option<StoredConfig>,
        #[case] over: Option<&str>,
    ) {
        assert_eq!(stored(held).settings().as_deref(), over);
    }

    #[rstest]
    #[case::joined_by_name("claude", Some("database model_list claude"))]
    #[case::joined_by_a_wildcard("gpt-5.5", Some("database model_list gpt-*"))]
    #[case::of_the_config_alone("haiku", None)]
    fn a_group_goes_to_python_when_a_database_model_joins_it(
        #[case] group: &str,
        #[case] over: Option<&str>,
    ) {
        let held = stored(Some(StoredConfig {
            sections: vec![switched_on()],
            models: vec!["claude".to_owned(), "gpt-*".to_owned()],
            ..StoredConfig::default()
        }));

        assert_eq!(held.model(group).as_deref(), over);
    }

    #[rstest]
    fn an_unread_database_sends_every_request_to_python_when_the_config_switches_it_on() {
        let mut config = Config::default();
        config.general_settings.store_model_in_db = Some(true);

        assert_eq!(
            Stored::holding(&config, None).settings().as_deref(),
            Some("database_unread")
        );
    }

    #[rstest]
    #[case::by_the_variable(Some("True"), None, Some(STORE_MODEL_IN_DB))]
    #[case::not_by_a_false_variable(Some("False"), None, None)]
    #[case::by_the_config(None, Some(true), Some("general_settings.store_model_in_db"))]
    #[case::by_neither(None, None, None)]
    fn without_a_database_the_switch_alone_sends_every_request_to_python(
        #[case] variable: Option<&'static str>,
        #[case] in_config: Option<bool>,
        #[case] over: Option<&str>,
    ) {
        let mut config = Config::default();
        config.general_settings.store_model_in_db = in_config;
        let environment =
            move |name: &str| (name == STORE_MODEL_IN_DB).then_some(variable?.to_owned());

        let stored = Stored::switched_by(&config, &environment);

        assert_eq!(stored.settings().as_deref(), over);
    }
}
