//! The config the proxy stores in its database, read at boot and layered into the config file the
//! way the Python proxy layers it when `store_model_in_db` is on: the `LiteLLM_Config` settings
//! rows under the file before any `os.environ/` reference is read, then the
//! `LiteLLM_ProxyModelTable` rows added to its models with the `LiteLLM_CredentialsTable`
//! credentials filled in, each row reading only the references Python reads in it. The poller in
//! `reload.rs` reads and layers it again the same way.
//!
//! The runtime flags the admin UI saves in `LiteLLM_UISettings` are layered under the file whenever
//! the gateway has a database, `store_model_in_db` or not, since the proxy syncs them at startup
//! whenever it has one (`sync_ui_settings_to_general_settings`). The poller reads them again only
//! while `store_model_in_db` is on, as the proxy does on each reload.
//!
//! Every stored string is decrypted the way Python decrypts it before it is layered in: each
//! top-level string of a model's `litellm_params` (`_resolve_db_litellm_param`), each credential
//! value (`decrypted_or_stored`) and each stored variable (`decrypt_and_set_db_env_variables`). A
//! value that does not decrypt is kept as stored, since Python stores plain text beside
//! ciphertexts.
//!
//! Boot never fails over the database. When `store_model_in_db` is on and the stored config
//! cannot be read, the native app runs the file alone and [`StoredConfig::Unread`] has `admit`
//! refuse every request until a later read loads it.

use std::{collections::BTreeMap, path::Path, sync::Arc};

use litellm_at_rest::AtRestKey;
use litellm_config::{
    Config, Object, Value,
    db::{
        models::{DbCredential, DbModel, MissingCredential},
        settings::{Section, SettingsRows, UiSettings},
        switches::{DbObject, store_model_in_db},
    },
};
use litellm_config_load::Document;
use litellm_env::Lookup;
use litellm_error::Report;
use litellm_repositories::{
    ConfigParam, ConfigRepository, Credential, CredentialRepository, Database, ProxyModel,
    ProxyModelRepository, RawJson, UiSettingsRepository,
};
use litellm_secret_value::SecretValue;
use litellm_secrets::SecretResolver;

use crate::{Error, Resources, reload::Source};

const SALT_KEY: &str = "LITELLM_SALT_KEY";
const MASTER_KEY: &str = "LITELLM_MASTER_KEY";

/// What reads every `os.environ/` reference of the config, made fresh for each resolve.
pub(crate) type Secrets = dyn Fn() -> SecretResolver + Send + Sync;

/// Whether the config the proxy stores in its database is part of the config the native app runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoredConfig {
    /// `store_model_in_db` is off, so the proxy reads no config from its database.
    Unused,
    /// It is on and the database was read, so the config holds what it stores.
    Loaded,
    /// It is on and the database could not be read, so the native app runs the file alone and
    /// serves nothing.
    Unread,
}

/// A config the native app is built from, and whether it holds the config the proxy stores in its
/// database.
pub struct Loaded {
    pub config: Config,
    pub stored: StoredConfig,
    /// Where the poller reads the stored config again, `None` when `store_model_in_db` is off or
    /// the gateway has no database to read it from.
    pub source: Option<Source>,
}

/// The stored config one read found, every string decrypted.
#[derive(Clone, Default)]
pub(crate) struct Fetched {
    pub(crate) settings: SettingsRows,
    pub(crate) models: Vec<DbModel>,
    pub(crate) credentials: Vec<DbCredential>,
}

/// The rows the stored config is read from.
pub(crate) trait StoredRows {
    async fn settings(&self) -> Result<Vec<ConfigParam>, litellm_db::Error>;
    async fn models(&self) -> Result<Vec<ProxyModel>, litellm_db::Error>;
    async fn credentials(&self) -> Result<Vec<Credential>, litellm_db::Error>;
    async fn ui_settings(&self) -> Result<Option<RawJson>, litellm_db::Error>;
}

impl StoredRows for Database {
    async fn settings(&self) -> Result<Vec<ConfigParam>, litellm_db::Error> {
        let sections: Vec<String> = <Section as strum::VariantArray>::VARIANTS
            .iter()
            .map(|section| <&str>::from(*section).to_owned())
            .collect();
        ConfigRepository::new(self.clone())
            .find_many(&sections)
            .await
    }

    async fn models(&self) -> Result<Vec<ProxyModel>, litellm_db::Error> {
        ProxyModelRepository::new(self.clone()).all().await
    }

    async fn credentials(&self) -> Result<Vec<Credential>, litellm_db::Error> {
        CredentialRepository::new(self.clone()).all().await
    }

    async fn ui_settings(&self) -> Result<Option<RawJson>, litellm_db::Error> {
        UiSettingsRepository::new(self.clone()).find().await
    }
}

/// The config at `path` with the config the proxy stores in its database layered in when
/// `store_model_in_db` is on. Only a config file that does not load fails, never the database.
/// When it is on, the database pool boot read it with is kept in the [`Source`] the poller reads
/// it again from.
pub(crate) async fn load(path: &Path, secrets: Arc<Secrets>) -> Result<Loaded, Error> {
    let document = Document::read(path)?;
    let file = document.clone().resolve(secrets()).await?;
    let resources = Resources::build(&file)?;
    let environment = resources.environment.clone();
    let Some((pools, database)) = crate::db::connect(&file, &resources.secrets).await? else {
        let on = store_model_in_db(
            file.general_settings.store_model_in_db,
            &*environment,
            &SettingsRows::default(),
        );
        if on {
            tracing::warn!(
                "store_model_in_db is on and the gateway has no database to read it from, so python answers every request"
            );
        }
        return Ok(without_stored(file, on));
    };
    let key = salt_key(&*environment, &file);
    let (loaded, fetched) = layered(
        document.clone(),
        file,
        &*environment,
        &database,
        key.as_ref(),
        &*secrets,
    )
    .await;
    if loaded.stored == StoredConfig::Unused {
        pools.close().await;
        return Ok(loaded);
    }
    let source = Source::new(document, key, secrets, database, pools, fetched);
    Ok(Loaded {
        source: Some(source),
        ..loaded
    })
}

/// `document` with the config stored in `rows` layered in, and what the read found when it was.
/// When `store_model_in_db` is off only the UI settings are, and when there are none it is `file`,
/// the same document resolved alone. A UI settings row that cannot be read leaves every request to
/// Python, since one of its flags may change how the proxy answers. `environment` is the file's
/// `environment_variables` in front of the process environment.
pub(crate) async fn layered(
    document: Document,
    file: Config,
    environment: &dyn Lookup,
    rows: &impl StoredRows,
    key: Option<&AtRestKey>,
    secrets: &Secrets,
) -> (Loaded, Option<Fetched>) {
    let switched_on = file.general_settings.store_model_in_db;
    let ui_settings = match rows.ui_settings().await {
        Ok(column) => column.map(|column| UiSettings::from_json(column.get())),
        Err(error) => {
            tracing::warn!(
                error = %Report::new(&error),
                "could not read the UI settings the proxy stores in its database"
            );
            return (without_stored(file, true), None);
        }
    };
    let settings = match rows.settings().await {
        Ok(found) => SettingsRows {
            ui_settings: ui_settings.clone().unwrap_or_default(),
            ..settings_rows(&found, key)
        },
        Err(error) => {
            let on = store_model_in_db(switched_on, environment, &SettingsRows::default());
            tracing::warn!(
                error = %Report::new(&error),
                store_model_in_db = on,
                "could not read the settings the proxy stores in its database"
            );
            if on {
                return (without_stored(file, true), None);
            }
            return (
                with_ui_settings(document, file, ui_settings, secrets).await,
                None,
            );
        }
    };
    if !store_model_in_db(switched_on, environment, &settings) {
        return (
            with_ui_settings(document, file, ui_settings, secrets).await,
            None,
        );
    }
    let fetched = match fetch(&document, settings, rows, key).await {
        Ok(fetched) => fetched,
        Err(error) => {
            tracing::warn!(
                error = %Report::new(&error),
                "could not read the models the proxy stores in its database, so python answers every request until a later read loads them"
            );
            return (without_stored(file, true), None);
        }
    };
    match build(document, &fetched, secrets).await {
        Ok(config) => (
            Loaded {
                config,
                stored: StoredConfig::Loaded,
                source: None,
            },
            Some(fetched),
        ),
        Err(error) => {
            tracing::warn!(
                error = %Report::new(&error),
                "the config with the stored config layered in does not load, so python answers every request until a later read loads it"
            );
            (without_stored(file, true), None)
        }
    }
}

/// `document` with only the UI settings layered in, for a proxy that reads no other stored
/// config, or `file` when the UI never saved settings.
async fn with_ui_settings(
    document: Document,
    file: Config,
    ui_settings: Option<UiSettings>,
    secrets: &Secrets,
) -> Loaded {
    let Some(ui_settings) = ui_settings else {
        return without_stored(file, false);
    };
    let rows = SettingsRows {
        ui_settings,
        ..SettingsRows::default()
    };
    let (document, _) = document.overlay(&rows);
    match document.resolve(secrets()).await {
        Ok(config) => Loaded {
            config,
            stored: StoredConfig::Unused,
            source: None,
        },
        Err(error) => {
            tracing::warn!(
                error = %Report::new(&error),
                "the config with the stored UI settings layered in does not load, so python answers every request"
            );
            without_stored(file, true)
        }
    }
}

/// `file` alone, which leaves every request to Python when `on`.
fn without_stored(file: Config, on: bool) -> Loaded {
    Loaded {
        config: file,
        stored: match on {
            true => StoredConfig::Unread,
            false => StoredConfig::Unused,
        },
        source: None,
    }
}

/// The stored models and credentials beside `settings`. The models are read only when
/// `supported_db_objects`, with the stored settings under the file, loads them.
pub(crate) async fn fetch(
    document: &Document,
    settings: SettingsRows,
    rows: &impl StoredRows,
    key: Option<&AtRestKey>,
) -> Result<Fetched, litellm_db::Error> {
    let (overlaid, _) = document.clone().overlay(&settings);
    let models: Vec<DbModel> = match overlaid.supported_db_objects().loads(DbObject::Models) {
        true => rows
            .models()
            .await?
            .into_iter()
            .filter_map(|row| stored_model(row, key))
            .collect(),
        false => Vec::new(),
    };
    let credentials: Vec<DbCredential> = rows
        .credentials()
        .await?
        .into_iter()
        .filter_map(|row| stored_credential(row, key))
        .collect();
    Ok(Fetched {
        settings,
        models,
        credentials,
    })
}

/// The config `document` describes with `fetched` layered in.
pub(crate) async fn build(
    document: Document,
    fetched: &Fetched,
    secrets: &Secrets,
) -> Result<Config, litellm_config_load::Error> {
    let (document, shadowed) = document.overlay(&fetched.settings);
    for shadowed in &shadowed {
        tracing::warn!(
            section = <&str>::from(shadowed.section),
            key = shadowed.key,
            "the config file sets this key, so the value the database stores for it is ignored"
        );
    }
    let (config, missing) = without_unloadable_models(
        document,
        fetched.models.clone(),
        fetched.credentials.clone(),
        secrets,
    )
    .await?;
    for missing in &missing {
        tracing::warn!(
            model_name = missing.model_name,
            credential_name = missing.credential_name,
            "a model names a credential neither the database nor the config file holds, so it is called without it"
        );
    }
    tracing::info!(
        models = config.model_list.len(),
        "loaded the config the proxy stores in its database"
    );
    Ok(config)
}

/// The config with every stored model in it, or, when that does not load, with the stored models
/// that load beside the file alone. Python skips a stored model whose params do not validate, so
/// one bad row does not keep every other model from being served.
async fn without_unloadable_models(
    document: Document,
    models: Vec<DbModel>,
    credentials: Vec<DbCredential>,
    secrets: &Secrets,
) -> Result<(Config, Box<[MissingCredential]>), litellm_config_load::Error> {
    let whole = merged(
        document.clone(),
        models.clone(),
        credentials.clone(),
        secrets,
    )
    .await;
    let Err(error) = whole else {
        return whole;
    };
    let stored = models.len();
    let mut loadable = Vec::with_capacity(stored);
    for model in models {
        let alone = merged(
            document.clone(),
            [model.clone()],
            credentials.clone(),
            secrets,
        )
        .await;
        match alone {
            Ok(_) => loadable.push(model),
            Err(error) => tracing::warn!(
                model_id = model.model_id,
                model_name = model.model_name,
                error = %Report::new(&error),
                "skipped a model stored in the database that does not load"
            ),
        }
    }
    if loadable.len() == stored {
        return Err(error);
    }
    merged(document, loadable, credentials, secrets).await
}

async fn merged(
    document: Document,
    models: impl IntoIterator<Item = DbModel>,
    credentials: impl IntoIterator<Item = DbCredential>,
    secrets: &Secrets,
) -> Result<(Config, Box<[MissingCredential]>), litellm_config_load::Error> {
    document
        .resolve_with_stored(secrets(), models, credentials)
        .await
}

/// `get_salt_key`: `LITELLM_SALT_KEY`, else the master key the proxy runs with, which is the
/// file's `general_settings.master_key` with its `os.environ/` reference read, else
/// `LITELLM_MASTER_KEY`. Without either Python decrypts nothing.
fn salt_key(environment: &dyn Lookup, file: &Config) -> Option<AtRestKey> {
    environment
        .get(SALT_KEY)
        .map(SecretValue::new)
        .or_else(|| file.general_settings.master_key.clone())
        .or_else(|| environment.get(MASTER_KEY).map(SecretValue::new))
        .map(|salt_key| AtRestKey::from_salt_key(&salt_key))
}

/// `value` decrypted when it is a string that decrypts under `key`, and as stored otherwise.
fn decrypted(value: Value, key: Option<&AtRestKey>) -> Value {
    match (value, key) {
        (Value::String(stored), Some(key)) => {
            Value::String(key.decrypt_or_original(&stored).into_value().into_owned())
        }
        (value, _) => value,
    }
}

/// A JSON column read as an object, `None` when it holds anything else.
fn object(column: &RawJson) -> Option<BTreeMap<String, Value>> {
    serde_json::from_str(column.get()).ok()
}

fn decrypted_object(column: &RawJson, key: Option<&AtRestKey>) -> Option<Object> {
    Some(
        object(column)?
            .into_iter()
            .map(|(name, value)| (name, decrypted(value, key)))
            .collect(),
    )
}

/// The settings rows by section. A row that is `null` or not an object is an empty section, and
/// a stored variable that is not a string is left out, as `os.environ` takes only strings.
pub(crate) fn settings_rows(rows: &[ConfigParam], key: Option<&AtRestKey>) -> SettingsRows {
    let section = |section: Section| {
        rows.iter()
            .find(|row| row.param_name == <&str>::from(section))
            .and_then(|row| row.param_value.as_ref())
            .and_then(object)
            .unwrap_or_default()
    };
    SettingsRows {
        general_settings: section(Section::GeneralSettings).into_iter().collect(),
        router_settings: section(Section::RouterSettings).into_iter().collect(),
        litellm_settings: section(Section::LitellmSettings).into_iter().collect(),
        environment_variables: section(Section::EnvironmentVariables)
            .into_iter()
            .filter_map(|(name, value)| match decrypted(value, key) {
                Value::String(text) => Some((name, SecretValue::new(text))),
                _ => None,
            })
            .collect(),
        ui_settings: UiSettings::default(),
    }
}

/// The row with its params decrypted. Python skips a row whose `litellm_params` is not an object.
fn stored_model(row: ProxyModel, key: Option<&AtRestKey>) -> Option<DbModel> {
    let Some(litellm_params) = decrypted_object(&row.litellm_params, key) else {
        tracing::warn!(
            model_id = row.model_id,
            model_name = row.model_name,
            "skipped a model stored in the database whose litellm_params is not an object"
        );
        return None;
    };
    Some(DbModel {
        model_id: row.model_id,
        model_name: row.model_name,
        litellm_params,
        model_info: row
            .model_info
            .as_ref()
            .and_then(object)
            .unwrap_or_default()
            .into_iter()
            .collect(),
        blocked: row.blocked,
    })
}

/// The row with its values decrypted. Python drops every stored credential when one row does not
/// validate, and only that row is skipped here.
fn stored_credential(row: Credential, key: Option<&AtRestKey>) -> Option<DbCredential> {
    let Some(credential_values) = decrypted_object(&row.credential_values, key) else {
        tracing::warn!(
            credential_name = row.credential_name,
            "skipped a credential stored in the database whose credential_values is not an object"
        );
        return None;
    };
    Some(DbCredential {
        credential_name: row.credential_name,
        credential_values,
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use litellm_at_rest::Algorithm;
    use rstest::rstest;

    use super::*;
    use crate::test_support::{Rows, Variables, credential, environment, model, secrets, setting};

    const MASTER_KEY: &str = "sk-master";
    const FILE: &str = "model_list:\n  - model_name: file\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n";
    const ON: &str = "general_settings:\n  store_model_in_db: true\n";
    const STORED: &str = r#"{"model": "anthropic/claude-haiku-5-5", "api_key": "stored-key"}"#;

    fn encrypted(salt_key: &str, plaintext: &str) -> String {
        AtRestKey::from_salt_key(&SecretValue::new(salt_key.to_owned()))
            .encrypt(plaintext, Algorithm::XSalsa20Poly1305)
            .unwrap()
    }

    fn params_with_api_key(api_key: &str) -> String {
        format!(r#"{{"model": "anthropic/claude", "api_key": "{api_key}"}}"#)
    }

    /// The config file `yaml` with `rows` layered in, with `variables` as the process environment.
    async fn layered_over(yaml: &str, rows: &Rows, variables: Variables) -> Loaded {
        let mut file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
        let document = Document::read(file.path()).unwrap();
        let config = document
            .clone()
            .resolve(secrets(variables)())
            .await
            .unwrap();
        let key = salt_key(&environment(variables), &config);
        layered(
            document,
            config,
            &environment(variables),
            rows,
            key.as_ref(),
            &secrets(variables),
        )
        .await
        .0
    }

    fn served(loaded: &Loaded) -> (StoredConfig, Vec<&str>) {
        (
            loaded.stored,
            loaded
                .config
                .model_list
                .iter()
                .map(|model| model.model_name.as_str())
                .collect(),
        )
    }

    fn rows_with(general_settings: Option<&str>, models: bool, credentials: bool) -> Rows {
        Rows {
            settings: Some(
                general_settings
                    .map(|value| setting("general_settings", value))
                    .into_iter()
                    .collect(),
            ),
            models: models.then(|| vec![model("stored", STORED)]),
            credentials: credentials.then(Vec::new),
            ui_settings: Ok(None),
        }
    }

    #[rstest]
    #[case::off_everywhere(FILE, rows_with(None, true, true), (StoredConfig::Unused, vec!["file"]))]
    #[case::on_in_the_file(
        &format!("{FILE}{ON}"),
        rows_with(None, true, true),
        (StoredConfig::Loaded, vec!["file", "stored"])
    )]
    #[case::on_in_the_stored_row(
        FILE,
        rows_with(Some(r#"{"store_model_in_db": true}"#), true, true),
        (StoredConfig::Loaded, vec!["file", "stored"])
    )]
    #[case::settings_unreadable_while_off(
        FILE,
        Rows::default(),
        (StoredConfig::Unused, vec!["file"])
    )]
    #[case::settings_unreadable_while_on(
        &format!("{FILE}{ON}"),
        Rows::default(),
        (StoredConfig::Unread, vec!["file"])
    )]
    #[case::models_unreadable(
        &format!("{FILE}{ON}"),
        rows_with(None, false, true),
        (StoredConfig::Unread, vec!["file"])
    )]
    #[case::credentials_unreadable(
        &format!("{FILE}{ON}"),
        rows_with(None, true, false),
        (StoredConfig::Unread, vec!["file"])
    )]
    #[case::models_left_out_by_supported_db_objects(
        &format!("{FILE}{ON}  supported_db_objects: [mcp]\n"),
        rows_with(None, false, true),
        (StoredConfig::Loaded, vec!["file"])
    )]
    #[tokio::test]
    async fn the_stored_config_is_layered_in_only_when_store_model_in_db_is_on_and_it_reads(
        #[case] yaml: &str,
        #[case] rows: Rows,
        #[case] expected: (StoredConfig, Vec<&str>),
    ) {
        let loaded = layered_over(yaml, &rows, &[]).await;

        assert_eq!(served(&loaded), expected);
    }

    #[rstest]
    #[case::under_the_master_key(
        &[],
        params_with_api_key(&encrypted(MASTER_KEY, "decrypted-key")),
        Some("decrypted-key")
    )]
    #[case::under_the_salt_key(
        &[("LITELLM_SALT_KEY", "salt")],
        params_with_api_key(&encrypted("salt", "decrypted-key")),
        Some("decrypted-key")
    )]
    #[case::under_another_key_kept_as_stored(
        &[("LITELLM_SALT_KEY", "salt")],
        params_with_api_key(&encrypted(MASTER_KEY, "same")),
        None
    )]
    #[case::an_encrypted_reference_read_from_the_environment(
        &[("PROVIDER_KEY", "from-environment")],
        params_with_api_key(&encrypted(MASTER_KEY, "os.environ/PROVIDER_KEY")),
        Some("from-environment")
    )]
    #[case::a_reference_to_an_encrypted_stored_variable(
        &[],
        params_with_api_key("os.environ/STORED_KEY"),
        Some("from-stored-variable")
    )]
    #[case::an_encrypted_credential(
        &[],
        r#"{"model": "anthropic/claude", "litellm_credential_name": "stored-credential"}"#.to_owned(),
        Some("from-credential")
    )]
    #[tokio::test]
    async fn every_stored_string_is_decrypted_with_the_salt_key_before_its_reference_is_read(
        #[case] variables: Variables,
        #[case] litellm_params: String,
        #[case] api_key: Option<&str>,
    ) {
        let rows = Rows {
            settings: Some(vec![
                setting("general_settings", r#"{"store_model_in_db": true}"#),
                setting(
                    "environment_variables",
                    &format!(
                        r#"{{"STORED_KEY": "{}"}}"#,
                        encrypted(MASTER_KEY, "from-stored-variable")
                    ),
                ),
            ]),
            models: Some(vec![model("stored", &litellm_params)]),
            credentials: Some(vec![credential(
                "stored-credential",
                &format!(
                    r#"{{"api_key": "{}"}}"#,
                    encrypted(MASTER_KEY, "from-credential")
                ),
            )]),
            ui_settings: Ok(None),
        };
        let file = format!("model_list: []\ngeneral_settings:\n  master_key: {MASTER_KEY}\n");

        let loaded = layered_over(&file, &rows, variables).await;

        let stored = loaded.config.model_list[0]
            .litellm_params
            .api_key
            .as_ref()
            .map(|key| key.expose().to_owned());
        let written: BTreeMap<String, String> = serde_json::from_str(&litellm_params).unwrap();
        let expected = api_key.or(written.get("api_key").map(String::as_str));
        assert_eq!(stored.as_deref(), expected);
    }

    #[rstest]
    #[case::params_that_are_not_an_object(r#""anthropic/claude""#)]
    #[case::params_that_do_not_load(r#"{"model": ["not", "text"]}"#)]
    #[tokio::test]
    async fn a_stored_model_that_does_not_load_is_skipped_and_the_others_are_served(
        #[case] litellm_params: &str,
    ) {
        let rows = Rows {
            models: Some(vec![model("bad", litellm_params), model("stored", STORED)]),
            ..rows_with(None, true, true)
        };

        let loaded = layered_over(&format!("{FILE}{ON}"), &rows, &[]).await;

        assert_eq!(
            served(&loaded),
            (StoredConfig::Loaded, vec!["file", "stored"])
        );
    }

    const FORWARD_HEADERS: &str = r#"{"forward_client_headers_to_llm_api": true}"#;

    #[rstest]
    #[case::store_model_in_db_on(
        &format!("{FILE}{ON}"),
        Ok(Some(FORWARD_HEADERS)),
        StoredConfig::Loaded,
        Some(true)
    )]
    #[case::store_model_in_db_off(
        FILE,
        Ok(Some(FORWARD_HEADERS)),
        StoredConfig::Unused,
        Some(true)
    )]
    #[case::saved_as_json_text(
        FILE,
        Ok(Some(r#""{\"forward_client_headers_to_llm_api\": true}""#)),
        StoredConfig::Unused,
        Some(true)
    )]
    #[case::the_file_wins(
        "general_settings:\n  forward_client_headers_to_llm_api: false\n",
        Ok(Some(FORWARD_HEADERS)),
        StoredConfig::Unused,
        Some(false)
    )]
    #[case::no_row(FILE, Ok(None), StoredConfig::Unused, None)]
    #[case::unreadable(FILE, Err(()), StoredConfig::Unread, None)]
    #[tokio::test]
    async fn the_ui_settings_flags_are_layered_under_the_file(
        #[case] yaml: &str,
        #[case] ui_settings: Result<Option<&'static str>, ()>,
        #[case] stored: StoredConfig,
        #[case] forward: Option<bool>,
    ) {
        let rows = Rows {
            ui_settings,
            ..rows_with(None, true, true)
        };

        let loaded = layered_over(yaml, &rows, &[]).await;

        assert_eq!(
            (
                loaded.stored,
                loaded
                    .config
                    .general_settings
                    .forward_client_headers_to_llm_api
            ),
            (stored, forward)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_ui_settings_flag_python_honors_sends_requests_to_python() {
        let rows = Rows {
            ui_settings: Ok(Some(FORWARD_HEADERS)),
            ..rows_with(None, true, true)
        };
        let loaded = layered_over(FILE, &rows, &[]).await;
        let admit = std::sync::Arc::new(crate::admit::Admit::over(&loaded.config, vec![]));
        let request = axum::extract::Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .body(axum::body::Body::from(r#"{"model": "file"}"#))
            .unwrap();

        let refused = admit
            .admit(request)
            .await
            .err()
            .map(|(_, reason)| reason.over);

        assert_eq!(
            refused.as_deref(),
            Some("general_settings.forward_client_headers_to_llm_api")
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_model_its_credentials_oidc_key_may_not_reach_is_left_to_python() {
        let rows = Rows {
            models: Some(vec![model(
                "bound",
                r#"{"model": "anthropic/claude", "litellm_credential_name": "shared", "api_base": "https://elsewhere.example"}"#,
            )]),
            credentials: Some(vec![credential(
                "shared",
                r#"{"api_key": "oidc/env/TOKEN", "api_base": "https://bound.example"}"#,
            )]),
            ..rows_with(None, true, true)
        };
        let loaded =
            layered_over(&format!("{FILE}{ON}  master_key: sk-gateway\n"), &rows, &[]).await;
        let stored = &loaded.config.model_list[1];
        let admit = std::sync::Arc::new(crate::admit::Admit::over(&loaded.config, vec![]));
        let request = axum::extract::Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("authorization", "Bearer sk-gateway")
            .body(axum::body::Body::from(r#"{"model": "bound"}"#))
            .unwrap();

        let refused = admit
            .admit(request)
            .await
            .err()
            .map(|(_, reason)| (reason.why, reason.over));

        assert_eq!(stored.model_name, "bound");
        assert!(stored.litellm_params.api_key.is_none());
        assert_eq!(
            refused,
            Some((
                crate::admit::Refused::Model,
                "model_list.litellm_params.litellm_credential_name".into()
            ))
        );
    }
}
