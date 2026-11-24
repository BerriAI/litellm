//! The config file as one document, with the config the proxy stores in its database layered in
//! before any `os.environ/` reference is read, which is the order `ProxyConfig.get_config` and
//! `add_deployment` leave the Python proxy's config in.

use std::path::Path;

use litellm_config::{
    Config, Value,
    db::{
        models::{self, DbCredential, DbModel, MissingCredential},
        settings::{self, SettingsRows, Shadowed},
        switches::SupportedDbObjects,
    },
};
use litellm_secrets::SecretResolver;

use crate::{ConfigEnvironment, Error, environ, environment_values, includes};

/// The config file with every file its `include` names merged in, its `os.environ/` references
/// not read yet. It holds whatever the file holds in plain text, so it has no `Debug`.
#[derive(Clone)]
pub struct Document(Value);

impl Document {
    pub fn read(path: &Path) -> Result<Self, Error> {
        includes::load(path).map(Self)
    }

    /// The typed config, with every `os.environ/` reference read through `secrets` and the
    /// config's `environment_variables` in front of the environment `secrets` reads. That same
    /// environment is put under the config, so a key the config leaves unset is read from its
    /// variable.
    pub async fn resolve(self, secrets: SecretResolver) -> Result<Config, Error> {
        let process = secrets.environment();
        let config = Config::from_value(environ::resolve(self.0, secrets).await?)?;
        let environment =
            ConfigEnvironment::new(environment_values(&config.environment_variables)?, process);
        Ok(config.with_environment(&environment))
    }

    /// The document with the stored settings rows under it, and the stored keys the file
    /// shadows.
    pub fn overlay(self, rows: &SettingsRows) -> (Self, Box<[Shadowed]>) {
        let overlaid = settings::overlay(self.0, rows);
        (Self(overlaid.document), overlaid.shadowed)
    }

    /// The keys of `section` and the value each holds, which after [`Self::overlay`] is the
    /// stored value when the file leaves the key unset.
    pub fn section(&self, section: &str) -> impl Iterator<Item = (&str, &Value)> {
        self.0
            .get(section)
            .and_then(Value::as_mapping)
            .into_iter()
            .flatten()
            .filter_map(|(key, value)| Some((key.as_str()?, value)))
    }

    /// `general_settings.supported_db_objects` as the document sets it, which after
    /// [`Self::overlay`] is the stored value when the file leaves it unset.
    pub fn supported_db_objects(&self) -> SupportedDbObjects {
        SupportedDbObjects::written(
            self.0
                .get("general_settings")
                .and_then(|general| general.get("supported_db_objects")),
        )
    }

    /// The document with the stored models added to its `model_list` and every model's named
    /// credential filled in, and the credential names neither source holds.
    pub fn merge(
        self,
        models: impl IntoIterator<Item = DbModel>,
        credentials: impl IntoIterator<Item = DbCredential>,
    ) -> Result<(Self, Box<[MissingCredential]>), Error> {
        let merged = models::merge(self.0, models, credentials)?;
        Ok((Self(merged.document), merged.missing_credentials))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use litellm_config::{Object, db::switches::DbObject};
    use litellm_secret_value::SecretValue;
    use litellm_secrets::{OidcResolver, SecretManagerState};
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    fn secrets() -> SecretResolver {
        SecretResolver::new_python_compatible(
            Arc::new(SecretManagerState::default()),
            Arc::new(|name: &str| (name == "PROCESS_KEY").then(|| "from-process".to_owned())),
            OidcResolver::new(litellm_http::Client::for_test()),
        )
    }

    fn document(yaml: &str) -> Document {
        Document(litellm_yaml::from_str(yaml).unwrap())
    }

    fn object(yaml: &str) -> Object {
        serde_yaml_ng::from_str(yaml).unwrap()
    }

    fn stored_model(params: &str) -> DbModel {
        DbModel {
            model_id: "stored-id".into(),
            model_name: "stored".into(),
            litellm_params: object(params),
            model_info: Object::default(),
            blocked: false,
        }
    }

    #[rstest]
    #[case::from_the_stored_variables("api_key: os.environ/STORED_KEY", "from-row")]
    #[case::from_the_process("api_key: os.environ/PROCESS_KEY", "from-process")]
    #[case::from_a_stored_credential("litellm_credential_name: stored-credential", "credential")]
    #[tokio::test]
    async fn a_stored_model_reads_its_references_after_the_stored_variables_are_layered_in(
        secrets: SecretResolver,
        #[case] params: &str,
        #[case] api_key: &str,
    ) {
        let rows = SettingsRows {
            environment_variables: BTreeMap::from([(
                "STORED_KEY".to_owned(),
                SecretValue::new("from-row".to_owned()),
            )]),
            ..SettingsRows::default()
        };
        let credential = DbCredential {
            credential_name: "stored-credential".into(),
            credential_values: object("api_key: credential"),
        };
        let (overlaid, _) = document("model_list: []\n").overlay(&rows);
        let (merged, missing) = overlaid
            .merge(
                [stored_model(&format!("model: anthropic/claude\n{params}"))],
                [credential],
            )
            .unwrap();

        let config = merged.resolve(secrets).await.unwrap();

        assert!(missing.is_empty());
        let key = config.model_list[0]
            .litellm_params
            .api_key
            .as_ref()
            .unwrap();
        assert_eq!(key.expose(), api_key);
    }

    #[rstest]
    fn a_section_lists_the_files_keys_with_the_stored_ones_it_leaves_unset() {
        let rows = SettingsRows {
            router_settings: object("redis_host: stored\nnum_retries: 3"),
            ..SettingsRows::default()
        };
        let (overlaid, _) = document("router_settings:\n  num_retries: 1\n").overlay(&rows);

        let section: BTreeMap<&str, &Value> = overlaid.section("router_settings").collect();

        assert_eq!(
            section,
            BTreeMap::from([
                ("num_retries", &Value::from(1)),
                ("redis_host", &Value::from("stored")),
            ])
        );
    }

    #[rstest]
    #[case::unset_everywhere("general_settings: {}\n", "{}", true)]
    #[case::stored("general_settings: {}\n", "supported_db_objects: [mcp]", false)]
    #[case::the_file_shadows_the_stored_list(
        "general_settings:\n  supported_db_objects: [models]\n",
        "supported_db_objects: [mcp]",
        true
    )]
    fn the_stored_models_load_as_the_layered_supported_db_objects_say(
        #[case] file: &str,
        #[case] stored: &str,
        #[case] loads: bool,
    ) {
        let rows = SettingsRows {
            general_settings: object(stored),
            ..SettingsRows::default()
        };

        let (overlaid, _) = document(file).overlay(&rows);

        assert_eq!(
            overlaid.supported_db_objects().loads(DbObject::Models),
            loads
        );
    }
}
