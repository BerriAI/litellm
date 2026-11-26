//! What several inline test modules build the same way.

use std::sync::Arc;

use litellm_env::Lookup;
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{Caller, record::EndUserRecord};
use litellm_repositories::{ConfigParam, Credential, ProxyModel, RawJson, StoredObjects};
use litellm_secrets::{OidcResolver, SecretManagerState, SecretResolver};

use crate::{host::NativeRows, stored::StoredRows};

/// The caller the master key authenticates.
pub(crate) fn master() -> Authenticated<NativeRows> {
    Authenticated::from(Caller::master_key(EndUserRecord::default()))
}

pub(crate) type Variables = &'static [(&'static str, &'static str)];

/// What each read of the stored config returns, `None` for a read that fails.
pub(crate) struct Rows {
    pub(crate) settings: Option<Vec<ConfigParam>>,
    pub(crate) models: Option<Vec<ProxyModel>>,
    pub(crate) credentials: Option<Vec<Credential>>,
    /// The UI settings column, `Err(())` for a read that fails.
    pub(crate) ui_settings: Result<Option<&'static str>, ()>,
    /// Which other stored objects the database holds, `Err(())` for a read that fails.
    pub(crate) objects: Result<StoredObjects, ()>,
}

impl Default for Rows {
    fn default() -> Self {
        Self {
            settings: None,
            models: None,
            credentials: None,
            ui_settings: Ok(None),
            objects: Ok(StoredObjects::default()),
        }
    }
}

fn read<T: Clone>(rows: &Option<Vec<T>>) -> Result<Vec<T>, litellm_db::Error> {
    rows.clone().ok_or(litellm_db::Error::TransactionTimedOut)
}

impl StoredRows for Rows {
    async fn settings(&self) -> Result<Vec<ConfigParam>, litellm_db::Error> {
        read(&self.settings)
    }

    async fn models(&self) -> Result<Vec<ProxyModel>, litellm_db::Error> {
        read(&self.models)
    }

    async fn credentials(&self) -> Result<Vec<Credential>, litellm_db::Error> {
        read(&self.credentials)
    }

    async fn ui_settings(&self) -> Result<Option<RawJson>, litellm_db::Error> {
        self.ui_settings
            .map(|column| column.map(json))
            .map_err(|()| litellm_db::Error::TransactionTimedOut)
    }

    async fn objects(&self) -> Result<StoredObjects, litellm_db::Error> {
        self.objects
            .map_err(|()| litellm_db::Error::TransactionTimedOut)
    }
}

pub(crate) fn json(text: &str) -> RawJson {
    serde_json::from_str(text).unwrap()
}

pub(crate) fn setting(name: &str, value: &str) -> ConfigParam {
    ConfigParam {
        param_name: name.into(),
        param_value: Some(json(value)),
        reload_revision: 0,
    }
}

pub(crate) fn model(name: &str, litellm_params: &str) -> ProxyModel {
    ProxyModel {
        model_id: format!("{name}-id"),
        model_name: name.into(),
        litellm_params: json(litellm_params),
        model_info: None,
        blocked: false,
        created_at: chrono::NaiveDateTime::default(),
        created_by: "admin".into(),
        updated_at: chrono::NaiveDateTime::default(),
        updated_by: "admin".into(),
    }
}

pub(crate) fn credential(name: &str, values: &str) -> Credential {
    Credential {
        credential_name: name.into(),
        credential_values: json(values),
        credential_info: None,
    }
}

pub(crate) fn environment(variables: Variables) -> impl Lookup + Send + Sync + 'static {
    move |name: &str| {
        variables
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| (*value).to_owned())
    }
}

pub(crate) fn secrets(variables: Variables) -> impl Fn() -> SecretResolver + Send + Sync + use<> {
    move || {
        SecretResolver::new_python_compatible(
            Arc::new(SecretManagerState::default()),
            Arc::new(environment(variables)),
            OidcResolver::new(litellm_http::Client::for_test()),
        )
    }
}
