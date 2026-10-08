use litellm_auth_types::{
    ConnectionArguments, CredentialResolverHandle, Error, InputSource, SecretValue, Sourced,
    TokenProviderHandle,
    fields::{ConnectionField, azure},
};
#[cfg(test)]
use serde_json::Map;
use serde_json::Value;
use strum::EnumString;

pub const DEFAULT_AZURE_SCOPE: &str = "https://cognitiveservices.azure.com/.default";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ConfigValue<T> {
    #[default]
    Absent,
    ExplicitNone(InputSource),
    Value(Sourced<T>),
}

impl<T> ConfigValue<T> {
    pub fn as_value(&self) -> Option<&Sourced<T>> {
        match self {
            Self::Value(value) => Some(value),
            Self::Absent | Self::ExplicitNone(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, EnumString, PartialEq, Eq, Hash)]
#[allow(clippy::enum_variant_names)]
pub enum AzureCredentialType {
    ClientSecretCredential,
    ManagedIdentityCredential,
    DefaultAzureCredential,
    DeploymentIdentityCredential,
    WorkloadIdentityCredential,
}

#[derive(Clone, Debug, Default)]
pub struct AzureAuthInputs {
    pub azure_ad_token: ConfigValue<SecretValue>,
    pub azure_ad_token_provider: Option<TokenProviderHandle>,
    pub credential_resolver: Option<CredentialResolverHandle>,
    pub tenant_id: ConfigValue<String>,
    pub client_id: ConfigValue<String>,
    pub client_secret: ConfigValue<SecretValue>,
    pub azure_scope: ConfigValue<String>,
    pub azure_authority_host: ConfigValue<String>,
    pub azure_credential: ConfigValue<String>,
    pub federated_token_file: ConfigValue<String>,
    pub enable_azure_ad_token_refresh: Sourced<bool>,
}

impl AzureAuthInputs {
    pub fn default_credential_for_scope(scope: &str) -> Self {
        Self {
            azure_scope: ConfigValue::Value(Sourced::new(
                scope.to_string(),
                InputSource::Deployment,
            )),
            azure_credential: ConfigValue::Value(Sourced::new(
                "DefaultAzureCredential".to_string(),
                InputSource::Deployment,
            )),
            enable_azure_ad_token_refresh: Sourced::new(true, InputSource::Deployment),
            ..Self::default()
        }
    }

    pub fn or_configured_token_refresh(self, enabled: bool) -> Self {
        if *self.enable_azure_ad_token_refresh.value() || !enabled {
            return self;
        }
        Self {
            enable_azure_ad_token_refresh: Sourced::new(true, InputSource::Deployment),
            ..self
        }
    }

    #[cfg(test)]
    pub fn from_optional_params(params: &Map<String, Value>) -> Result<Self, Error> {
        Self::from_credentials(&ConnectionArguments::from_arguments(params, |_| {
            InputSource::default()
        }))
    }

    #[cfg(test)]
    pub fn from_sourced_optional_params(
        params: &Map<String, Value>,
        sources: &std::collections::BTreeMap<String, InputSource>,
    ) -> Result<Self, Error> {
        Self::from_credentials(&ConnectionArguments::from_arguments(params, |name| {
            sources.get(name).copied().unwrap_or_default()
        }))
    }

    pub fn from_credentials(credentials: &ConnectionArguments) -> Result<Self, Error> {
        Ok(Self {
            azure_ad_token: secret_config(credentials, &azure::AD_TOKEN)?,
            azure_ad_token_provider: None,
            credential_resolver: None,
            tenant_id: string_config(credentials, &azure::TENANT_ID)?,
            client_id: string_config(credentials, &azure::CLIENT_ID)?,
            client_secret: secret_config(credentials, &azure::CLIENT_SECRET)?,
            azure_scope: string_config(credentials, &azure::SCOPE)?,
            azure_authority_host: string_config(credentials, &azure::AUTHORITY_HOST)?,
            azure_credential: string_config(credentials, &azure::CREDENTIAL)?,
            federated_token_file: string_config(credentials, &azure::FEDERATED_TOKEN_FILE)?,
            enable_azure_ad_token_refresh: Sourced::new(
                azure::ENABLE_TOKEN_REFRESH
                    .kwarg(credentials)
                    .and_then(|(_, value)| value.as_bool())
                    .unwrap_or(false),
                credentials.source(azure::ENABLE_TOKEN_REFRESH.kwargs[0]),
            ),
        })
    }
}

fn string_config(
    credentials: &ConnectionArguments,
    field: &ConnectionField,
) -> Result<ConfigValue<String>, Error> {
    let Some((name, value)) = field.kwarg(credentials) else {
        return Ok(ConfigValue::Absent);
    };
    let source = credentials.source(name);
    match value {
        Value::Null => Ok(ConfigValue::ExplicitNone(source)),
        Value::String(value) => Ok(ConfigValue::Value(Sourced::new(value.clone(), source))),
        _ => Err(Error::InvalidConfiguration(
            litellm_auth_types::ErrorDetail::InvalidType {
                field: name.into(),
                expected: "a string or null",
            },
        )),
    }
}

fn secret_config(
    credentials: &ConnectionArguments,
    field: &ConnectionField,
) -> Result<ConfigValue<SecretValue>, Error> {
    Ok(match string_config(credentials, field)? {
        ConfigValue::Absent => ConfigValue::Absent,
        ConfigValue::ExplicitNone(source) => ConfigValue::ExplicitNone(source),
        ConfigValue::Value(value) => ConfigValue::Value(value.map(SecretValue::new)),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use litellm_auth_types::{InputSource, Sourced};
    use serde_json::json;

    use super::{AzureAuthInputs, AzureCredentialType, ConfigValue};

    #[test]
    fn selector_parsing_is_exact() {
        assert_eq!(
            "ClientSecretCredential".parse::<AzureCredentialType>(),
            Ok(AzureCredentialType::ClientSecretCredential)
        );
        assert!(
            "clientsecretcredential"
                .parse::<AzureCredentialType>()
                .is_err()
        );
    }

    #[test]
    fn defaults_preserve_absence() {
        let inputs = AzureAuthInputs::default();

        assert_eq!(inputs.tenant_id, ConfigValue::Absent);
        assert_eq!(inputs.azure_ad_token, ConfigValue::Absent);
    }

    #[test]
    fn parsing_distinguishes_null_empty_and_absent() {
        let params = json!({"tenant_id": null, "client_id": ""});
        let inputs = AzureAuthInputs::from_optional_params(params.as_object().unwrap()).unwrap();

        assert_eq!(
            inputs.tenant_id,
            ConfigValue::ExplicitNone(InputSource::Deployment)
        );
        assert_eq!(
            inputs.client_id,
            ConfigValue::Value(Sourced::new(String::new(), InputSource::Deployment))
        );
        assert_eq!(inputs.client_secret, ConfigValue::Absent);
    }

    #[test]
    fn parsing_preserves_trusted_input_sources() {
        let params = json!({"tenant_id": "tenant", "client_secret": null});
        let sources = BTreeMap::from([
            ("tenant_id".to_string(), InputSource::Request),
            ("client_secret".to_string(), InputSource::Request),
        ]);
        let inputs =
            AzureAuthInputs::from_sourced_optional_params(params.as_object().unwrap(), &sources)
                .unwrap();

        assert_eq!(
            inputs.tenant_id,
            ConfigValue::Value(Sourced::new("tenant".to_string(), InputSource::Request))
        );
        assert_eq!(
            inputs.client_secret,
            ConfigValue::ExplicitNone(InputSource::Request)
        );
    }

    #[test]
    fn debug_does_not_expose_secrets() {
        let params = json!({"azure_ad_token": "token-value", "client_secret": "secret-value"});
        let inputs = AzureAuthInputs::from_optional_params(params.as_object().unwrap()).unwrap();
        let debug = format!("{inputs:?}");

        assert!(!debug.contains("token-value"));
        assert!(!debug.contains("secret-value"));
    }

    #[rstest::rstest]
    #[case::global_turns_refresh_on(json!({}), true, true, InputSource::Deployment)]
    #[case::global_overrides_a_call_false_like_python(json!({"enable_azure_ad_token_refresh": false}), true, true, InputSource::Deployment)]
    #[case::call_true_survives_a_global_false(json!({"enable_azure_ad_token_refresh": true}), false, true, InputSource::Request)]
    #[case::both_off(json!({}), false, false, InputSource::Deployment)]
    fn token_refresh_follows_the_configured_global(
        #[case] params: serde_json::Value,
        #[case] global: bool,
        #[case] enabled: bool,
        #[case] source: InputSource,
    ) {
        let sources = BTreeMap::from([(
            "enable_azure_ad_token_refresh".to_string(),
            InputSource::Request,
        )]);
        let inputs =
            AzureAuthInputs::from_sourced_optional_params(params.as_object().unwrap(), &sources)
                .unwrap()
                .or_configured_token_refresh(global);
        assert_eq!(
            inputs.enable_azure_ad_token_refresh,
            Sourced::new(enabled, source)
        );
    }
}
