use serde_json::Value;

use crate::ConnectionArguments;

/// One connection setting: the call arguments that carry it (aliases in precedence order),
/// the environment variables that back it when no argument does, and whether its value is
/// a secret.
#[derive(Debug)]
pub struct ConnectionField {
    pub kwargs: &'static [&'static str],
    pub env: &'static [&'static str],
    pub secret: bool,
}

impl ConnectionField {
    pub fn kwarg<'a>(
        &self,
        connection: &'a ConnectionArguments,
    ) -> Option<(&'static str, &'a Value)> {
        self.kwargs
            .iter()
            .find_map(|name| connection.get(name).map(|value| (*name, value)))
    }

    pub fn str_kwarg<'a>(&self, connection: &'a ConnectionArguments) -> Option<&'a str> {
        self.kwarg(connection).and_then(|(_, value)| value.as_str())
    }

    pub fn env(&self, env_lookup: &dyn Fn(&str) -> Option<String>) -> Option<String> {
        self.env.iter().find_map(|name| env_lookup(name))
    }

    pub fn non_empty_env(&self, env_lookup: &dyn Fn(&str) -> Option<String>) -> Option<String> {
        self.env.iter().find_map(|name| {
            env_lookup(name)
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
    }
}

const fn field(kwargs: &'static [&'static str], env: &'static [&'static str]) -> ConnectionField {
    ConnectionField {
        kwargs,
        env,
        secret: false,
    }
}

const fn secret(kwargs: &'static [&'static str], env: &'static [&'static str]) -> ConnectionField {
    ConnectionField {
        kwargs,
        env,
        secret: true,
    }
}

pub fn env_names(fields: &[&ConnectionField]) -> Vec<&'static str> {
    fields
        .iter()
        .flat_map(|field| field.env.iter().copied())
        .collect()
}

pub mod aws {
    use super::{ConnectionField, field, secret};

    pub const ACCESS_KEY_ID: ConnectionField =
        field(&["aws_access_key_id"], &["AWS_ACCESS_KEY_ID"]);
    pub const SECRET_ACCESS_KEY: ConnectionField =
        secret(&["aws_secret_access_key"], &["AWS_SECRET_ACCESS_KEY"]);
    pub const SESSION_TOKEN: ConnectionField =
        secret(&["aws_session_token"], &["AWS_SESSION_TOKEN"]);
    pub const REGION_NAME: ConnectionField =
        field(&["aws_region_name"], &["AWS_REGION_NAME", "AWS_REGION"]);
    pub const SESSION_NAME: ConnectionField = field(&["aws_session_name"], &["AWS_SESSION_NAME"]);
    pub const PROFILE_NAME: ConnectionField = field(&["aws_profile_name"], &["AWS_PROFILE_NAME"]);
    pub const ROLE_NAME: ConnectionField = field(&["aws_role_name"], &["AWS_ROLE_NAME"]);
    pub const WEB_IDENTITY_TOKEN: ConnectionField =
        secret(&["aws_web_identity_token"], &["AWS_WEB_IDENTITY_TOKEN"]);
    pub const STS_ENDPOINT: ConnectionField = field(&["aws_sts_endpoint"], &["AWS_STS_ENDPOINT"]);
    pub const EXTERNAL_ID: ConnectionField = field(&["aws_external_id"], &["AWS_EXTERNAL_ID"]);
    pub const BEDROCK_RUNTIME_ENDPOINT: ConnectionField = field(
        &["aws_bedrock_runtime_endpoint"],
        &["AWS_BEDROCK_RUNTIME_ENDPOINT"],
    );

    pub const FIELDS: &[&ConnectionField] = &[
        &ACCESS_KEY_ID,
        &SECRET_ACCESS_KEY,
        &SESSION_TOKEN,
        &REGION_NAME,
        &SESSION_NAME,
        &PROFILE_NAME,
        &ROLE_NAME,
        &WEB_IDENTITY_TOKEN,
        &STS_ENDPOINT,
        &EXTERNAL_ID,
        &BEDROCK_RUNTIME_ENDPOINT,
    ];
}

pub mod azure {
    use super::{ConnectionField, field, secret};

    pub const AD_TOKEN: ConnectionField = secret(&["azure_ad_token"], &["AZURE_AD_TOKEN"]);
    pub const TENANT_ID: ConnectionField = field(&["tenant_id"], &["AZURE_TENANT_ID"]);
    pub const CLIENT_ID: ConnectionField = field(&["client_id"], &["AZURE_CLIENT_ID"]);
    pub const CLIENT_SECRET: ConnectionField = secret(&["client_secret"], &["AZURE_CLIENT_SECRET"]);
    pub const SCOPE: ConnectionField = field(&["azure_scope"], &["AZURE_SCOPE"]);
    pub const AUTHORITY_HOST: ConnectionField =
        field(&["azure_authority_host"], &["AZURE_AUTHORITY_HOST"]);
    pub const CREDENTIAL: ConnectionField = field(&["azure_credential"], &["AZURE_CREDENTIAL"]);
    pub const FEDERATED_TOKEN_FILE: ConnectionField = secret(
        &["azure_federated_token_file"],
        &["AZURE_FEDERATED_TOKEN_FILE"],
    );
    pub const ENABLE_TOKEN_REFRESH: ConnectionField =
        field(&["enable_azure_ad_token_refresh"], &[]);

    pub const FIELDS: &[&ConnectionField] = &[
        &AD_TOKEN,
        &TENANT_ID,
        &CLIENT_ID,
        &CLIENT_SECRET,
        &SCOPE,
        &AUTHORITY_HOST,
        &CREDENTIAL,
        &FEDERATED_TOKEN_FILE,
        &ENABLE_TOKEN_REFRESH,
    ];
}

pub mod vertex {
    use super::{ConnectionField, field, secret};

    pub const CREDENTIALS: ConnectionField = secret(
        &["vertex_credentials", "vertex_ai_credentials"],
        &["VERTEXAI_CREDENTIALS"],
    );
    pub const APPLICATION_CREDENTIALS: ConnectionField =
        field(&[], &["GOOGLE_APPLICATION_CREDENTIALS"]);
    pub const PROJECT: ConnectionField = field(
        &["vertex_project", "vertex_ai_project"],
        &["VERTEXAI_PROJECT"],
    );
    pub const LOCATION: ConnectionField = field(
        &["vertex_location", "vertex_ai_location"],
        &["VERTEXAI_LOCATION", "VERTEX_LOCATION"],
    );
    pub const API_KEY: ConnectionField = secret(&[], &["VERTEX_AI_API_KEY", "VERTEXAI_API_KEY"]);

    pub const FIELDS: &[&ConnectionField] = &[
        &CREDENTIALS,
        &APPLICATION_CREDENTIALS,
        &PROJECT,
        &LOCATION,
        &API_KEY,
    ];
}

pub const CUSTOM_ENDPOINT: ConnectionField = field(&["custom_endpoint"], &[]);

pub(crate) const ALL: &[&[&ConnectionField]] = &[
    aws::FIELDS,
    azure::FIELDS,
    vertex::FIELDS,
    &[&CUSTOM_ENDPOINT],
];

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::ALL;

    #[test]
    fn every_kwarg_belongs_to_one_field() {
        let kwargs = ALL
            .iter()
            .flat_map(|fields| fields.iter())
            .flat_map(|field| field.kwargs.iter())
            .collect::<Vec<_>>();
        assert_eq!(kwargs.len(), kwargs.iter().collect::<BTreeSet<_>>().len());
    }
}
