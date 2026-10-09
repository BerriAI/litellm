use litellm_auth_types::{Setting, env_names, secret, setting};

pub(crate) const AZURE_AD_TOKEN_ENV: &str = "AZURE_AD_TOKEN";
pub(crate) const AZURE_TENANT_ID_ENV: &str = "AZURE_TENANT_ID";
pub(crate) const AZURE_CLIENT_ID_ENV: &str = "AZURE_CLIENT_ID";
pub(crate) const AZURE_CLIENT_SECRET_ENV: &str = "AZURE_CLIENT_SECRET";
pub(crate) const AZURE_SCOPE_ENV: &str = "AZURE_SCOPE";
pub(crate) const AZURE_AUTHORITY_HOST_ENV: &str = "AZURE_AUTHORITY_HOST";
pub(crate) const AZURE_CREDENTIAL_ENV: &str = "AZURE_CREDENTIAL";
pub(crate) const AZURE_FEDERATED_TOKEN_FILE_ENV: &str = "AZURE_FEDERATED_TOKEN_FILE";

pub const AD_TOKEN: Setting = secret(&["azure_ad_token"], &[AZURE_AD_TOKEN_ENV]);
pub const TENANT_ID: Setting = setting(&["tenant_id"], &[AZURE_TENANT_ID_ENV]);
pub const CLIENT_ID: Setting = setting(&["client_id"], &[AZURE_CLIENT_ID_ENV]);
pub const CLIENT_SECRET: Setting = secret(&["client_secret"], &[AZURE_CLIENT_SECRET_ENV]);
pub const SCOPE: Setting = setting(&["azure_scope"], &[AZURE_SCOPE_ENV]);
pub const AUTHORITY_HOST: Setting = setting(&["azure_authority_host"], &[AZURE_AUTHORITY_HOST_ENV]);
pub const CREDENTIAL: Setting = setting(&["azure_credential"], &[AZURE_CREDENTIAL_ENV]);
pub const FEDERATED_TOKEN_FILE: Setting = secret(
    &["azure_federated_token_file"],
    &[AZURE_FEDERATED_TOKEN_FILE_ENV],
);
pub const AD_TOKEN_PROVIDER: Setting = setting(&["azure_ad_token_provider"], &[]);
pub const TOKEN_REFRESH: Setting = setting(&["enable_azure_ad_token_refresh"], &[]);

pub const SETTINGS: &[Setting] = &[
    AD_TOKEN,
    TENANT_ID,
    CLIENT_ID,
    CLIENT_SECRET,
    SCOPE,
    AUTHORITY_HOST,
    CREDENTIAL,
    FEDERATED_TOKEN_FILE,
    AD_TOKEN_PROVIDER,
    TOKEN_REFRESH,
];

pub fn secret_names() -> Vec<&'static str> {
    env_names(SETTINGS).collect()
}
