use litellm_auth_types::{Setting, env_names, secret, setting};

use crate::constants::{
    AWS_ACCESS_KEY_ID, AWS_BEDROCK_RUNTIME_ENDPOINT, AWS_EXTERNAL_ID, AWS_PROFILE_NAME, AWS_REGION,
    AWS_REGION_NAME, AWS_ROLE_NAME, AWS_SECRET_ACCESS_KEY, AWS_SESSION_NAME, AWS_SESSION_TOKEN,
    AWS_STS_ENDPOINT, AWS_WEB_IDENTITY_TOKEN,
};

pub const ACCESS_KEY_ID: Setting = setting(&["aws_access_key_id"], &[AWS_ACCESS_KEY_ID]);
pub const SECRET_ACCESS_KEY: Setting = secret(&["aws_secret_access_key"], &[AWS_SECRET_ACCESS_KEY]);
pub const SESSION_TOKEN: Setting = secret(&["aws_session_token"], &[AWS_SESSION_TOKEN]);
pub const REGION_NAME: Setting = setting(&["aws_region_name"], &[AWS_REGION_NAME, AWS_REGION]);
pub const SESSION_NAME: Setting = setting(&["aws_session_name"], &[AWS_SESSION_NAME]);
pub const PROFILE_NAME: Setting = setting(&["aws_profile_name"], &[AWS_PROFILE_NAME]);
pub const ROLE_NAME: Setting = setting(&["aws_role_name"], &[AWS_ROLE_NAME]);
pub const WEB_IDENTITY_TOKEN: Setting =
    secret(&["aws_web_identity_token"], &[AWS_WEB_IDENTITY_TOKEN]);
pub const STS_ENDPOINT: Setting = setting(&["aws_sts_endpoint"], &[AWS_STS_ENDPOINT]);
pub const EXTERNAL_ID: Setting = setting(&["aws_external_id"], &[AWS_EXTERNAL_ID]);
pub const BEDROCK_RUNTIME_ENDPOINT: Setting = setting(
    &["aws_bedrock_runtime_endpoint"],
    &[AWS_BEDROCK_RUNTIME_ENDPOINT],
);

pub const SETTINGS: &[Setting] = &[
    ACCESS_KEY_ID,
    SECRET_ACCESS_KEY,
    SESSION_TOKEN,
    REGION_NAME,
    SESSION_NAME,
    PROFILE_NAME,
    ROLE_NAME,
    WEB_IDENTITY_TOKEN,
    STS_ENDPOINT,
    EXTERNAL_ID,
    BEDROCK_RUNTIME_ENDPOINT,
];

pub fn secret_names() -> Vec<&'static str> {
    env_names(SETTINGS).collect()
}
