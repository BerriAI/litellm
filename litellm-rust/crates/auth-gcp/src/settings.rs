use litellm_auth_types::{Setting, env_names, secret, setting};

pub(crate) const GOOGLE_APPLICATION_CREDENTIALS_ENV: &str = "GOOGLE_APPLICATION_CREDENTIALS";
pub(crate) const VERTEX_AI_API_KEY_ENV: &str = "VERTEX_AI_API_KEY";
pub(crate) const VERTEXAI_API_KEY_ENV: &str = "VERTEXAI_API_KEY";
pub(crate) const VERTEXAI_CREDENTIALS_ENV: &str = "VERTEXAI_CREDENTIALS";
pub(crate) const VERTEXAI_PROJECT_ENV: &str = "VERTEXAI_PROJECT";
pub(crate) const VERTEXAI_LOCATION_ENV: &str = "VERTEXAI_LOCATION";
pub(crate) const VERTEX_LOCATION_ENV: &str = "VERTEX_LOCATION";

pub const API_KEY: Setting = secret(&[], &[VERTEX_AI_API_KEY_ENV, VERTEXAI_API_KEY_ENV]);
pub const CREDENTIALS: Setting = secret(
    &["vertex_credentials", "vertex_ai_credentials"],
    &[VERTEXAI_CREDENTIALS_ENV, GOOGLE_APPLICATION_CREDENTIALS_ENV],
);
pub const PROJECT: Setting = setting(
    &["vertex_project", "vertex_ai_project"],
    &[VERTEXAI_PROJECT_ENV],
);
pub const LOCATION: Setting = setting(
    &["vertex_location", "vertex_ai_location"],
    &[VERTEXAI_LOCATION_ENV, VERTEX_LOCATION_ENV],
);

pub const SETTINGS: &[Setting] = &[API_KEY, CREDENTIALS, PROJECT, LOCATION];

pub fn secret_names() -> Vec<&'static str> {
    env_names(SETTINGS).collect()
}
