use std::sync::LazyLock;

use litellm_auth_types::VertexParams;

pub const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
pub const GOOGLE_OAUTH_TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";
pub const GOOGLE_APPLICATION_CREDENTIALS_ENV: &str = "GOOGLE_APPLICATION_CREDENTIALS";
pub const GOOGLE_CLOUD_PROJECT_ENV: &str = "GOOGLE_CLOUD_PROJECT";
pub const GCLOUD_PROJECT_ENV: &str = "GCLOUD_PROJECT";
pub const CLOUDSDK_CONFIG_ENV: &str = "CLOUDSDK_CONFIG";
pub const CLOUDSDK_CORE_PROJECT_ENV: &str = "CLOUDSDK_CORE_PROJECT";
pub const CLOUDSDK_ACTIVE_CONFIG_NAME_ENV: &str = "CLOUDSDK_ACTIVE_CONFIG_NAME";
pub const VERTEX_AI_API_KEY_ENV: &str = "VERTEX_AI_API_KEY";
pub const VERTEXAI_API_KEY_ENV: &str = "VERTEXAI_API_KEY";
pub const VERTEXAI_CREDENTIALS_ENV: &str = VertexParams::CREDENTIALS.env[0];
pub const VERTEX_LOCATION_ENV: &str = VertexParams::LOCATION.env[1];

/// Every environment name the Vertex params and the token acquisition read, for hosts
/// that resolve secrets up front.
pub fn secret_names() -> &'static [&'static str] {
    static NAMES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
        [
            VERTEX_AI_API_KEY_ENV,
            VERTEXAI_API_KEY_ENV,
            GOOGLE_APPLICATION_CREDENTIALS_ENV,
            GOOGLE_CLOUD_PROJECT_ENV,
            GCLOUD_PROJECT_ENV,
            CLOUDSDK_CONFIG_ENV,
            CLOUDSDK_CORE_PROJECT_ENV,
            CLOUDSDK_ACTIVE_CONFIG_NAME_ENV,
        ]
        .into_iter()
        .chain(
            VertexParams::SPECS
                .iter()
                .flat_map(|spec| spec.env.iter().copied()),
        )
        .fold(Vec::new(), |mut names, name| {
            if !names.contains(&name) {
                names.push(name);
            }
            names
        })
    });
    &NAMES
}
