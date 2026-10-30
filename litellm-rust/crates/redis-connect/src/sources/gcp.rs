use std::time::SystemTime;

use litellm_auth_gcp::{VertexAuth, VertexConfig};
use litellm_config::SharedLookup;
use serde::Deserialize;

use crate::{CredentialSource, Expiring, SourceFuture};

const IAM_CREDENTIALS: &str = "https://iamcredentials.googleapis.com/v1";
const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

pub(crate) struct GcpIam {
    http: litellm_http::Client,
    identity: VertexAuth,
    service_account: String,
    lookup: SharedLookup,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct AccessToken {
    access_token: String,
    expire_time: String,
}

impl GcpIam {
    pub(crate) fn new(http: litellm_http::Client, service_account: String, lookup: SharedLookup) -> Self {
        Self {
            http,
            identity: VertexAuth::default(),
            service_account,
            lookup,
        }
    }
}

impl CredentialSource for GcpIam {
    fn fetch(&self) -> SourceFuture<'_> {
        Box::pin(async move {
            let lookup = |name: &str| (self.lookup)(name);
            let caller = self
                .identity
                .access_token(&VertexConfig::new(None, None, None), &lookup)
                .await?;
            let reply: AccessToken = self
                .http
                .post(format!("{IAM_CREDENTIALS}/{}:generateAccessToken", self.service_account))
                .bearer_auth(caller)
                .json(&serde_json::json!({ "scope": [CLOUD_PLATFORM_SCOPE] }))
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            let expires_at = time::OffsetDateTime::parse(&reply.expire_time, &time::format_description::well_known::Rfc3339)
                .ok()
                .map(SystemTime::from);
            Ok(Expiring {
                username: None,
                password: reply.access_token,
                expires_at,
            })
        })
    }
}
