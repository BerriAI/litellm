mod common_utils;
mod handler;
mod prepare;
pub mod route;
mod types;

use litellm_auth::AuthServices;
use litellm_secrets::source::SecretSource;
use std::sync::Arc;

pub use crate::error::RouteError as Error;
pub use types::{MessagesCall, MessagesResponse, MessagesShaping, messages_body};

#[derive(Clone)]
pub struct MessagesRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
}

impl MessagesRoute {
    pub fn new(
        http: litellm_http::Client,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self {
        Self {
            http,
            auth,
            secrets,
        }
    }

    pub async fn execute(
        &self,
        call: MessagesCall,
        hooks: &impl litellm_host::hooks::RouteHooks<Error>,
    ) -> Result<MessagesResponse, Error> {
        litellm_host::lifecycle::observe_call(hooks.observer(), self.run(call, hooks)).await
    }

    async fn run(
        &self,
        call: MessagesCall,
        hooks: &impl litellm_host::hooks::RouteHooks<Error>,
    ) -> Result<MessagesResponse, Error> {
        let request = prepare::prepare(call, self.secrets.as_ref()).await?;
        handler::execute(&self.http, &self.auth, request, hooks).await
    }
}
