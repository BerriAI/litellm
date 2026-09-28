pub use crate::error::RouteError as Error;
pub mod websocket;

mod handler;
mod prepare;
pub mod route;
pub mod types;

use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_host::hooks::RouteHooks;
use litellm_secrets::source::SecretSource;
use types::{ResponsesCall, ResponsesOutput};

#[derive(Clone)]
pub struct ResponsesRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
}

impl ResponsesRoute {
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
        call: ResponsesCall,
        hooks: &impl RouteHooks<Error>,
    ) -> Result<ResponsesOutput, Error> {
        litellm_host::lifecycle::observe_call(hooks.observer(), self.run(call, hooks)).await
    }

    async fn run(
        &self,
        call: ResponsesCall,
        hooks: &impl RouteHooks<Error>,
    ) -> Result<ResponsesOutput, Error> {
        let request = prepare::prepare(call, self.secrets.as_ref()).await?;
        handler::execute(&self.http, &self.auth, request, hooks).await
    }
}
