pub use crate::error::RouteError as Error;
pub mod websocket;

mod handler;
mod prepare;
pub mod route;
pub mod types;

use litellm_host::hooks::RouteHooks;
use litellm_http::{ClientVariant, HttpClientConfig};
use litellm_secrets::source::SecretSource;
use types::{ResponsesCall, ResponsesOutput};

pub async fn responses(
    resources: &crate::resources::CoreResources,
    config: &HttpClientConfig,
    secrets: &dyn SecretSource,
    call: ResponsesCall,
) -> Result<ResponsesOutput, Error> {
    responses_with_hooks(resources, config, secrets, call, &()).await
}

pub async fn responses_with_hooks(
    resources: &crate::resources::CoreResources,
    config: &HttpClientConfig,
    secrets: &dyn SecretSource,
    call: ResponsesCall,
    hooks: &impl RouteHooks<Error>,
) -> Result<ResponsesOutput, Error> {
    litellm_host::lifecycle::observe_call(hooks.observer(), async {
        let http = resources.pool.client(config, ClientVariant::Provider)?;
        execute(&http, &resources.auth, secrets, call, hooks).await
    })
    .await
}

async fn execute(
    http: &litellm_http::Client,
    auth: &litellm_auth::AuthServices,
    secrets: &dyn SecretSource,
    call: ResponsesCall,
    hooks: &impl RouteHooks<Error>,
) -> Result<ResponsesOutput, Error> {
    let request = prepare::prepare(call, secrets).await?;
    handler::execute(http, auth, request, hooks).await
}
