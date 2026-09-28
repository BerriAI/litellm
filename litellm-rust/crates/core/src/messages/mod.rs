use litellm_host::observation::ObservationSender;
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
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<ObservationSender>,
    ) -> Result<MessagesResponse, Error> {
        litellm_host::lifecycle::observe_call(
            observers.clone(),
            self.run(call, interceptors, observers.as_ref()),
        )
        .await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "messages",
        model = %call.body.model,
        provider,
        resolved_model,
        stream = call.body.params.stream == Some(true),
        outcome
    ))]
    async fn run(
        &self,
        call: MessagesCall,
        interceptors: &impl litellm_host::interceptors::Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<MessagesResponse, Error> {
        crate::diagnostic::call(async {
            let request = prepare::prepare(call, self.secrets.as_ref()).await?;
            crate::diagnostic::provider(&request.body.model, request.provider.as_str());
            let execute: futures_util::future::BoxFuture<'_, Result<MessagesResponse, Error>> =
                Box::pin(handler::execute(
                    &self.http,
                    &self.auth,
                    request,
                    interceptors,
                    observers,
                ));
            execute.await
        })
        .await
    }
}
