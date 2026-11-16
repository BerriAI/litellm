use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{
    Extension, Json,
    extract::State,
    response::{IntoResponse, Response},
};
use litellm_cache_response::CachePolicy;
use litellm_inference_chat::{
    ChatCompletionsRoute,
    types::{ChatCompletionsCall, ClientSideCredentials, DropParams, OptionalParams},
};
use litellm_llms_types::formats::chat_completions::ChatCompletionsResponse;
use litellm_router_core::{Attempt, Deployment, Op, Served};
use serde_json::{Map, Value};

use crate::{
    Ended, Error, Gateway, JsonObject,
    caching::CacheHeaders,
    headers::Routing,
    request::{self, Path},
    routing::{self, Host, RouteAttempt},
};

pub(crate) async fn create<H: Host>(
    State(gateway): State<Arc<Gateway<H>>>,
    identity: AuthenticatedRequest,
    JsonObject(body): JsonObject,
) -> Result<impl IntoResponse, Ended> {
    handle(&gateway, &identity, body).await
}

pub(crate) async fn create_from_model_path<H: Host>(
    State(gateway): State<Arc<Gateway<H>>>,
    identity: AuthenticatedRequest,
    Path(model): Path<String>,
    JsonObject(body): JsonObject,
) -> Result<impl IntoResponse, Ended> {
    let body = match body.get("model") {
        None | Some(Value::Null) => body
            .into_iter()
            .chain([("model".into(), Value::String(model))])
            .collect(),
        Some(_) => body,
    };
    handle(&gateway, &identity, body).await
}

async fn handle<H: Host>(
    gateway: &Gateway<H>,
    identity: &AuthenticatedRequest,
    body: Map<String, Value>,
) -> Result<Response, Ended> {
    let model = request::model(&body)?.to_owned();
    let (body, cache_options) = crate::caching::prepare(identity, body)?;
    let route = gateway.chat_completions.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };
    let attempt = ChatAttempt {
        gateway,
        route,
        body,
        policy: cache_options.policy,
    };
    let routed = routing::call(
        gateway,
        identity,
        Op::COMPLETION,
        "/chat/completions",
        &model,
        &attempt,
    )
    .await?;
    let routing = Routing::served(&routed);
    let (response, cache) = routed.output.response;
    Ok((
        Extension(routing),
        cache.apply(Json(response).into_response()),
    )
        .into_response())
}

struct ChatAttempt<'a, H> {
    gateway: &'a Gateway<H>,
    route: ChatCompletionsRoute,
    body: Map<String, Value>,
    policy: CachePolicy,
}

impl<H: Host> RouteAttempt for ChatAttempt<'_, H> {
    type Outcome = (ChatCompletionsResponse, CacheHeaders);

    async fn run(&self, attempt: &Attempt) -> Result<Served<Self::Outcome>, Error> {
        let deployment = &attempt.deployment;
        let body = routing::attempt_body(&self.body, attempt);
        let messages = body.get("messages").cloned().unwrap_or_default();
        let params = body
            .into_iter()
            .filter(|(name, _)| !matches!(name.as_str(), "model" | "messages"))
            .collect();
        let cache = CacheHeaders::default();
        let response = litellm_host_http::run_unary(
            self.route.clone().machine(
                ChatCompletionsCall {
                    model: deployment.params.model.clone(),
                    messages,
                    optional_params: OptionalParams::OpenAi {
                        params,
                        client_side: self.client_side(deployment),
                        drop_params: self.drop_params(deployment),
                    },
                    api_key: routing::api_key(deployment),
                    api_base: deployment.params.api_base.clone(),
                    custom_llm_provider: deployment.params.custom_llm_provider.clone(),
                    extra_headers: None,
                    timeout: routing::timeout(attempt, routing::streamed(&self.body)),
                },
                self.policy,
            ),
            (),
            cache.clone(),
            None,
        )
        .await?;
        let total_tokens = response.usage.total_tokens;
        Ok(Served::new((response, cache), total_tokens))
    }
}

impl<H: Host> ChatAttempt<'_, H> {
    fn client_side(&self, deployment: &Deployment) -> ClientSideCredentials {
        if self.gateway.client_side_credentials {
            return ClientSideCredentials::Any;
        }
        ClientSideCredentials::Only(
            deployment
                .params
                .extra
                .get("configurable_clientside_auth_params")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect(),
        )
    }

    fn drop_params(&self, deployment: &Deployment) -> DropParams {
        let shaping = self.gateway.shaping(&deployment.id);
        if !shaping.settings.drop_params {
            return DropParams::Off;
        }
        DropParams::On {
            model_rejects_sampling: !shaping.capabilities.supports_sampling_params,
        }
    }
}
