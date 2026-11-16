use std::sync::Arc;

use axum::{Json, body::Bytes, extract::State, response::Response};
use litellm_cache_response::CachePolicy;
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_host::call::CallOutput;
use litellm_host_http::{Ran, Sse};
use litellm_inference_responses::{
    ResponsesRoute,
    encoded_stream::encode_stream,
    ids::{self, IdScope},
    route::Responses,
    types::ResponsesCall,
};
use litellm_router_core::{Affinity, Attempt, Op, Served};
use serde_json::{Map, Value};

use crate::{
    Error, Gateway, JsonObject,
    caching::CacheHeaders,
    request,
    routing::{self, Host, RouteAttempt},
};

pub(crate) async fn create<H: Host>(
    State(gateway): State<Arc<Gateway<H>>>,
    identity: AuthenticatedRequest,
    JsonObject(body): JsonObject,
) -> Result<Response, Error> {
    let model = request::model(&body)?.to_owned();
    let (body, cache_options) = crate::caching::prepare(&identity, body)?;
    let route = gateway.responses.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };
    let attempt = ResponsesAttempt {
        gateway: &gateway,
        route,
        body,
        policy: cache_options.policy,
    };
    let routed = routing::call(
        &gateway,
        &identity,
        Op::RESPONSES,
        "/responses",
        &model,
        &attempt,
    )
    .await?;
    let routed_headers = routing::headers(&routed);
    let (ran, cache) = routed.output.response;
    let stream = Sse::<Responses, _, _>::new(Json, |error| {
        Bytes::from(Error::from(error).responses_sse_frame())
    });
    let response = litellm_host_http::render(stream, ran)?;
    Ok(routing::respond(routed_headers, cache.apply(response)))
}

struct ResponsesAttempt<'a, H> {
    gateway: &'a Gateway<H>,
    route: ResponsesRoute,
    body: Map<String, Value>,
    policy: CachePolicy,
}

impl<H: Host> RouteAttempt for ResponsesAttempt<'_, H> {
    type Outcome = (Ran<Responses>, CacheHeaders);

    fn body(&self) -> Value {
        Value::Object(self.body.clone())
    }

    fn affinity(&self) -> Affinity {
        Affinity {
            previous_response: self
                .body
                .get("previous_response_id")
                .and_then(Value::as_str)
                .and_then(ids::previous_response_deployment),
            encrypted_content: self
                .body
                .get("input")
                .and_then(ids::encrypted_content_origin),
        }
    }

    async fn run(&self, attempt: &Attempt) -> Result<Served<Self::Outcome>, Error> {
        let deployment = &attempt.deployment;
        let router = &self.gateway.router;
        let affinity = router.encrypted_content_affinity(&attempt.model_group);
        let mut body = routing::attempt_body(&self.body, attempt);
        let mut input = body.remove("input").unwrap_or_default();
        if let Some(origin) = attempt.request.affinity.encrypted_content.as_deref()
            && affinity
        {
            ids::strip_undecryptable_reasoning(&mut input, origin, |minted_by| {
                router.decrypts(minted_by, deployment)
            });
        }
        ids::restore_input(&mut input);
        if let Some(Value::String(previous)) = body.get_mut("previous_response_id") {
            *previous = ids::decode_response_id(previous).response_id;
        }
        let scope = IdScope {
            provider: Some(deployment.provider().to_owned()),
            model_id: Some(deployment.id.clone()),
            encrypted_content_affinity: affinity,
        };
        let call = ResponsesCall {
            model: deployment.params.model.clone(),
            input,
            optional_params: body
                .into_iter()
                .filter(|(name, _)| name != "model")
                .collect(),
            api_key: routing::api_key(deployment),
            api_base: deployment.params.api_base.clone(),
            custom_llm_provider: deployment.params.custom_llm_provider.clone(),
            extra_headers: None,
            timeout: routing::timeout(attempt),
        };
        let machine = self.route.clone().machine(call, self.policy);
        let cache = CacheHeaders::default();
        let ran = litellm_host_http::run(machine, (), cache.clone(), None).await?;
        Ok(match ran {
            CallOutput::Complete(mut response) => {
                ids::encode_response(&mut response, &scope);
                let total_tokens = routing::total_tokens(response.extra.get("usage"));
                Served::new((CallOutput::Complete(response), cache), total_tokens)
            }
            CallOutput::Stream { head, chunks } => {
                let chunks = routing::reporting_success(
                    self.gateway,
                    deployment.id.clone(),
                    encode_stream(chunks, scope),
                );
                Served {
                    deferred: true,
                    ..Served::new((CallOutput::Stream { head, chunks }, cache), 0)
                }
            }
        })
    }
}
