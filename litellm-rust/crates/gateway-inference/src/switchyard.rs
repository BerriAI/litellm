use std::{
    collections::{BTreeMap, HashMap, HashSet},
    convert::Infallible,
    sync::Arc,
};

use async_trait::async_trait;
use axum::{
    Json,
    http::HeaderMap,
    response::{
        IntoResponse, Response as HttpResponse,
        sse::{Event, Sse},
    },
};
use futures_util::{StreamExt, stream};
use litellm_cache_response::{CacheOptions, CachePolicy, ScopedCache};
use litellm_config::Model;
use litellm_host::call::CallOutput;
use litellm_http::transport::Error as TransportError;
use litellm_inference::RouteError;
use litellm_inference_chat::ChatCompletionsRoute;
use litellm_inference_messages::MessagesRoute;
use litellm_inference_responses::ResponsesRoute;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use switchyard_libsy::{Algorithm, LibsyError, RuntimeModels};
use switchyard_llm_client::ClientRouter;
use switchyard_protocol::{
    BoxError, LlmClientError, LlmResponse, Metadata, ModelId, Request, Response, RoutedLlmClient,
    WireFormat,
};
use switchyard_runner::AlgorithmSpec;
use switchyard_translation::{
    LlmStreamError, RawEventStream, decode_aggregated_response, decode_request, decode_stream,
    encode_aggregated_response_with_extensions, encode_request, encode_stream_with_extensions,
};

use crate::{
    Deployment, Error, Gateway, ModelRouter, SwitchyardConfigError, caching::CacheHeaders,
    chat_completions, messages, responses,
};

const SWITCHYARD_MODEL: &str = "switchyard/algorithm";

#[derive(Default)]
pub(crate) struct Routes(HashMap<String, Route>);

impl Routes {
    pub(crate) fn from_models(
        model_list: &[Model],
        deployments: &ModelRouter,
    ) -> Result<Self, SwitchyardConfigError> {
        let mut routes = HashMap::new();
        let mut seen = HashSet::new();
        // ModelRouter uses the final entry when public model names repeat.
        for model in model_list.iter().rev() {
            if !seen.insert(model.model_name.as_str())
                || model.litellm_params.model != SWITCHYARD_MODEL
            {
                continue;
            }
            if deployments
                .get(&model.model_name)
                .is_none_or(|deployment| deployment.model != SWITCHYARD_MODEL)
            {
                return Err(invalid(model, "model is not a Switchyard route"));
            }
            let config = model
                .litellm_params
                .extra
                .get("switchyard_config")
                .ok_or_else(|| invalid(model, "switchyard_config is required"))?;
            let config = AlgorithmSpec::deserialize(config)
                .map_err(|error| invalid(model, error.to_string()))?;
            let targets = targets(model, &config, model_list, deployments)?;
            let (algorithm, runtime_models) = config
                .build_with_runtime_models(&model.model_name, &targets.models)
                .map_err(|error| invalid(model, error.to_string()))?;
            routes.insert(
                model.model_name.clone(),
                Route {
                    algorithm,
                    runtime_models: Arc::new(runtime_models),
                    targets: Arc::new(targets.by_model),
                },
            );
        }
        Ok(Self(routes))
    }

    pub(crate) fn get(&self, model: &str) -> Option<&Route> {
        self.0.get(model)
    }
}

pub(crate) struct Route {
    algorithm: Arc<dyn Algorithm>,
    runtime_models: Arc<RuntimeModels>,
    targets: Arc<HashMap<ModelId, Target>>,
}

impl Route {
    pub(crate) async fn execute(
        &self,
        gateway: &Gateway,
        cache_options: CacheOptions,
        wire_format: WireFormat,
        body: Map<String, Value>,
        headers: HeaderMap,
    ) -> Result<HttpResponse, Error> {
        let llm_request = decode_request(wire_format, &Value::Object(body))
            .map_err(|error| Error::InvalidBody(error.to_string()))?;
        let extensions = llm_request.extensions.clone();
        let mut metadata = Metadata::from_headers(&headers);
        metadata.http_headers = Some(headers);
        let request = Request {
            llm_request,
            raw_request: None,
            metadata: Some(metadata),
        };
        let client = Arc::new(LiteLlmClient {
            routes: ProviderRoutes::new(gateway, &cache_options),
            targets: Arc::clone(&self.targets),
            cache_policy: cache_options.policy,
        });
        let (selected_model, response) = switchyard_llm_client::run(
            Arc::clone(&self.algorithm),
            ClientRouter::single(client),
            request,
            Arc::clone(&self.runtime_models),
            None,
        )
        .await
        .map_err(map_libsy_error)?;
        let served_model = response.served_model().cloned().unwrap_or(selected_model);
        let upstream_headers = response.upstream_headers;
        let mut response = match response.llm_response {
            LlmResponse::Agg(response) => {
                let body = encode_aggregated_response_with_extensions(
                    &response,
                    wire_format,
                    Some(served_model.as_str()),
                    &extensions,
                )
                .map_err(|error| Error::Internal(error.to_string()))?;
                Json(body).into_response()
            }
            LlmResponse::Stream(response) => {
                let events = encode_stream_with_extensions(
                    response,
                    wire_format,
                    Some(served_model.into_string()),
                    &extensions,
                )
                .map_err(|error| Error::Internal(error.to_string()))?;
                stream_response(events, wire_format)
            }
        };
        response.headers_mut().extend(upstream_headers);
        Ok(response)
    }
}

struct Target {
    deployment: Deployment,
    wire_format: WireFormat,
}

struct ProviderRoutes {
    chat: ChatCompletionsRoute,
    responses: ResponsesRoute,
    messages: MessagesRoute,
}

impl ProviderRoutes {
    fn new(gateway: &Gateway, options: &CacheOptions) -> Self {
        let mut routes = Self {
            chat: gateway.chat_completions.clone(),
            responses: gateway.responses.clone(),
            messages: gateway.messages.clone(),
        };
        if let Some(cache) = &gateway.cache {
            let scoped = || ScopedCache::new(Arc::clone(cache), options.scope.clone());
            routes.chat = routes.chat.with_cache(scoped());
            routes.responses = routes.responses.with_cache(scoped());
            routes.messages = routes.messages.with_cache(scoped());
        }
        routes
    }
}

struct LiteLlmClient {
    routes: ProviderRoutes,
    targets: Arc<HashMap<ModelId, Target>>,
    cache_policy: CachePolicy,
}

#[async_trait]
impl RoutedLlmClient for LiteLlmClient {
    async fn call(&self, request: Request) -> Result<Response, LlmClientError> {
        let Some(model) = request.llm_request.model.as_deref() else {
            return Err(LlmClientError::InvalidRequest {
                message: "routed request has no model".to_string(),
            });
        };
        let target = self
            .targets
            .get(model)
            .ok_or_else(|| LlmClientError::Configuration {
                message: format!("unknown LiteLLM deployment {model}"),
            })?;
        let Request {
            llm_request,
            metadata,
            ..
        } = request;
        let Value::Object(body) = encode_request(&llm_request, target.wire_format)
            .map_err(|error| LlmClientError::RequestEncoding(error.to_string()))?
        else {
            return Err(LlmClientError::RequestEncoding(format!(
                "{} request is not an object",
                target.wire_format
            )));
        };
        let cache_headers = CacheHeaders::default();
        let response = match target.wire_format {
            WireFormat::OpenAiChat => {
                let response = self
                    .routes
                    .chat
                    .execute(
                        chat_completions::provider_request(&target.deployment, body),
                        &cache_headers,
                        self.cache_policy,
                    )
                    .await
                    .map_err(map_route_error)?;
                decode_complete(response, target.wire_format)?
            }
            WireFormat::OpenAiResponses => {
                let response = self
                    .routes
                    .responses
                    .execute(
                        responses::provider_request(&target.deployment, body),
                        &cache_headers,
                        self.cache_policy,
                    )
                    .await
                    .map_err(map_route_error)?;
                decode_output(response, target.wire_format)?
            }
            WireFormat::AnthropicMessages => {
                let empty_headers = HeaderMap::new();
                let headers = metadata
                    .as_ref()
                    .and_then(|metadata| metadata.http_headers.as_ref())
                    .unwrap_or(&empty_headers);
                let call = messages::provider_request(&target.deployment, body, headers)
                    .map_err(map_route_error)?;
                let response = self
                    .routes
                    .messages
                    .execute(call, &cache_headers, self.cache_policy)
                    .await
                    .map_err(map_route_error)?;
                decode_output(response, target.wire_format)?
            }
        };
        let mut response = Response {
            llm_response: response,
            metadata,
            upstream_headers: HeaderMap::new(),
        };
        cache_headers.apply_to(&mut response.upstream_headers);
        Ok(response)
    }
}

fn decode_complete(
    response: impl serde::Serialize,
    wire_format: WireFormat,
) -> Result<LlmResponse, LlmClientError> {
    let body = serde_json::to_value(response)
        .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
    decode_aggregated_response(&body, wire_format)
        .map(LlmResponse::Agg)
        .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))
}

fn decode_output<T, H>(
    response: CallOutput<T, H, bytes::Bytes, RouteError>,
    wire_format: WireFormat,
) -> Result<LlmResponse, LlmClientError>
where
    T: serde::Serialize,
{
    match response {
        CallOutput::Complete(response) => decode_complete(response, wire_format),
        CallOutput::Stream { chunks, .. } => {
            let bytes =
                chunks.map(|chunk| chunk.map(|bytes| bytes.to_vec()).map_err(map_route_error));
            decode_stream(bytes, wire_format).map(LlmResponse::Stream)
        }
    }
}

// A stream error ends the response; Chat's success sentinel must not follow it.
fn stream_response(events: RawEventStream, wire_format: WireFormat) -> HttpResponse {
    let frames = stream::unfold(Some(events), move |events| async move {
        let mut events = events?;
        let (event, next) = match events.next().await {
            Some(Ok(value)) => (frame_event(&value, wire_format), Some(events)),
            Some(Err(LlmStreamError::Upstream(value))) => (frame_event(&value, wire_format), None),
            Some(Err(LlmStreamError::Client(error))) => {
                let error = map_client_error(error);
                let value = match wire_format {
                    WireFormat::OpenAiChat => error.openai_body(),
                    WireFormat::AnthropicMessages => error.body(None),
                    WireFormat::OpenAiResponses => json!({
                        "type": "error", "code": error.status().as_u16().to_string(),
                        "message": error.to_string(), "param": null
                    }),
                };
                (frame_event(&value, wire_format), None)
            }
            None if wire_format == WireFormat::OpenAiChat => {
                (Event::default().data("[DONE]"), None)
            }
            None => return None,
        };
        Some((Ok::<_, Infallible>(event), next))
    });
    Sse::new(frames).into_response()
}

fn frame_event(value: &Value, wire_format: WireFormat) -> Event {
    let event = Event::default().data(value.to_string());
    match wire_format {
        WireFormat::OpenAiChat => event,
        WireFormat::OpenAiResponses | WireFormat::AnthropicMessages => {
            let event_type = value
                .get("type")
                .and_then(Value::as_str)
                .filter(|name| !name.contains(['\r', '\n']))
                .unwrap_or("message");
            event.event(event_type)
        }
    }
}

struct Targets {
    models: BTreeMap<String, ModelId>,
    by_model: HashMap<ModelId, Target>,
}

fn targets(
    route: &Model,
    config: &AlgorithmSpec,
    model_list: &[Model],
    deployments: &ModelRouter,
) -> Result<Targets, SwitchyardConfigError> {
    let mut models = BTreeMap::new();
    let mut by_model = HashMap::new();
    for target in config.callable_target_names() {
        let configured = model_list
            .iter()
            .rev()
            .find(|model| model.model_name == target)
            .ok_or_else(|| invalid(route, format!("target {target:?} is not configured")))?;
        let deployment = deployments
            .get(target)
            .ok_or_else(|| invalid(route, format!("target {target:?} is not configured")))?;
        if deployment.model == SWITCHYARD_MODEL {
            return Err(invalid(
                route,
                format!("target {target:?} is another Switchyard route"),
            ));
        }
        let model = ModelId::from(target);
        models.insert(target.to_string(), model.clone());
        by_model.insert(
            model,
            Target {
                deployment: deployment.clone(),
                wire_format: target_wire_format(route, configured)?,
            },
        );
    }
    Ok(Targets { models, by_model })
}

// Caller format and target execution format are independent, so each target declares its own.
fn target_wire_format(route: &Model, target: &Model) -> Result<WireFormat, SwitchyardConfigError> {
    let value = target
        .litellm_params
        .extra
        .get("switchyard_wire_format")
        .ok_or_else(|| {
            invalid(
                route,
                format!(
                    "target {:?} requires switchyard_wire_format",
                    target.model_name
                ),
            )
        })?;
    WireFormat::deserialize(value).map_err(|error| {
        invalid(
            route,
            format!(
                "target {:?} has invalid switchyard_wire_format: {error}",
                target.model_name,
            ),
        )
    })
}

fn invalid(model: &Model, message: impl Into<String>) -> SwitchyardConfigError {
    SwitchyardConfigError {
        model: model.model_name.clone(),
        reason: message.into(),
    }
}

fn map_route_error(error: RouteError) -> LlmClientError {
    match error {
        RouteError::Transport(TransportError::Http { status, body }) => {
            match axum::http::StatusCode::from_u16(status) {
                Ok(status) => LlmClientError::UpstreamHttp { status, body },
                Err(source) => LlmClientError::InvalidResponse {
                    source: Box::new(source),
                },
            }
        }
        // Only Connect is known to precede dispatch, so only it is safe to fall back from.
        error @ RouteError::Transport(TransportError::Connect(_)) => LlmClientError::Transport {
            source: Box::new(error),
        },
        error @ RouteError::InvalidResponse(_) => LlmClientError::InvalidResponse {
            source: Box::new(error),
        },
        error => LlmClientError::Host {
            source: Box::new(error),
        },
    }
}

fn map_libsy_error(error: LibsyError) -> Error {
    match error {
        LibsyError::ClientCall { source, .. } => map_client_error(source),
        error => Error::Internal(error.to_string()),
    }
}

fn map_client_error(error: LlmClientError) -> Error {
    match error {
        LlmClientError::InvalidRequest { message }
        | LlmClientError::RequestTranslation(message) => {
            Error::Route(RouteError::InvalidRequest(message.into()))
        }
        LlmClientError::UpstreamHttp { status, body } => {
            Error::Route(RouteError::Transport(TransportError::Http {
                status: status.as_u16(),
                body,
            }))
        }
        LlmClientError::Transport { source }
        | LlmClientError::Timeout { source }
        | LlmClientError::InvalidResponse { source }
        | LlmClientError::Host { source }
        | LlmClientError::Ffi { source } => map_source(source),
        LlmClientError::ContextWindowExceeded { message, .. } => {
            Error::Route(RouteError::InvalidRequest(message.into()))
        }
        LlmClientError::ResponseTranslation(message) => {
            Error::Route(RouteError::InvalidResponse(message.into()))
        }
        error => Error::Internal(error.to_string()),
    }
}

fn map_source(source: BoxError) -> Error {
    match source.downcast::<RouteError>() {
        Ok(error) => Error::Route(*error),
        Err(source) => Error::Internal(source.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn host_errors_keep_their_gateway_classification() {
        let error = RouteError::Auth(litellm_auth::Error::MissingApiKey {
            provider: "OpenAI",
            environment_variable: "OPENAI_API_KEY",
        });

        assert!(matches!(
            map_client_error(map_route_error(error)),
            Error::Route(RouteError::Auth(_))
        ));
    }

    #[test]
    fn targets_require_an_explicit_wire_format() {
        let config = litellm_config::Config::from_yaml(
            "model_list:\n  - model_name: target\n    litellm_params: { model: upstream }\n  - model_name: route\n    litellm_params: { model: switchyard/algorithm }\n",
        )
        .expect("config");

        let error = target_wire_format(&config.model_list[1], &config.model_list[0])
            .expect_err("ambiguous target format");
        assert!(error.reason.contains("requires switchyard_wire_format"));
    }

    #[rstest]
    #[tokio::test]
    async fn a_stream_error_is_terminal() {
        let events: RawEventStream = Box::pin(stream::iter([
            Ok(json!({"chunk": "before"})),
            Err(LlmStreamError::Upstream(json!({"error": "failed"}))),
            Ok(json!({"chunk": "after"})),
        ]));

        let body = to_bytes(
            stream_response(events, WireFormat::OpenAiChat).into_body(),
            4096,
        )
        .await
        .expect("read stream");
        let body = std::str::from_utf8(&body).expect("stream is UTF-8");

        assert!(body.contains("before"));
        assert!(body.contains("failed"));
        assert!(!body.contains("after"));
        assert!(!body.contains("[DONE]"));
    }
}
