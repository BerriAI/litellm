use std::{
    collections::{BTreeMap, HashMap, HashSet},
    sync::Arc,
};

use async_trait::async_trait;
use axum::http::HeaderMap;
use litellm_cache_response::CachePolicy;
use litellm_config::Model;
use litellm_inference::RouteError;
use litellm_inference_chat::ChatCompletionsRoute;
use litellm_http::transport::Error as TransportError;
use serde::Deserialize;
use serde_json::{Map, Value};
use switchyard_libsy::{Algorithm, LibsyError, RuntimeModels};
use switchyard_llm_client::ClientRouter;
use switchyard_protocol::{
    BoxError, LlmClientError, LlmResponse, Metadata, ModelId, Request, Response, RoutedLlmClient,
    WireFormat,
};
use switchyard_runner::AlgorithmSpec;
use switchyard_translation::{
    decode_aggregated_response, decode_request, encode_aggregated_response_with_extensions,
    encode_request,
};

use crate::{
    Error, ModelRouter, SwitchyardConfigError, caching::CacheHeaders,
    chat_completions::provider_request,
};

const SWITCHYARD_MODEL: &str = "switchyard/algorithm";

#[derive(Default)]
pub(crate) struct Routes(HashMap<String, Route>);

impl Routes {
    pub(crate) fn from_models(
        model_list: &[Model],
        deployments: Arc<ModelRouter>,
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
            let targets = targets(model, &config, &deployments)?;
            let (algorithm, runtime_models) = config
                .build_with_runtime_models(&model.model_name, &targets)
                .map_err(|error| invalid(model, error.to_string()))?;
            routes.insert(
                model.model_name.clone(),
                Route {
                    algorithm,
                    runtime_models: Arc::new(runtime_models),
                    deployments: Arc::clone(&deployments),
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
    deployments: Arc<ModelRouter>,
}

impl Route {
    pub(crate) async fn execute(
        &self,
        route: ChatCompletionsRoute,
        cache_policy: CachePolicy,
        body: Map<String, Value>,
        headers: HeaderMap,
    ) -> Result<(Value, HeaderMap), Error> {
        let llm_request = decode_request(WireFormat::OpenAiChat, &Value::Object(body))
            .map_err(|error| Error::InvalidBody(error.to_string()))?;
        // Reject unsupported streaming before a classifier or judge can make a model call.
        if llm_request.stream {
            return Err(Error::Route(RouteError::Unsupported("streaming")));
        }
        let extensions = llm_request.extensions.clone();
        let mut metadata = Metadata::from_headers(&headers);
        metadata.http_headers = Some(headers);
        metadata.wire_format = Some(WireFormat::OpenAiChat);
        let request = Request {
            llm_request,
            raw_request: None,
            metadata: Some(metadata),
        };
        let client = Arc::new(LiteLlmClient {
            route,
            deployments: Arc::clone(&self.deployments),
            cache_policy,
        });
        let (_, response) = switchyard_llm_client::run(
            Arc::clone(&self.algorithm),
            ClientRouter::single(client),
            request,
            Arc::clone(&self.runtime_models),
            None,
        )
        .await
        .map_err(map_libsy_error)?;
        let upstream_headers = response.upstream_headers;
        let response = response
            .llm_response
            .into_agg()
            .await
            .map_err(map_client_error)?;
        let body = encode_aggregated_response_with_extensions(
            &response,
            WireFormat::OpenAiChat,
            None,
            &extensions,
        )
        .map_err(|error| Error::Internal(error.to_string()))?;
        Ok((body, upstream_headers))
    }
}

struct LiteLlmClient {
    route: ChatCompletionsRoute,
    deployments: Arc<ModelRouter>,
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
        let deployment =
            self.deployments
                .get(model)
                .ok_or_else(|| LlmClientError::Configuration {
                    message: format!("unknown LiteLLM deployment {model}"),
                })?;
        let Request {
            llm_request,
            metadata,
            ..
        } = request;
        let Value::Object(body) = encode_request(&llm_request, WireFormat::OpenAiChat)
            .map_err(|error| LlmClientError::RequestEncoding(error.to_string()))?
        else {
            return Err(LlmClientError::RequestEncoding(
                "OpenAI Chat request is not an object".to_string(),
            ));
        };
        let cache_headers = CacheHeaders::default();
        let response = self
            .route
            .execute(
                provider_request(deployment, body),
                &cache_headers,
                self.cache_policy,
            )
            .await
            .map_err(map_route_error)?;
        let body = serde_json::to_value(response)
            .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
        let response = decode_aggregated_response(&body, WireFormat::OpenAiChat)
            .map_err(|error| LlmClientError::ResponseTranslation(error.to_string()))?;
        Ok(Response {
            llm_response: LlmResponse::Agg(response),
            metadata,
            upstream_headers: cache_headers.headers(),
        })
    }
}

fn targets(
    route: &Model,
    config: &AlgorithmSpec,
    deployments: &ModelRouter,
) -> Result<BTreeMap<String, ModelId>, SwitchyardConfigError> {
    config
        .callable_target_names()
        .into_iter()
        .map(|target| {
            let deployment = deployments
                .get(target)
                .ok_or_else(|| invalid(route, format!("target {target:?} is not configured")))?;
            if deployment.model == SWITCHYARD_MODEL {
                return Err(invalid(
                    route,
                    format!("target {target:?} is another Switchyard route"),
                ));
            }
            Ok((target.to_string(), ModelId::from(target)))
        })
        .collect()
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
}
