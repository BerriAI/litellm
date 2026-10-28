use std::{
    collections::{HashMap, HashSet},
    future::Future,
    pin::Pin,
    sync::Arc,
};

use axum::{
    Json,
    extract::{Path, State},
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use litellm_cache_response::CacheOptions;
use litellm_config::Model;
use litellm_gateway_auth::AuthenticatedRequest;
use serde_json::{Map, Value};

use crate::{AutoRoutingConfigError, Error, Gateway, JsonObject, ModelRouter, RequestId, request};

#[cfg(feature = "switchyard")]
mod switchyard;

#[derive(Clone, Copy, strum::VariantArray)]
pub enum ApiFormat {
    Chat,
    Responses,
    Messages,
}

pub struct StrategyRequest {
    pub format: ApiFormat,
    pub body: Map<String, Value>,
    pub headers: HeaderMap,
    pub cache_options: CacheOptions,
}

pub trait AutoRoutingStrategy: Send + Sync {
    fn execute<'a>(
        &'a self,
        gateway: &'a Gateway,
        request: StrategyRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Response, Error>> + Send + 'a>>;
}

#[derive(Default)]
pub(crate) struct Registry(HashMap<String, Arc<dyn AutoRoutingStrategy>>);

impl Registry {
    pub(crate) fn insert(&mut self, model: String, strategy: Arc<dyn AutoRoutingStrategy>) {
        self.0.insert(model, strategy);
    }

    pub(crate) fn from_models(
        models: &[Model],
        deployments: &ModelRouter,
    ) -> Result<Self, AutoRoutingConfigError> {
        let mut seen = HashSet::new();
        let mut strategies = HashMap::new();
        for model in models.iter().rev() {
            if !seen.insert(&model.model_name)
                || model.litellm_params.model != "switchyard/algorithm"
            {
                continue;
            }
            if deployments
                .get(&model.model_name)
                .is_none_or(|deployment| deployment.model != model.litellm_params.model)
            {
                return Err(AutoRoutingConfigError {
                    model: model.model_name.clone(),
                    reason: "model does not match its configured deployment".into(),
                });
            }
            strategies.insert(
                model.model_name.clone(),
                build_strategy(model, models, deployments)?,
            );
        }
        Ok(Self(strategies))
    }

    fn get(&self, body: &Map<String, Value>) -> Option<&dyn AutoRoutingStrategy> {
        self.0.get(body.get("model")?.as_str()?).map(AsRef::as_ref)
    }
}

#[cfg(feature = "switchyard")]
fn build_strategy(
    model: &Model,
    models: &[Model],
    deployments: &ModelRouter,
) -> Result<Arc<dyn AutoRoutingStrategy>, AutoRoutingConfigError> {
    Ok(Arc::new(switchyard::SwitchyardStrategy::from_model(
        model,
        models,
        deployments,
    )?))
}

#[cfg(not(feature = "switchyard"))]
fn build_strategy(
    model: &Model,
    _: &[Model],
    _: &ModelRouter,
) -> Result<Arc<dyn AutoRoutingStrategy>, AutoRoutingConfigError> {
    Err(AutoRoutingConfigError {
        model: model.model_name.clone(),
        reason: "this build does not enable the switchyard feature".into(),
    })
}

async fn execute(
    strategy: &dyn AutoRoutingStrategy,
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    format: ApiFormat,
    headers: HeaderMap,
    body: Map<String, Value>,
) -> Result<Response, Error> {
    let deployment = request::resolve_deployment(gateway, &body)?;
    request::authorize_model(identity, deployment, &body).await?;
    let (body, cache_options) = crate::caching::prepare(identity, body)?;
    strategy
        .execute(
            gateway,
            StrategyRequest {
                format,
                body,
                headers,
                cache_options,
            },
        )
        .await
}

pub(crate) async fn chat(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    headers: HeaderMap,
    JsonObject(body): JsonObject,
) -> Result<Response, Error> {
    if let Some(strategy) = gateway.auto_routing.get(&body) {
        return execute(
            strategy,
            &gateway,
            &identity,
            ApiFormat::Chat,
            headers,
            body,
        )
        .await;
    }
    crate::chat_completions::create(State(gateway), identity, JsonObject(body))
        .await
        .map(IntoResponse::into_response)
}

pub(crate) async fn chat_from_model_path(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    headers: HeaderMap,
    Path(model): Path<String>,
    JsonObject(mut body): JsonObject,
) -> Result<Response, Error> {
    if body.get("model").is_none_or(Value::is_null) {
        body.insert("model".into(), Value::String(model.clone()));
    }
    if let Some(strategy) = gateway.auto_routing.get(&body) {
        return execute(
            strategy,
            &gateway,
            &identity,
            ApiFormat::Chat,
            headers,
            body,
        )
        .await;
    }
    crate::chat_completions::create_from_model_path(
        State(gateway),
        identity,
        Path(model),
        JsonObject(body),
    )
    .await
    .map(IntoResponse::into_response)
}

pub(crate) async fn responses(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    headers: HeaderMap,
    JsonObject(body): JsonObject,
) -> Result<Response, Error> {
    if let Some(strategy) = gateway.auto_routing.get(&body) {
        return execute(
            strategy,
            &gateway,
            &identity,
            ApiFormat::Responses,
            headers,
            body,
        )
        .await;
    }
    crate::responses::create(State(gateway), identity, JsonObject(body)).await
}

pub(crate) async fn messages(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    RequestId(request_id): RequestId,
    headers: HeaderMap,
    body: Result<JsonObject, Error>,
) -> Response {
    match body {
        Ok(JsonObject(body)) => {
            if let Some(strategy) = gateway.auto_routing.get(&body) {
                return execute(
                    strategy,
                    &gateway,
                    &identity,
                    ApiFormat::Messages,
                    headers,
                    body,
                )
                .await
                .unwrap_or_else(|error| {
                    (error.status(), Json(error.body(request_id.as_deref()))).into_response()
                });
            }
            crate::messages::create(
                State(gateway),
                identity,
                RequestId(request_id),
                headers,
                Ok(JsonObject(body)),
            )
            .await
            .into_response()
        }
        Err(error) => crate::messages::create(
            State(gateway),
            identity,
            RequestId(request_id),
            headers,
            Err(error),
        )
        .await
        .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use litellm_config::Config;
    use rstest::rstest;

    #[rstest]
    fn overwritten_virtual_models_do_not_initialize_a_strategy() {
        let config = Config::from_yaml(
            r#"
model_list:
  - model_name: public
    litellm_params: {model: switchyard/algorithm}
  - model_name: public
    litellm_params: {model: openai/model}
"#,
        )
        .unwrap();
        let registry = Registry::from_models(
            &config.model_list,
            &ModelRouter::from_model_list(&config.model_list),
        )
        .unwrap();
        assert!(
            registry
                .get(
                    &serde_json::json!({"model":"public"})
                        .as_object()
                        .unwrap()
                        .clone()
                )
                .is_none()
        );
    }

    #[rstest]
    fn strategy_configuration_must_match_the_deployment_registry() {
        let config = Config::from_yaml(
            r#"
model_list:
  - model_name: public
    litellm_params: {model: switchyard/algorithm}
"#,
        )
        .unwrap();
        let error = Registry::from_models(&config.model_list, &ModelRouter::default())
            .err()
            .unwrap();
        assert!(error.to_string().contains("does not match"));
    }

    #[cfg(not(feature = "switchyard"))]
    #[rstest]
    fn disabled_strategies_fail_at_configuration_time() {
        let config = Config::from_yaml(
            r#"
model_list:
  - model_name: public
    litellm_params: {model: switchyard/algorithm}
"#,
        )
        .unwrap();
        let error = Registry::from_models(
            &config.model_list,
            &ModelRouter::from_model_list(&config.model_list),
        )
        .err()
        .unwrap();
        assert!(
            error
                .to_string()
                .contains("does not enable the switchyard feature")
        );
    }
}
