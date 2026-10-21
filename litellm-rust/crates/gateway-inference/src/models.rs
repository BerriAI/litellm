//! `GET /v1/models` and `GET /v1/models/{model_id}`: the model groups a caller may use, in the
//! OpenAI shape the proxy's `model_list` and `model_info` answer with.

use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, State},
};
use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_gateway_auth::{AuthenticatedRequest, AuthenticationMethod};
use litellm_router::{Deployment, ModelGroup};
use serde::Serialize;

use crate::{Error, Gateway, request};

/// The proxy's `DEFAULT_MODEL_CREATED_AT_TIME`.
const CREATED: u64 = 1_677_610_602;

#[derive(Debug, PartialEq, Serialize)]
pub(crate) struct Model {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: String,
}

impl Model {
    fn new(id: &str, owned_by: &str) -> Self {
        Self {
            id: id.to_owned(),
            object: "model",
            created: CREATED,
            owned_by: owned_by.to_owned(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct ModelList {
    data: Vec<Model>,
    object: &'static str,
}

/// One entry per model group the caller may call, in config order. The listing names every
/// model's owner `openai`, as the proxy's does.
pub(crate) async fn list(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
) -> Json<ModelList> {
    let mut data = Vec::new();
    for group in gateway.models.groups() {
        if visible(&identity, group, true).await.is_some() {
            data.push(Model::new(group.name(), "openai"));
        }
    }
    Json(ModelList {
        data,
        object: "list",
    })
}

/// The model `model_id` names when the caller may call it, owned by its deployment's provider.
pub(crate) async fn retrieve(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    Path(model_id): Path<String>,
) -> Result<Json<Model>, Error> {
    let Some(group) = gateway.models.group(&model_id) else {
        return Err(Error::ModelNotFound(model_id));
    };
    match visible(&identity, group, false).await {
        Some(deployment) => Ok(Json(Model::new(&model_id, provider(deployment)))),
        None => Err(Error::ModelNotFound(model_id)),
    }
}

/// The first deployment of `group` the caller may be routed to and is authorized for, as a
/// request naming the group would be. When `listing`, only the master key sees a group every
/// such deployment marks `discoverable: false`; any caller allowed it still retrieves it by
/// name, as with the proxy. A wildcard group is never shown: the gateway routes model names exactly, so no
/// request reaches it under another name.
async fn visible<'a>(
    identity: &AuthenticatedRequest,
    group: &'a ModelGroup,
    listing: bool,
) -> Option<&'a Deployment> {
    if group.name().contains('*') {
        return None;
    }
    let hide_undiscoverable =
        listing && identity.caller().authentication().method != AuthenticationMethod::MasterKey;
    for deployment in group.deployments() {
        if !identity.may_use_deployment(deployment.team_id.as_deref())
            || (deployment.undiscoverable && hide_undiscoverable)
        {
            continue;
        }
        if request::authorize(identity, group.name(), deployment)
            .await
            .is_ok()
        {
            return Some(deployment);
        }
    }
    None
}

/// The deployment's provider: its `custom_llm_provider`, else its model's prefix, else
/// `openai`, which the proxy infers for a bare model name.
fn provider(deployment: &Deployment) -> &str {
    get_custom_llm_provider(&deployment.model, deployment.custom_llm_provider.as_deref())
        .map_or("openai", |resolved| resolved.custom_llm_provider)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::prefix("anthropic/claude-x", None, "anthropic")]
    #[case::custom_provider("claude-x", Some("bedrock"), "bedrock")]
    #[case::custom_provider_wins("openai/gpt-x", Some("azure"), "azure")]
    #[case::bare_name("gpt-x", None, "openai")]
    fn names_the_deployments_provider(
        #[case] model: &str,
        #[case] custom_llm_provider: Option<&str>,
        #[case] expected: &str,
    ) {
        let deployment = Deployment {
            model: model.to_owned(),
            custom_llm_provider: custom_llm_provider.map(str::to_owned),
            ..Deployment::default()
        };

        assert_eq!(provider(&deployment), expected);
    }

    #[test]
    fn serializes_the_openai_model_shape() {
        assert_eq!(
            serde_json::to_value(Model::new("claude-a", "openai")).unwrap(),
            serde_json::json!({
                "id": "claude-a",
                "object": "model",
                "created": 1_677_610_602,
                "owned_by": "openai",
            })
        );
    }
}
