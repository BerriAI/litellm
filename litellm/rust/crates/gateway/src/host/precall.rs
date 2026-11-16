//! The facts `enable_pre_call_checks` decides by (`litellm/router_rust/steps/precall.py`), answered
//! from the gateway's model table: each candidate's context window as `get_router_model_info`
//! merges it, and the prompt's token count. No requested param is checked: the gateway cannot
//! tell which params a provider takes, and Python keeps a deployment it cannot resolve.

use std::{collections::BTreeMap, sync::Arc};

use litellm_core_utils::get_llm_provider_logic::get_custom_llm_provider;
use litellm_model_catalog::Catalog;
use litellm_router_core::{
    Deployment, RoutingContext,
    model_info::RouterModelInfoQuery,
    precall::{DeploymentFacts, PreCallFacts},
};
use litellm_token_counter::{CountableRequest, TokenCounter};
use serde_json::{Value, json};

use super::NativeModelInfoHost;
use crate::Gateway;

/// The tokenizer `litellm.token_counter` uses for a model it does not know.
const DEFAULT_ENCODING: &str = "cl100k_base";

fn base_model(deployment: &Deployment) -> Value {
    deployment
        .model_info
        .raw
        .get("base_model")
        .cloned()
        .unwrap_or(Value::Null)
}

/// `_deployment_max_input_tokens`: the merged context window, or `None` when there is none or
/// the deployment cannot be resolved.
async fn max_input_tokens(
    gateway: &Gateway,
    catalog: &Catalog,
    deployment: &Deployment,
    model_group: &str,
) -> Option<i64> {
    let query = RouterModelInfoQuery {
        deployment_id: Value::String(deployment.id.clone()),
        model_info_base_model: base_model(deployment),
        params_base_model: deployment
            .raw_params
            .get("base_model")
            .cloned()
            .unwrap_or(Value::Null),
        params_model: deployment.params.model.clone(),
        extra_base_model: deployment
            .params
            .extra
            .get("base_model")
            .cloned()
            .unwrap_or(Value::Null),
        received_model_name: model_group.to_owned(),
    };
    let host = NativeModelInfoHost {
        catalog,
        deployment,
    };
    match gateway.router.router_model_info(&host, &query).await {
        Ok(merged) => merged?.info.get("max_input_tokens")?.as_i64(),
        Err(error) => {
            tracing::debug!(%error, id = %deployment.id, "skipping the deployment's context window");
            None
        }
    }
}

/// The OpenAI encoding every deployment of the group shares, else the default one
/// (`single_encoding_openai_model`).
fn encoding(group: &[Arc<Deployment>]) -> &'static str {
    let served = group.iter().map(|deployment| {
        let base = base_model(deployment);
        let model = base.as_str().map(str::to_owned).unwrap_or_else(|| {
            let params = &deployment.params;
            get_custom_llm_provider(&params.model, params.custom_llm_provider.as_deref())
                .map_or(params.model.clone(), |resolved| resolved.model.to_owned())
        });
        litellm_token_counter::tiktoken::encoding_for_model(&model)
    });
    let mut encodings = served.collect::<Option<Vec<_>>>().unwrap_or_default();
    encodings.dedup();
    match encodings.as_slice() {
        [single] => single,
        _ => DEFAULT_ENCODING,
    }
}

/// A Responses `input` item as the chat message `transform_responses_api_input_to_messages` makes
/// of it, for the item kinds that carry text.
fn input_message(item: &Value) -> Option<Value> {
    if let Value::String(text) = item {
        return Some(json!({"role": "user", "content": text}));
    }
    let kind = item.get("type").and_then(Value::as_str);
    match kind {
        Some("message") | None => Some(json!({
            "role": item.get("role").cloned().unwrap_or(json!("user")),
            "content": item.get("content").cloned().unwrap_or(Value::Null),
        })),
        Some("function_call") => Some(json!({
            "role": "assistant",
            "content": item.get("arguments").cloned().unwrap_or(Value::Null),
        })),
        Some("function_call_output") => Some(json!({
            "role": "tool",
            "content": item.get("output").cloned().unwrap_or(Value::Null),
        })),
        _ => None,
    }
}

/// `_count_pre_call_check_tokens`: the messages, an Anthropic `system` block first, or the
/// Responses input as messages after its `instructions`, with the tools.
fn countable(body: &Value) -> Option<Value> {
    let system = body.get("system").or_else(|| body.get("instructions"));
    let leading = system
        .filter(|system| !system.is_null())
        .map(|system| json!({"role": "system", "content": system}));
    let messages: Vec<Value> = match (body.get("messages"), body.get("input")) {
        (Some(Value::Array(messages)), _) => messages.clone(),
        (_, Some(Value::Array(items))) => items.iter().filter_map(input_message).collect(),
        (_, Some(text @ Value::String(_))) => input_message(text).into_iter().collect(),
        _ => return None,
    };
    let mut request = json!({"messages": leading.into_iter().chain(messages).collect::<Vec<_>>()});
    if let Some(tools) = body
        .get("tools")
        .filter(|tools| tools.as_array().is_some_and(|tools| !tools.is_empty()))
    {
        request["tools"] = tools.clone();
    }
    Some(request)
}

fn count(body: &Value, encoding: &str) -> Option<i64> {
    let request = CountableRequest::parse(&serde_json::to_vec(&countable(body)?).ok()?).ok()?;
    let counter = TokenCounter::from_tiktoken(encoding).ok()?;
    let counted = counter.count_request(&request);
    match counted {
        Ok(counted) => i64::try_from(counted.input_tokens).ok(),
        Err(error) => {
            tracing::error!(%error, "litellm.router.py::_acount_pre_call_check_tokens: failed to count tokens");
            None
        }
    }
}

pub(super) async fn facts(
    gateway: &Gateway,
    ctx: &RoutingContext<'_>,
    candidates: &[Arc<Deployment>],
) -> PreCallFacts {
    let body = &ctx.request.body;
    let has_input = ["messages", "input"]
        .iter()
        .any(|field| body.get(field).is_some_and(|value| !value.is_null()));
    if !has_input {
        return PreCallFacts::default();
    }
    let models = gateway.models();
    let catalog = models.cost_map.catalog();
    let mut deployments = BTreeMap::new();
    for deployment in candidates {
        let window = max_input_tokens(gateway, catalog, deployment, ctx.model_group).await;
        deployments.insert(
            deployment.id.clone(),
            DeploymentFacts {
                max_input_tokens: window,
                ..DeploymentFacts::default()
            },
        );
    }
    let needs_count = deployments
        .values()
        .any(|facts| facts.max_input_tokens.is_some());
    let input_tokens = if needs_count {
        let group = ctx.registry.group(ctx.model_group).unwrap_or_default();
        let encoding = encoding(if group.is_empty() { candidates } else { &group });
        let body = body.clone();
        tokio::task::spawn_blocking(move || count(&body, encoding))
            .await
            .ok()
            .flatten()
    } else {
        None
    };
    PreCallFacts {
        has_input: true,
        countable: true,
        input_tokens,
        allowed_model_region: body.get("allowed_model_region").cloned(),
        requested_params: Vec::new(),
        deployments,
    }
}
