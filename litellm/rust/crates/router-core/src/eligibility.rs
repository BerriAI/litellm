//! Which deployments may serve a request's endpoint: a port of
//! `litellm/router_utils/endpoint_eligibility.py`. A deployment that sets
//! `enforce_supported_endpoints` and lists none of the chat-shaped endpoints serves none of them;
//! the host names the endpoint for the calls `endpoint_selection` wraps.

use std::{collections::BTreeSet, sync::Arc};

use serde_json::{Map, Value};

use crate::{error::Error, registry::Deployment, resolve::Caller};

/// `CHAT_ENDPOINTS`: `_endpoint_path` of each value of `CHAT_ENDPOINT_BY_CALL`.
const CHAT_ENDPOINTS: [&str; 8] = [
    "/chat/completions",
    "/completions",
    "/messages",
    "/responses",
    "/generatecontent",
    "/streamgeneratecontent",
    "/interactions",
    "/responses/compact",
];

/// `_endpoint_path`.
fn endpoint_path(endpoint: &str) -> String {
    let rooted = format!("/{}", endpoint.trim().trim_matches('/').to_lowercase());
    let path = rooted.strip_prefix("/openai").unwrap_or(&rooted);
    let path = path.strip_prefix("/v1beta").unwrap_or(path);
    let path = path.strip_prefix("/v1").unwrap_or(path);
    match path.rsplit_once(':') {
        Some((_, action)) => format!("/{action}"),
        None => path.to_owned(),
    }
}

/// The flag as pydantic's lax `bool | None` reads it: `None` for a value it refuses, which fails
/// the whole declaration.
fn enforce_flag(value: Option<&Value>) -> Option<bool> {
    match value {
        None | Some(Value::Null) => Some(false),
        Some(Value::Bool(flag)) => Some(*flag),
        Some(Value::Number(number)) => match number.as_f64() {
            Some(0.0) => Some(false),
            Some(1.0) => Some(true),
            _ => None,
        },
        Some(Value::String(text)) => match text.to_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
            "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
            _ => None,
        },
        Some(_) => None,
    }
}

/// `_enforced_endpoints`: the declared endpoints of a deployment that enforces them. One endpoint
/// may be written as a string; a declaration that does not validate enforces nothing.
fn enforced_endpoints(model_info: &Map<String, Value>) -> Option<Vec<String>> {
    if !enforce_flag(model_info.get("enforce_supported_endpoints"))? {
        return None;
    }
    match model_info.get("supported_endpoints")? {
        Value::String(endpoint) => Some(vec![endpoint.clone()]),
        Value::Array(endpoints) => endpoints
            .iter()
            .map(|endpoint| endpoint.as_str().map(str::to_owned))
            .collect(),
        _ => None,
    }
}

/// `non_chat_endpoints`: what a deployment serves when it serves no chat-shaped endpoint, else
/// `None`.
fn non_chat_endpoints(deployment: &Deployment) -> Option<BTreeSet<String>> {
    let declared = enforced_endpoints(&deployment.model_info.raw)?;
    let chat = declared
        .iter()
        .any(|endpoint| CHAT_ENDPOINTS.contains(&endpoint_path(endpoint).as_str()));
    (!chat).then(|| declared.into_iter().collect())
}

/// `eligible_deployments`: the deployments that may serve the caller's endpoint. When none may,
/// the refusal names what they do serve.
pub(crate) fn eligible_deployments(
    model: &str,
    caller: Caller<'_>,
    deployments: Vec<Arc<Deployment>>,
) -> Result<Vec<Arc<Deployment>>, Error> {
    let Some(endpoint) = caller.endpoint.filter(|_| !deployments.is_empty()) else {
        return Ok(deployments);
    };
    let declared: Vec<Option<BTreeSet<String>>> = deployments
        .iter()
        .map(|deployment| non_chat_endpoints(deployment))
        .collect();
    let eligible: Vec<Arc<Deployment>> = deployments
        .into_iter()
        .zip(&declared)
        .filter(|(_, endpoints)| endpoints.is_none())
        .map(|(deployment, _)| deployment)
        .collect();
    if !eligible.is_empty() {
        return Ok(eligible);
    }
    let supported: BTreeSet<String> = declared.into_iter().flatten().flatten().collect();
    let supported = if supported.is_empty() {
        "none".to_owned()
    } else {
        Vec::from_iter(supported).join(", ")
    };
    let requested = if caller.requested.is_empty() {
        model
    } else {
        caller.requested
    };
    Err(Error::EndpointNotSupported {
        model: model.to_owned(),
        message: format!(
            "{endpoint} is not supported by model={requested}. Supported endpoints: {supported}"
        ),
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::registry::DeploymentSpec;

    fn deployment(id: &str, info: Value) -> Arc<Deployment> {
        let mut model_info = info;
        model_info["id"] = json!(id);
        let spec: DeploymentSpec = serde_json::from_value(json!({
            "model_name": "g", "litellm_params": {"model": "openai/a"}, "model_info": model_info,
        }))
        .unwrap();
        Arc::new(Deployment::from_spec(spec).unwrap())
    }

    fn chat<'a>() -> Caller<'a> {
        Caller {
            endpoint: Some("/v1/chat/completions"),
            requested: "g",
            ..Caller::default()
        }
    }

    #[rstest]
    #[case::v1_prefix("/v1/chat/completions", "/chat/completions")]
    #[case::openai_prefix("/openai/v1/responses/", "/responses")]
    #[case::bare_gemini_action("generateContent", "/generatecontent")]
    #[case::beta_prefix("/v1beta/interactions", "/interactions")]
    #[case::gemini_model_path(
        "/v1beta/models/gemini:streamGenerateContent",
        "/streamgeneratecontent"
    )]
    fn endpoint_paths_normalize(#[case] endpoint: &str, #[case] expected: &str) {
        assert_eq!(endpoint_path(endpoint), expected);
    }

    #[rstest]
    #[case::embeddings_only(
        json!({"enforce_supported_endpoints": true, "supported_endpoints": ["/v1/embeddings"]}),
        false
    )]
    #[case::one_endpoint_as_a_string(
        json!({"enforce_supported_endpoints": true, "supported_endpoints": "/v1/embeddings"}),
        false
    )]
    #[case::string_flag(
        json!({"enforce_supported_endpoints": "yes", "supported_endpoints": ["/v1/embeddings"]}),
        false
    )]
    #[case::an_empty_list_serves_nothing(
        json!({"enforce_supported_endpoints": true, "supported_endpoints": []}),
        false
    )]
    #[case::any_chat_shaped_endpoint_serves_every_one(
        json!({"enforce_supported_endpoints": true, "supported_endpoints": ["/v1/responses"]}),
        true
    )]
    #[case::not_enforced(
        json!({"enforce_supported_endpoints": false, "supported_endpoints": ["/v1/embeddings"]}),
        true
    )]
    #[case::unset_flag(json!({"supported_endpoints": ["/v1/embeddings"]}), true)]
    #[case::enforced_without_a_list(json!({"enforce_supported_endpoints": true}), true)]
    #[case::an_invalid_flag_enforces_nothing(
        json!({"enforce_supported_endpoints": "maybe", "supported_endpoints": ["/v1/embeddings"]}),
        true
    )]
    #[case::an_invalid_list_enforces_nothing(
        json!({"enforce_supported_endpoints": true, "supported_endpoints": [1]}),
        true
    )]
    fn a_declaration_keeps_or_drops_a_deployment(#[case] info: Value, #[case] kept: bool) {
        let open = deployment("open", json!({}));
        let declared = deployment("declared", info);
        let eligible = eligible_deployments("g", chat(), vec![declared, open]).unwrap();
        assert_eq!(eligible.len(), if kept { 2 } else { 1 });
    }

    #[test]
    fn no_endpoint_keeps_everything() {
        let declared = deployment(
            "d",
            json!({"enforce_supported_endpoints": true, "supported_endpoints": []}),
        );
        let eligible = eligible_deployments("g", Caller::default(), vec![declared]).unwrap();
        assert_eq!(eligible.len(), 1);
    }

    #[rstest]
    #[case::lists_what_they_serve(
        json!(["/v1/rerank", "/v1/embeddings"]),
        "/v1/chat/completions is not supported by model=g. \
         Supported endpoints: /v1/embeddings, /v1/rerank"
    )]
    #[case::nothing_declared(
        json!([]),
        "/v1/chat/completions is not supported by model=g. Supported endpoints: none"
    )]
    fn every_deployment_refusing_names_the_endpoint(
        #[case] endpoints: Value,
        #[case] expected: &str,
    ) {
        let declared = deployment(
            "d",
            json!({"enforce_supported_endpoints": true, "supported_endpoints": endpoints}),
        );
        let refused = eligible_deployments("resolved", chat(), vec![declared]);
        assert!(matches!(
            refused,
            Err(Error::EndpointNotSupported { model, message })
                if model == "resolved" && message == expected
        ));
    }
}
