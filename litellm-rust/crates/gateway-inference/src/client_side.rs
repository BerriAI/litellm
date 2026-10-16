//! The proxy's `is_request_body_safe`: a request body may not choose where a call goes or
//! which credentials it uses, unless `general_settings.allow_client_side_credentials` lets it.

use std::borrow::Cow;

use litellm_gateway_auth::Error;
use serde_json::{Map, Value};

/// Python's `_BANNED_REQUEST_BODY_PARAMS`, in its order so a body carrying several names the
/// same one, followed by the connection and credential fields the Rust transforms read from a
/// body that Python's list leaves out.
const BANNED: &[&str] = &[
    "api_base",
    "base_url",
    "user_config",
    "aws_sts_endpoint",
    "aws_web_identity_token",
    "aws_role_name",
    "aws_profile_name",
    "aws_session_name",
    "aws_external_id",
    "aws_session_tags",
    "vertex_credentials",
    "azure_ad_token",
    "aws_bedrock_runtime_endpoint",
    "aws_bedrock_project_id",
    "workspace_id",
    "aws_workspace_id",
    "anthropic_workspace_id",
    "anthropic-workspace-id",
    "bedrock_tags",
    "agentcore_runtime_session_id",
    "s3_endpoint_url",
    "sagemaker_base_url",
    "deployment_url",
    "nvcf_function_id",
    "use_ssl",
    "rust",
    "fireworks_forward_user_id",
    "model_list",
    "vertex_ai_credentials",
    "github_copilot_auth_type",
    "user_provider_credentials",
    "github_copilot_user_session",
    // `_build_banned_observability_params()`, sorted as Python sorts it.
    "arize_api_key",
    "arize_error_sampling_rate",
    "arize_space_id",
    "arize_space_key",
    "arize_success_sampling_rate",
    "braintrust_api_key",
    "braintrust_host",
    "braintrust_project",
    "dd_agent_host",
    "dd_agent_port",
    "dd_api_key",
    "dd_site",
    "gcs_bucket_name",
    "gcs_path_service_account",
    "humanloop_api_key",
    "langfuse_environment",
    "langfuse_host",
    "langfuse_public_key",
    "langfuse_secret",
    "langfuse_secret_key",
    "langsmith_api_key",
    "langsmith_base_url",
    "langsmith_project",
    "langsmith_tenant_id",
    "lunary_public_key",
    "newrelic_api_key",
    "newrelic_region",
    "posthog_api_key",
    "posthog_api_url",
    "posthog_host",
    "signoz_ingestion_endpoint",
    "signoz_ingestion_key",
    "slack_webhook_url",
    "turn_off_message_logging",
    "user_api_key_auth_metadata",
    "wandb_api_key",
    "weave_project_id",
    // Read from the body by the Rust Bedrock chat, Bedrock transcription and Textract OCR
    // configs (`AwsParams::from_optional_params`): the region picks the endpoint host and the
    // rest pick the identity that signs the call.
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    // Read from the body by the Rust Azure OCR configs to pick the Entra identity.
    "tenant_id",
    "client_id",
    "client_secret",
    "azure_scope",
    "azure_authority_host",
    "azure_credential",
    "azure_federated_token_file",
    "enable_azure_ad_token_refresh",
    // Read from the body by the Rust Vertex OCR configs: the location picks the endpoint host.
    "vertex_project",
    "vertex_ai_project",
    "vertex_location",
    "vertex_ai_location",
];

/// Python's `_NESTED_CONFIG_KEYS` and `_NESTED_METADATA_KEYS`: dicts whose keys reach the
/// call or its callbacks the same way the body's own do, checked one level down.
const NESTED: [&str; 4] = [
    "litellm_embedding_config",
    "extra_body",
    "metadata",
    "litellm_metadata",
];

/// Refuses a body naming a banned param at its root or one level into [`NESTED`], with the
/// proxy's message and status. `model_list` is refused even when client-side credentials are
/// allowed, as Python refuses it before the opt-in is read.
pub(crate) fn check(
    body: &Map<String, Value>,
    allow_client_side_credentials: bool,
) -> Result<(), Error> {
    if body.contains_key("model_list") {
        return Err(Error::ModelListInBody);
    }
    if allow_client_side_credentials {
        return Ok(());
    }
    let nested = NESTED
        .iter()
        .filter_map(|key| body.get(*key).and_then(as_object));
    match std::iter::once(Cow::Borrowed(body))
        .chain(nested)
        .find_map(|fields| {
            BANNED
                .iter()
                .find(|param| fields.contains_key(**param))
                .copied()
        }) {
        Some(param) => Err(Error::BannedParam(param)),
        None => Ok(()),
    }
}

/// Python's `_coerce_metadata_to_dict`: a dict, or a string holding a JSON object.
fn as_object(value: &Value) -> Option<Cow<'_, Map<String, Value>>> {
    match value {
        Value::Object(fields) => Some(Cow::Borrowed(fields)),
        Value::String(text) => serde_json::from_str(text).ok().map(Cow::Owned),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn body(value: Value) -> Map<String, Value> {
        let Value::Object(fields) = value else {
            unreachable!("test bodies are objects")
        };
        fields
    }

    #[rstest]
    #[case::bedrock_endpoint(json!({"aws_bedrock_runtime_endpoint": "https://attacker.test"}), "aws_bedrock_runtime_endpoint")]
    #[case::api_base(json!({"api_base": "https://attacker.test"}), "api_base")]
    #[case::null_value_still_counts(json!({"base_url": null}), "base_url")]
    #[case::region(json!({"aws_region_name": "attacker.test#"}), "aws_region_name")]
    #[case::static_keys(json!({"aws_access_key_id": "a", "aws_secret_access_key": "b"}), "aws_access_key_id")]
    #[case::profile(json!({"aws_profile_name": "admin"}), "aws_profile_name")]
    #[case::azure_identity(json!({"client_secret": "s"}), "client_secret")]
    #[case::vertex_location(json!({"vertex_location": "attacker.test#"}), "vertex_location")]
    #[case::observability(json!({"langfuse_host": "https://attacker.test"}), "langfuse_host")]
    #[case::first_in_python_order(json!({"aws_role_name": "r", "api_base": "b"}), "api_base")]
    #[case::extra_body(json!({"extra_body": {"aws_web_identity_token": "t"}}), "aws_web_identity_token")]
    #[case::metadata(json!({"metadata": {"langfuse_host": "h"}}), "langfuse_host")]
    #[case::metadata_as_json_text(json!({"litellm_metadata": "{\"api_base\": \"h\"}"}), "api_base")]
    #[case::embedding_config(json!({"litellm_embedding_config": {"api_base": "h"}}), "api_base")]
    #[case::root_before_nested(json!({"extra_body": {"api_base": "h"}, "user_config": {}}), "user_config")]
    fn refuses_banned_params_with_the_proxy_error(#[case] request: Value, #[case] param: &str) {
        let error = check(&body(request), false).unwrap_err();
        assert!(
            matches!(error, Error::BannedParam(name) if name == param),
            "{error}"
        );
    }

    #[rstest]
    #[case::chat(json!({"model": "m", "messages": [], "temperature": 0.2, "user": "u"}))]
    #[case::messages_metadata(json!({"model": "m", "max_tokens": 8, "metadata": {"user_id": "u"}}))]
    #[case::extra_body_params(json!({"extra_body": {"top_k": 3}}))]
    #[case::unparsable_metadata_text(json!({"metadata": "not json"}))]
    #[case::nested_two_levels_down(json!({"extra_body": {"metadata": {"api_base": "h"}}}))]
    fn passes_ordinary_bodies(#[case] request: Value) {
        assert!(check(&body(request), false).is_ok());
    }

    #[rstest]
    #[case::opted_in_passes(true, json!({"api_base": "h", "aws_bedrock_runtime_endpoint": "h"}), false)]
    #[case::model_list_ignores_the_opt_in(true, json!({"model_list": []}), true)]
    #[case::model_list_precedes_others(false, json!({"api_base": "h", "model_list": []}), true)]
    fn honours_allow_client_side_credentials(
        #[case] allowed: bool,
        #[case] request: Value,
        #[case] refuses_model_list: bool,
    ) {
        let result = check(&body(request), allowed);
        assert_eq!(
            matches!(result, Err(Error::ModelListInBody)),
            refuses_model_list
        );
        assert_eq!(result.is_ok(), !refuses_model_list);
    }
}
