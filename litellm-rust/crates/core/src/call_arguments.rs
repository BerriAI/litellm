use std::ops::Deref;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::{Map, Value};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CallArguments(Map<String, Value>);

impl CallArguments {
    pub(crate) fn select(&self, names: &[&str]) -> Map<String, Value> {
        self.iter()
            .filter(|(name, _)| names.contains(&name.as_str()))
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("invalid argument: {path}")]
pub struct ArgumentError {
    pub path: String,
}

pub fn parse_options<T: DeserializeOwned>(arguments: &CallArguments) -> Result<T, ArgumentError> {
    use serde::de::IntoDeserializer;
    serde_path_to_error::deserialize(Value::Object(arguments.0.clone()).into_deserializer())
        .map_err(|error| ArgumentError {
            path: error.path().to_string(),
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArgumentSpec {
    pub name: &'static str,
    pub secret: bool,
}

pub fn should_project(name: &str, consumed: &[ArgumentSpec], bound_fields: &[&str]) -> bool {
    consumed.iter().any(|field| field.name == name)
        || (!bound_fields.contains(&name) && !is_control(name))
}

pub fn is_control(name: &str) -> bool {
    crate::params::is_control_param(name) || HOST_CONTROLS.contains(&name)
}

const HOST_CONTROLS: &[&str] = &[
    "_agentic_loop_api_surface",
    "_agentic_loop_depth",
    "_agentic_loop_fingerprints",
    "_code_interpreter_interception_active",
    "_code_interpreter_interception_converted_stream",
    "_code_interpreter_interception_sandbox_key",
    "_code_interpreter_interception_session_scoped",
    "_headroom_interception_converted_stream",
    "_litellm_strip_stream_usage",
    "_router_weights",
    "_websearch_interception_converted_stream",
    "_websearch_interception_emit_native_blocks",
    "acompletion",
    "adaptive_router_config",
    "adaptive_router_default_model",
    "aembedding",
    "aimg_generation",
    "allm_passthrough_route",
    "allow_client_keepalive_override",
    "allowed_model_region",
    "allowed_openai_params",
    "annotation_cost_per_page",
    "api_version",
    "arize_api_key",
    "arize_space_id",
    "arize_space_key",
    "assistant_continue_message",
    "async_call",
    "atext_completion",
    "attempted_targets",
    "auto_router_config",
    "auto_router_config_path",
    "auto_router_default_model",
    "auto_router_embedding_model",
    "auto_router_max_input_chars",
    "auto_router_model_compression",
    "auto_router_routing_compression",
    "aws_batch_role_arn",
    "azure",
    "azure_password",
    "azure_username",
    "base_model",
    "bedrock_tags",
    "bos_token",
    "budget_duration",
    "cache",
    "cache_creation_input_audio_token_cost",
    "cache_creation_input_token_cost",
    "cache_creation_input_token_cost_above_1hr",
    "cache_creation_input_token_cost_above_200k_tokens",
    "cache_creation_input_token_cost_above_272k_tokens",
    "cache_creation_input_token_cost_above_272k_tokens_flex",
    "cache_creation_input_token_cost_above_272k_tokens_priority",
    "cache_creation_input_token_cost_flex",
    "cache_creation_input_token_cost_priority",
    "cache_creation_input_token_cost_ultrafast",
    "cache_key",
    "cache_read_input_audio_token_cost",
    "cache_read_input_token_cost",
    "cache_read_input_token_cost_above_200k_tokens",
    "cache_read_input_token_cost_above_200k_tokens_priority",
    "cache_read_input_token_cost_above_272k_tokens",
    "cache_read_input_token_cost_above_272k_tokens_flex",
    "cache_read_input_token_cost_above_272k_tokens_priority",
    "cache_read_input_token_cost_above_512k_tokens",
    "cache_read_input_token_cost_flex",
    "cache_read_input_token_cost_priority",
    "cache_read_input_token_cost_ultrafast",
    "caching",
    "caching_groups",
    "citation_cost_per_token",
    "client",
    "client_side_timeout",
    "complete_response",
    "completion_call_id",
    "complexity_router_config",
    "complexity_router_default_model",
    "configurable_clientside_auth_params",
    "context_window_fallback_dict",
    "cooldown_time",
    "cost_per_query",
    "custom_prompt_dict",
    "data_residency",
    "dd_agent_host",
    "dd_agent_port",
    "dd_api_key",
    "dd_site",
    "default_api_key_rpm_limit",
    "default_api_key_tpm_limit",
    "disable_add_transform_inline_image_block",
    "enable_json_schema_validation",
    "enable_prompt_caching",
    "enable_tag_filtering",
    "ensure_alternating_roles",
    "eos_token",
    "fallback_depth",
    "fallbacks",
    "fastest_response",
    "final_prompt_value",
    "force_timeout",
    "gcs_bucket_name",
    "gcs_path_service_account",
    "google_maps_grounding_cost_per_query",
    "headers",
    "hf_model_name",
    "humanloop_api_key",
    "id",
    "input_cost_per_audio_per_second",
    "input_cost_per_audio_per_second_above_128k_tokens",
    "input_cost_per_audio_token",
    "input_cost_per_audio_token_batches",
    "input_cost_per_character",
    "input_cost_per_character_above_128k_tokens",
    "input_cost_per_image",
    "input_cost_per_image_above_128k_tokens",
    "input_cost_per_image_token",
    "input_cost_per_image_token_batches",
    "input_cost_per_pixel",
    "input_cost_per_query",
    "input_cost_per_second",
    "input_cost_per_token",
    "input_cost_per_token_above_128k_tokens",
    "input_cost_per_token_above_200k_tokens",
    "input_cost_per_token_above_200k_tokens_priority",
    "input_cost_per_token_above_272k_tokens",
    "input_cost_per_token_above_272k_tokens_flex",
    "input_cost_per_token_above_272k_tokens_priority",
    "input_cost_per_token_above_512k_tokens",
    "input_cost_per_token_batches",
    "input_cost_per_token_cache_hit",
    "input_cost_per_token_flex",
    "input_cost_per_token_priority",
    "input_cost_per_token_ultrafast",
    "input_cost_per_video_per_second",
    "input_cost_per_video_per_second_above_128k_tokens",
    "input_cost_per_video_per_second_above_15s_interval",
    "input_cost_per_video_per_second_above_8s_interval",
    "input_cost_per_video_token",
    "input_cost_per_video_token_batches",
    "itpm",
    "keepalive_seconds",
    "langfuse_environment",
    "langfuse_host",
    "langfuse_prompt_version",
    "langfuse_public_key",
    "langfuse_secret",
    "langfuse_secret_key",
    "langsmith_api_key",
    "langsmith_base_url",
    "langsmith_project",
    "langsmith_sampling_rate",
    "langsmith_tenant_id",
    "litellm_credential_name",
    "litellm_disabled_callbacks",
    "litellm_request_debug",
    "litellm_session_id",
    "litellm_system_prompt",
    "litellm_trace_id",
    "litellm_trusted_callback_vars",
    "logger_fn",
    "max_agentic_loops",
    "max_budget",
    "max_fallbacks",
    "max_parallel_requests",
    "merge_reasoning_content_in_choices",
    "metadata",
    "mock_response",
    "mock_timeout",
    "model_alias_map",
    "model_config",
    "model_file_id_mapping",
    "model_info",
    "model_list",
    "newrelic_api_key",
    "newrelic_region",
    "no-log",
    "num_retries",
    "ocr_cost_per_credit",
    "ocr_cost_per_page",
    "order",
    "otpm",
    "output_cost_per_audio_per_second",
    "output_cost_per_audio_token",
    "output_cost_per_character",
    "output_cost_per_character_above_128k_tokens",
    "output_cost_per_image",
    "output_cost_per_image_token",
    "output_cost_per_pixel",
    "output_cost_per_reasoning_token",
    "output_cost_per_reasoning_token_flex",
    "output_cost_per_reasoning_token_priority",
    "output_cost_per_second",
    "output_cost_per_second_1080p",
    "output_cost_per_second_480p",
    "output_cost_per_second_4k",
    "output_cost_per_second_720p",
    "output_cost_per_token",
    "output_cost_per_token_above_128k_tokens",
    "output_cost_per_token_above_200k_tokens",
    "output_cost_per_token_above_200k_tokens_priority",
    "output_cost_per_token_above_272k_tokens",
    "output_cost_per_token_above_272k_tokens_flex",
    "output_cost_per_token_above_272k_tokens_priority",
    "output_cost_per_token_above_512k_tokens",
    "output_cost_per_token_batches",
    "output_cost_per_token_flex",
    "output_cost_per_token_priority",
    "output_cost_per_token_ultrafast",
    "output_cost_per_video_per_second",
    "output_cost_per_video_token",
    "output_vector_size",
    "posthog_api_key",
    "posthog_api_url",
    "preset_cache_key",
    "prompt_environment",
    "prompt_id",
    "prompt_label",
    "prompt_variables",
    "prompt_version",
    "provider_specific_header",
    "quality_router_config",
    "quality_router_default_model",
    "region_name",
    "regional_endpoint_uplift_multiplier",
    "regional_processing_uplift_multiplier_eu",
    "regional_processing_uplift_multiplier_us",
    "retry_policy",
    "retry_strategy",
    "roles",
    "routing_strategy",
    "rpm",
    "rust",
    "s3_bucket_name",
    "s3_output_bucket_name",
    "s3_region_name",
    "search_context_cost_per_query",
    "search_tool_name",
    "secret_fields",
    "self",
    "shared_session",
    "ssl_verify",
    "stream_response",
    "stream_timeout",
    "supports_system_message",
    "tags",
    "text_completion",
    "tiered_pricing",
    "tpm",
    "ttl",
    "turn_off_message_logging",
    "use_chat_completions_api",
    "use_client",
    "use_in_pass_through",
    "use_litellm_proxy",
    "use_xai_oauth",
    "user_continue_message",
    "verbose",
    "wandb_api_key",
    "weave_project_id",
    "weight",
];

pub fn compose_body<B: Serialize>(
    arguments: &CallArguments,
    body: &B,
    consumed: &[&str],
) -> Result<Value, crate::params::Error> {
    let Value::Object(fields) =
        serde_json::to_value(body).map_err(|_| crate::params::Error::Body)?
    else {
        return Err(crate::params::Error::Body);
    };
    let overrides = match arguments.get("extra_body") {
        None | Some(Value::Null) => None,
        Some(Value::Object(fields)) => Some(fields),
        Some(_) => return Err(crate::params::Error::ExtraBody),
    };
    let extensions = arguments.iter().filter(|(name, _)| {
        !consumed.contains(&name.as_str()) && name.as_str() != "extra_body" && !is_control(name)
    });
    Ok(Value::Object(
        fields
            .into_iter()
            .chain(
                extensions
                    .chain(overrides.into_iter().flatten())
                    .filter(|(name, _)| {
                        name.as_str() != "model"
                            && name.as_str() != "extra_body"
                            && !crate::params::is_control_param(name)
                    })
                    .map(|(name, value)| (name.clone(), value.clone())),
            )
            .collect(),
    ))
}

impl Deref for CallArguments {
    type Target = Map<String, Value>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Map<String, Value>> for CallArguments {
    fn from(values: Map<String, Value>) -> Self {
        Self(values)
    }
}

impl From<CallArguments> for Map<String, Value> {
    fn from(arguments: CallArguments) -> Self {
        arguments.0
    }
}

impl FromIterator<(String, Value)> for CallArguments {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl IntoIterator for CallArguments {
    type Item = (String, Value);
    type IntoIter = serde_json::map::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn composition_preserves_extensions_and_applies_shallow_explicit_overrides() {
        let original = json!({
            "known": false, "future": {"old": 1}, "null": null, "zero": 0,
            "metadata": {"host": true}, "shared_session": "host", "api_key": "secret",
            "extra_body": {
                "known": null, "future": {"new": [false, 0, null]},
                "metadata": {"provider": true}, "model": "ignored", "api_key": "ignored"
            }
        });
        let arguments = serde_json::from_value(original.clone()).unwrap();
        let body = compose_body(
            &arguments,
            &json!({"model":"resolved", "known":false}),
            &["known"],
        )
        .unwrap();
        assert_eq!(
            body,
            json!({
                "model":"resolved", "known":null, "future":{"new":[false,0,null]},
                "null":null, "zero":0, "metadata":{"provider":true}
            })
        );
        assert_eq!(serde_json::to_value(arguments).unwrap(), original);
    }

    #[test]
    fn projection_prioritizes_consumed_fields_and_keeps_unknown_names() {
        let fields = [ArgumentSpec {
            name: "id",
            secret: false,
        }];
        assert!(should_project("id", &fields, &[]));
        assert!(!should_project("id", &[], &[]));
        assert!(should_project("future_option", &[], &[]));
        assert!(!should_project("document", &fields, &["document"]));
        assert!(!should_project("metadata", &fields, &[]));
        assert!(!should_project("callbacks", &fields, &[]));
        assert!(!should_project("ocr_cost_per_page", &fields, &[]));
    }

    #[test]
    fn invalid_extra_body_is_rejected_without_coercing_it_to_empty() {
        for value in [json!(false), json!(0), json!([]), json!("")] {
            let arguments = serde_json::from_value(json!({"extra_body":value})).unwrap();
            assert_eq!(
                compose_body(&arguments, &json!({}), &[]),
                Err(crate::params::Error::ExtraBody)
            );
        }
        let arguments = serde_json::from_value(json!({"extra_body":null})).unwrap();
        assert_eq!(
            compose_body(&arguments, &json!({}), &[]).unwrap(),
            json!({})
        );
    }

    #[test]
    fn typed_views_preserve_missing_and_explicit_null_in_the_source() {
        #[derive(Deserialize)]
        struct Options {
            enabled: Option<bool>,
        }
        let arguments: CallArguments =
            serde_json::from_value(json!({"enabled":null,"future":0})).unwrap();
        assert!(
            parse_options::<Options>(&arguments)
                .unwrap()
                .enabled
                .is_none()
        );
        assert_eq!(arguments.get("enabled"), Some(&Value::Null));
        assert_eq!(arguments.get("missing"), None);
        let invalid = serde_json::from_value(json!({"enabled":0})).unwrap();
        assert_eq!(
            parse_options::<Options>(&invalid).err().unwrap().path,
            "enabled"
        );
    }
}
