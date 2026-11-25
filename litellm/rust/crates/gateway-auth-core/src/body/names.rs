//! The names the body check reads a request body by: the deployment-level parameters a caller may
//! only override with an admin's grant, the positions a body nests them at, and the keys that
//! carry content instead.

use std::collections::BTreeSet;

use super::surface::Surface;

/// The Microsoft Entra credential fields the Azure provider resolves through `get_secret`. They
/// stay usable as plain metadata tags, where no provider reads them.
pub(crate) const ENTRA_CREDENTIAL_PARAMS: [&str; 6] = [
    "tenant_id",
    "client_id",
    "client_secret",
    "azure_username",
    "azure_password",
    "azure_scope",
];

pub const VERTEX_TARGET_PARAMS: [&str; 4] = [
    "vertex_project",
    "vertex_location",
    "vertex_ai_project",
    "vertex_ai_location",
];

/// Mappings whose entries are spread into the outbound call as keyword arguments.
pub(crate) const NESTED_CONFIG_KEYS: [&str; 2] = ["litellm_embedding_config", "extra_body"];

/// Mappings the observability callbacks read per-request configuration from.
pub(crate) const METADATA_KEYS: [&str; 2] = ["metadata", "litellm_metadata"];

pub(crate) const BUCKET_PARAMS: [&str; 2] = ["s3_bucket_name", "s3_output_bucket_name"];
pub(crate) const ENDPOINT_URL_PARAM: &str = "s3_endpoint_url";
pub(crate) const CREDENTIAL_NAME: &str = "litellm_credential_name";
pub(crate) const API_KEY: &str = "api_key";

/// The lists a body names its fallbacks in.
pub(crate) const FALLBACK_FIELDS: [&str; 3] = [
    "fallbacks",
    "context_window_fallbacks",
    "content_policy_fallbacks",
];
/// Router settings a body overrides for its own call, fallbacks among them.
pub(crate) const ROUTER_SETTINGS_OVERRIDE: &str = "router_settings_override";

/// Callback params that describe the request being logged, without choosing where it is logged or
/// with which credentials.
const SAFE_CALLBACK_PARAMS: [&str; 2] = ["langfuse_prompt_version", "langsmith_sampling_rate"];

/// Observability fields integrations read that the host's callback tables do not list.
/// `user_api_key_auth_metadata` is written only from the authenticated key's record, so a
/// caller's value would redirect that key's telemetry.
const EXTRA_OBSERVABILITY_PARAMS: [&str; 4] = [
    "posthog_api_url",
    "user_api_key_auth_metadata",
    "wandb_api_key",
    "weave_project_id",
];

/// What the body check itself bans, before the host's tables are added. Each either retargets the
/// outbound call or a storage write, selects an identity the proxy holds, or switches a decision
/// only a deployment makes.
const OVERRIDE_PARAMS: &[&str] = &[
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
    "vertex_location",
    "vertex_project",
    "vertex_ai_location",
    "vertex_ai_project",
    "azure_ad_token",
    "tenant_id",
    "client_id",
    "client_secret",
    "azure_username",
    "azure_password",
    "azure_scope",
    "aws_bedrock_runtime_endpoint",
    "aws_bedrock_control_plane_endpoint",
    "aws_bedrock_project_id",
    "workspace_id",
    "aws_workspace_id",
    "anthropic_workspace_id",
    "anthropic-workspace-id",
    "bedrock_tags",
    "agentcore_runtime_session_id",
    "s3_endpoint_url",
    "s3_bucket_name",
    "s3_output_bucket_name",
    "stored_s3_bucket_name",
    "allow_legacy_cloud_file_ids",
    "sagemaker_base_url",
    "deployment_url",
    "nvcf_function_id",
    "use_ssl",
    "riva_offline",
    "rust",
    "fireworks_forward_user_id",
    "model_list",
    "vertex_ai_credentials",
    "github_copilot_auth_type",
    "github_copilot_host",
    "user_provider_credentials",
    "github_copilot_user_session",
];

/// Root keys of a model call that carry what the caller says to the model. Their text reaches the
/// provider's payload and is never read as a parameter, so a string there that looks like a secret
/// reference is only text.
const CONTENT_KEYS: &[&str] = &[
    "messages",
    "input",
    "prompt",
    "instructions",
    "system",
    "tools",
    "functions",
    "tool_choice",
    "function_call",
    "response_format",
    "stop",
    "stop_sequences",
    "suffix",
    "prediction",
    "text",
    "query",
    "documents",
    "document",
    "contents",
    "systemInstruction",
    "system_instruction",
    "toolConfig",
    "tool_config",
    "generationConfig",
    "generation_config",
    "safetySettings",
    "safety_settings",
    "image",
    "images",
    "mask",
    "file",
    "voice",
    "user",
    "name",
    "description",
    "conversation",
    "output_config",
    "output_format",
    "thinking",
    "reasoning",
    "tags",
];

pub(crate) fn is_content_key(key: &str) -> bool {
    CONTENT_KEYS.contains(&key)
}

/// `metadata`, `litellm_metadata`, and the `metadata[tags]` spelling a form gives their entries.
pub(crate) fn is_metadata_key(key: &str) -> bool {
    METADATA_KEYS.iter().any(|metadata| {
        key.strip_prefix(metadata)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('['))
    })
}

/// Root keys whose value is a mapping the check walks.
fn is_container_key(key: &str) -> bool {
    NESTED_CONFIG_KEYS.contains(&key)
        || METADATA_KEYS.contains(&key)
        || FALLBACK_FIELDS.contains(&key)
        || key == ROUTER_SETTINGS_OVERRIDE
        || key == "user_config"
        || key == "litellm_params"
}

/// How much of the value under a root key a host reads into [`super::Params`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reading {
    /// No rule looks at it.
    Unread,
    /// A rule reads it only when it is a string.
    Shallow,
    /// A rule walks what is below it.
    Deep,
}

/// The name tables other Python modules own, handed over as they are. The gateway has no Python
/// to ask, so these move into Rust with the modules that own them.
#[derive(Clone, Debug, Default)]
pub struct HostTables {
    /// `_supported_callback_params`.
    pub callback_params: Vec<String>,
    /// `_request_blocked_callback_params`.
    pub blocked_callback_params: Vec<String>,
    /// The custom pricing fields and the pricing currency field, in the order a rejection checks
    /// them.
    pub pricing_params: Vec<String>,
    /// `server_owned_wif_litellm_params`.
    pub federation_params: Vec<String>,
    /// `oauth_token_exchange_litellm_params`.
    pub token_exchange_params: Vec<String>,
    /// `github_copilot_oauth_litellm_params`.
    pub github_copilot_params: Vec<String>,
    /// `UPSTREAM_BOUND_PARAMS`: the credential and endpoint fields a provider call reads.
    pub upstream_bound_params: Vec<String>,
}

/// Every name table the check reads, built once.
#[derive(Clone, Debug)]
pub struct Names {
    overrides: Vec<String>,
    extra_body_overrides: Vec<String>,
    fallbacks_only: Vec<String>,
    metadata_overrides: Vec<String>,
    secret_resolving: BTreeSet<String>,
    federation: Vec<String>,
    token_exchange: Vec<String>,
    github_copilot: Vec<String>,
}

impl Names {
    pub fn new(tables: HostTables) -> Self {
        let observability: BTreeSet<&str> = tables
            .callback_params
            .iter()
            .map(String::as_str)
            .filter(|param| !SAFE_CALLBACK_PARAMS.contains(param))
            .chain(tables.blocked_callback_params.iter().map(String::as_str))
            .chain(EXTRA_OBSERVABILITY_PARAMS)
            .collect();
        let overrides: Vec<String> = OVERRIDE_PARAMS
            .iter()
            .copied()
            .chain(observability)
            .chain(tables.pricing_params.iter().map(String::as_str))
            .map(str::to_owned)
            .collect();
        let metadata_overrides = overrides
            .iter()
            .filter(|param| {
                !VERTEX_TARGET_PARAMS.contains(&param.as_str())
                    && !ENTRA_CREDENTIAL_PARAMS.contains(&param.as_str())
            })
            .cloned()
            .collect();
        let secret_resolving = overrides
            .iter()
            .chain(&tables.upstream_bound_params)
            .cloned()
            .collect();
        let extra_body_overrides = overrides
            .iter()
            .cloned()
            .chain([FALLBACK_FIELDS[0].to_owned()])
            .collect();
        Self {
            overrides,
            extra_body_overrides,
            fallbacks_only: vec![FALLBACK_FIELDS[0].to_owned()],
            metadata_overrides,
            secret_resolving,
            federation: tables.federation_params,
            token_exchange: tables.token_exchange_params,
            github_copilot: tables.github_copilot_params,
        }
    }

    /// The parameters a caller may set only with an admin's grant, in the order a rejection
    /// checks them.
    pub fn overrides(&self) -> &[String] {
        &self.overrides
    }

    /// [`Self::overrides`] and `fallbacks`, which `extra_body` may not set either.
    pub(crate) fn extra_body_overrides(&self) -> &[String] {
        &self.extra_body_overrides
    }

    /// `fallbacks` alone, what a fallback target's `extra_body` is judged for.
    pub(crate) fn fallbacks_only(&self) -> &[String] {
        &self.fallbacks_only
    }

    /// [`Self::overrides`] without the names that are only tags inside metadata.
    pub(crate) fn metadata_overrides(&self) -> &[String] {
        &self.metadata_overrides
    }

    /// How much of the value under a root key a host reads. A model call is read everywhere
    /// outside its content, which can be megabytes of text. Any other body is read only under
    /// the names a rule knows.
    pub fn reading(&self, surface: &Surface, key: &str) -> Reading {
        if surface.model_call {
            return if is_content_key(key) {
                Reading::Unread
            } else {
                Reading::Deep
            };
        }
        if is_container_key(key) {
            Reading::Deep
        } else if self.resolves_secrets(key) || key == CREDENTIAL_NAME || key == API_KEY {
            Reading::Shallow
        } else {
            Reading::Unread
        }
    }

    /// Whether a provider call resolves this parameter's value through `get_secret`.
    pub(crate) fn resolves_secrets(&self, param: &str) -> bool {
        self.secret_resolving.contains(param)
    }

    pub(crate) fn federation(&self) -> &[String] {
        &self.federation
    }

    pub(crate) fn is_token_exchange(&self, param: &str) -> bool {
        self.token_exchange.iter().any(|name| name == param)
    }

    pub(crate) fn is_github_copilot(&self, param: &str) -> bool {
        self.github_copilot.iter().any(|name| name == param)
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    fn names() -> Names {
        Names::new(HostTables {
            callback_params: vec![
                "langfuse_host".to_owned(),
                "langfuse_prompt_version".to_owned(),
            ],
            blocked_callback_params: vec!["gcs_bucket_name".to_owned()],
            pricing_params: vec![
                "input_cost_per_token".to_owned(),
                "pricing_currency".to_owned(),
            ],
            upstream_bound_params: vec!["api_key".to_owned()],
            ..HostTables::default()
        })
    }

    #[rstest]
    #[case::own("api_base", true)]
    #[case::callback("langfuse_host", true)]
    #[case::blocked_callback("gcs_bucket_name", true)]
    #[case::extra_observability("wandb_api_key", true)]
    #[case::pricing("input_cost_per_token", true)]
    #[case::safe_callback("langfuse_prompt_version", false)]
    #[case::upstream_bound_only("api_key", false)]
    fn an_override_is_an_own_name_or_one_a_host_table_lists(
        names: Names,
        #[case] param: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(names.overrides().iter().any(|name| name == param), expected);
    }

    #[rstest]
    fn host_tables_follow_the_own_names_in_a_stable_order(names: Names) {
        assert_eq!(
            names.overrides()[OVERRIDE_PARAMS.len()..],
            [
                "gcs_bucket_name",
                "langfuse_host",
                "posthog_api_url",
                "user_api_key_auth_metadata",
                "wandb_api_key",
                "weave_project_id",
                "input_cost_per_token",
                "pricing_currency",
            ]
        );
    }

    #[rstest]
    #[case::entra("client_id", false)]
    #[case::vertex_target("vertex_project", false)]
    #[case::endpoint("api_base", true)]
    #[case::callback("langfuse_host", true)]
    fn metadata_keeps_every_override_but_the_names_that_are_only_tags_there(
        names: Names,
        #[case] param: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(
            names.metadata_overrides().iter().any(|name| name == param),
            expected
        );
    }

    #[rstest]
    #[case::override_param("client_id", true)]
    #[case::upstream_bound("api_key", true)]
    #[case::content("prompt", false)]
    fn a_param_resolves_secrets_when_a_provider_call_reads_it(
        names: Names,
        #[case] param: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(names.resolves_secrets(param), expected);
    }

    #[rstest]
    #[case::override_param("api_base", Reading::Shallow)]
    #[case::upstream_bound("api_key", Reading::Shallow)]
    #[case::credential_name("litellm_credential_name", Reading::Shallow)]
    #[case::nested_config("extra_body", Reading::Deep)]
    #[case::metadata("litellm_metadata", Reading::Deep)]
    #[case::caller_supplied_router("user_config", Reading::Deep)]
    #[case::fallbacks("context_window_fallbacks", Reading::Deep)]
    #[case::router_settings_override("router_settings_override", Reading::Deep)]
    #[case::content("messages", Reading::Unread)]
    #[case::unknown_param("some_provider_param", Reading::Unread)]
    fn a_host_reads_only_the_values_a_rule_looks_at_outside_a_model_call(
        names: Names,
        #[case] key: &str,
        #[case] expected: Reading,
    ) {
        assert_eq!(
            names.reading(&Surface::of(Some("/key/generate")), key),
            expected
        );
    }

    #[rstest]
    #[case::content("messages", Reading::Unread)]
    #[case::tool_schemas("tools", Reading::Unread)]
    #[case::override_param("api_base", Reading::Deep)]
    #[case::unknown_param("some_provider_param", Reading::Deep)]
    #[case::metadata("metadata", Reading::Deep)]
    fn a_host_reads_a_model_call_everywhere_outside_its_content(
        names: Names,
        #[case] key: &str,
        #[case] expected: Reading,
    ) {
        assert_eq!(names.reading(&Surface::of(None), key), expected);
    }

    #[rstest]
    #[case::bare("metadata", true)]
    #[case::litellm("litellm_metadata", true)]
    #[case::form_entry("metadata[tags]", true)]
    #[case::litellm_form_entry("litellm_metadata[spend_logs_metadata][owner]", true)]
    #[case::longer_name("metadata_extra", false)]
    #[case::unrelated("model", false)]
    fn a_metadata_key_is_a_container_or_one_of_its_form_entries(
        #[case] key: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(is_metadata_key(key), expected);
    }
}
