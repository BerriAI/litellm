#![forbid(unsafe_code)]

mod compose;
pub mod connection;
mod error;
mod owned;
pub mod route;

use litellm_auth_types::Setting;

pub use compose::compose_body;
pub use error::Error;

pub fn settings() -> impl Iterator<Item = &'static Setting> {
    route::SETTINGS
        .iter()
        .chain(connection::SETTINGS)
        .chain(litellm_auth_aws::settings::SETTINGS)
        .chain(litellm_auth_azure::settings::SETTINGS)
        .chain(litellm_auth_gcp::settings::SETTINGS)
}

pub fn python_owned(name: &str) -> bool {
    name.starts_with(owned::INTERNAL_PREFIX) || owned::PYTHON_OWNED.binary_search(&name).is_ok()
}

pub fn is_owned(name: &str) -> bool {
    python_owned(name) || settings().any(|setting| setting.claims(name))
}

pub fn is_secret(name: &str) -> bool {
    settings().any(|setting| setting.secret && setting.claims(name))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::connection("api_key", true)]
    #[case::connection_alias("base_url", true)]
    #[case::route("extra_body", true)]
    #[case::aws("aws_profile_name", true)]
    #[case::azure("azure_federated_token_file", true)]
    #[case::vertex_alias("vertex_ai_location", true)]
    #[case::python_proxy_metadata("metadata", true)]
    #[case::python_internal_state("proxy_server_request", true)]
    #[case::python_callback_credential("arize_api_key", true)]
    #[case::python_pricing_key("input_cost_per_token", true)]
    #[case::python_prefix("_litellm_anything", true)]
    #[case::provider_field("temperature", false)]
    #[case::provider_nested_name("metadata_extra", false)]
    #[case::env_name_is_not_a_kwarg("AWS_REGION_NAME", false)]
    fn is_owned_covers_every_declaring_crate_and_the_python_set(
        #[case] name: &str,
        #[case] owned: bool,
    ) {
        assert_eq!(is_owned(name), owned);
    }

    #[rstest]
    #[case::api_key("api_key", true)]
    #[case::aws_secret("aws_secret_access_key", true)]
    #[case::azure_secret("client_secret", true)]
    #[case::vertex_credentials("vertex_ai_credentials", true)]
    #[case::aws_identity("aws_region_name", false)]
    #[case::azure_identity("tenant_id", false)]
    #[case::destination("api_base", false)]
    #[case::python_only("metadata", false)]
    #[case::provider_field("temperature", false)]
    fn is_secret_reads_the_row_flag(#[case] name: &str, #[case] secret: bool) {
        assert_eq!(is_secret(name), secret);
    }

    #[test]
    fn no_kwarg_is_claimed_by_two_rows() {
        let claimed: Vec<&str> = settings()
            .flat_map(|setting| setting.kwargs.iter().copied())
            .collect();
        let unique: BTreeSet<&str> = claimed.iter().copied().collect();
        assert_eq!(claimed.len(), unique.len(), "{claimed:?}");
    }

    #[test]
    fn no_env_name_backs_two_rows() {
        let backing: Vec<&str> = settings()
            .flat_map(|setting| setting.env.iter().copied())
            .collect();
        let unique: BTreeSet<&str> = backing.iter().copied().collect();
        assert_eq!(backing.len(), unique.len(), "{backing:?}");
    }

    #[test]
    fn the_generated_python_set_is_sorted_and_unique_so_binary_search_is_valid() {
        let sorted: BTreeSet<&str> = owned::PYTHON_OWNED.iter().copied().collect();
        assert_eq!(
            owned::PYTHON_OWNED,
            sorted.into_iter().collect::<Vec<_>>().as_slice()
        );
    }

    const NOT_OWNED_BY_PYTHON_YET: &[&str] = &[
        "additional_drop_params",
        "aws_access_key_id",
        "aws_bedrock_runtime_endpoint",
        "aws_external_id",
        "aws_profile_name",
        "aws_region_name",
        "aws_role_name",
        "aws_secret_access_key",
        "aws_session_name",
        "aws_session_token",
        "aws_sts_endpoint",
        "aws_web_identity_token",
        "azure_ad_token",
        "azure_authority_host",
        "azure_credential",
        "azure_federated_token_file",
        "base_url",
        "custom_endpoint",
        "drop_params",
        "enable_azure_ad_token_refresh",
        "extra_body",
        "extra_headers",
        "max_response_bytes",
        "model",
        "req_format",
        "timeout",
        "timeout_seconds",
        "vertex_ai_credentials",
        "vertex_ai_location",
        "vertex_ai_project",
        "vertex_credentials",
        "vertex_location",
        "vertex_project",
    ];

    #[test]
    fn every_row_kwarg_python_does_not_own_is_a_known_gap_that_only_shrinks() {
        let gap: BTreeSet<&str> = settings()
            .flat_map(|setting| setting.kwargs.iter().copied())
            .filter(|name| !python_owned(name))
            .collect();
        let known: BTreeSet<&str> = NOT_OWNED_BY_PYTHON_YET.iter().copied().collect();
        assert_eq!(gap, known);
    }
}
