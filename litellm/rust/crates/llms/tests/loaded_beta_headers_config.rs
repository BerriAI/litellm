use litellm_llms::anthropic::beta_headers::{
    BetaPolicy, accepted_or_unlisted, install_beta_headers_config,
};
use litellm_llms_types::providers::anthropic::{BetaProvider, BetaSet};
use rstest::{fixture, rstest};
use serde_json::json;

#[fixture]
fn remote_config() {
    install_beta_headers_config(
        serde_json::from_value(json!({
            "azure_ai": {
                "web-search-2025-03-05": null,
                "example-beta-2099-01-01": "example-beta-2099-01-01",
            },
            "bedrock": {
                "skills-2025-10-02": "skills-2025-10-02",
                "effort-2025-11-24": null,
            },
        }))
        .unwrap(),
    );
}

fn beta_header(value: &str) -> Vec<(String, String)> {
    vec![("anthropic-beta".to_string(), value.to_string())]
}

#[rstest]
#[case::azure_ai(
    BetaProvider::AzureAi,
    "web-search-2025-03-05,example-beta-2099-01-01,oauth-2025-04-20",
    Some("example-beta-2099-01-01")
)]
#[case::bedrock(
    BetaProvider::Bedrock,
    "skills-2025-10-02,effort-2025-11-24",
    Some("skills-2025-10-02")
)]
#[case::nothing_survives(BetaProvider::AzureAi, "web-search-2025-03-05", None)]
fn filter_reads_the_loaded_config_over_the_compiled_columns(
    _remote_config: (),
    #[case] provider: BetaProvider,
    #[case] sent: &str,
    #[case] expected: Option<&str>,
) {
    assert_eq!(
        BetaPolicy::Filter(provider).apply(beta_header(sent)),
        expected.map(beta_header).unwrap_or_default()
    );
}

#[rstest]
fn foundry_forwards_only_what_the_loaded_row_does_not_list(_remote_config: ()) {
    let sent: BetaSet = "web-search-2025-03-05,example-beta-2099-01-01,oauth-2025-04-20"
        .parse()
        .unwrap_or_else(|never| match never {});
    assert_eq!(
        accepted_or_unlisted(sent, BetaProvider::AzureAi).to_string(),
        "example-beta-2099-01-01,oauth-2025-04-20"
    );
}
