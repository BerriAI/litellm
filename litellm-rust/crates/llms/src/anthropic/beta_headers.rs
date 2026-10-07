use litellm_http::request::{with_header, without_headers};
use litellm_llms_types::providers::anthropic::{BetaProvider, BetaSet};

use crate::{anthropic::common_utils::existing_betas, base_llm::auth::Headers};

const BETA_HEADER: &str = "anthropic-beta";

/// What happens to the caller's and LiteLLM's `anthropic-beta` values before they leave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BetaPolicy {
    /// The host decides for itself which betas it accepts.
    Forward,
    /// Python's `update_headers_with_filtered_beta`: only the betas the provider accepts are
    /// sent, under the names it expects.
    Filter(BetaProvider),
    /// The host accepts no beta, as Python's filter treats a provider with no column.
    Drop,
}

impl BetaPolicy {
    /// Every casing of `anthropic-beta` is replaced by one sorted header, or removed when no
    /// beta survives.
    pub fn apply(self, headers: Headers) -> Headers {
        let provider = match self {
            Self::Forward => return headers,
            Self::Drop => return without_headers(headers, &[BETA_HEADER]),
            Self::Filter(provider) => provider,
        };
        let accepted: BetaSet = existing_betas(&headers)
            .iter()
            .filter_map(|beta| beta.on(provider))
            .collect();
        let headers = without_headers(headers, &[BETA_HEADER]);
        if accepted.is_empty() {
            return headers;
        }
        with_header(headers, BETA_HEADER, accepted.to_string())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use litellm_llms_types::providers::anthropic::AnthropicBeta;
    use rstest::{fixture, rstest};
    use serde::Deserialize;

    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn beta_header(beta: &AnthropicBeta) -> Headers {
        headers(&[("x-api-key", "k"), ("anthropic-beta", beta.as_str())])
    }

    fn beta_headers_after(policy: BetaPolicy, input: &[(&str, &str)]) -> Headers {
        policy.apply(headers(input))
    }

    #[derive(Deserialize)]
    struct BetaConfig {
        anthropic: BTreeMap<String, Option<String>>,
    }

    #[fixture]
    fn configured_anthropic_betas() -> BetaSet {
        let config: BetaConfig = serde_json::from_str(include_str!(
            "../../../../../litellm/anthropic_beta_headers_config.json"
        ))
        .unwrap();
        config
            .anthropic
            .into_keys()
            .map(|beta| beta.parse().unwrap())
            .collect()
    }

    #[rstest]
    #[case::anthropic(BetaProvider::Anthropic)]
    #[case::azure_ai(BetaProvider::AzureAi)]
    #[case::bedrock_converse(BetaProvider::BedrockConverse)]
    #[case::bedrock(BetaProvider::Bedrock)]
    #[case::vertex_ai(BetaProvider::VertexAi)]
    fn filter_configured_values_and_drop_unknowns(
        configured_anthropic_betas: BetaSet,
        #[case] provider: BetaProvider,
    ) {
        let input: BetaSet = configured_anthropic_betas
            .clone()
            .into_iter()
            .chain([
                AnthropicBeta::Other("unknown-header-1".into()),
                AnthropicBeta::Other("unknown-header-2".into()),
                AnthropicBeta::Other("fake-beta-2025-01-01".into()),
            ])
            .collect();
        let expected: BetaSet = configured_anthropic_betas
            .iter()
            .filter_map(|beta| beta.on(provider))
            .collect();
        let filtered = beta_headers_after(
            BetaPolicy::Filter(provider),
            &[("anthropic-beta", &input.to_string()), ("x-api-key", "k")],
        );
        assert_eq!(
            filtered,
            headers(&[
                ("x-api-key", "k"),
                ("anthropic-beta", &expected.to_string())
            ])
        );
        assert_eq!(
            existing_betas(&filtered).contains(&AnthropicBeta::Compact20260904),
            provider == BetaProvider::Anthropic
        );
    }

    #[rstest]
    #[case::anthropic(
        BetaProvider::Anthropic,
        AnthropicBeta::AdvisorTool20260301,
        AnthropicBeta::Bash20241022
    )]
    #[case::azure_ai(
        BetaProvider::AzureAi,
        AnthropicBeta::AdvancedToolUse20251120,
        AnthropicBeta::AdvisorTool20260301
    )]
    #[case::bedrock_converse(
        BetaProvider::BedrockConverse,
        AnthropicBeta::ComputerUse20250124,
        AnthropicBeta::AdvisorTool20260301
    )]
    #[case::bedrock(
        BetaProvider::Bedrock,
        AnthropicBeta::AdvancedToolUse20251120,
        AnthropicBeta::AdvisorTool20260301
    )]
    #[case::vertex_ai(
        BetaProvider::VertexAi,
        AnthropicBeta::AdvancedToolUse20251120,
        AnthropicBeta::AdvisorTool20260301
    )]
    fn filter_mixed_supported_rejected_and_unknown_values(
        #[case] provider: BetaProvider,
        #[case] supported: AnthropicBeta,
        #[case] rejected: AnthropicBeta,
    ) {
        let input: BetaSet = [
            supported.clone(),
            rejected,
            AnthropicBeta::Other("unknown-header-123".into()),
        ]
        .into_iter()
        .collect();
        let mapped = supported.on(provider).unwrap();
        assert_eq!(
            beta_headers_after(
                BetaPolicy::Filter(provider),
                &[("anthropic-beta", &input.to_string())],
            ),
            headers(&[("anthropic-beta", mapped.as_str())])
        );
    }

    #[rstest]
    #[case::anthropic_blank(BetaProvider::Anthropic, "")]
    #[case::bedrock_blank(BetaProvider::Bedrock, "")]
    #[case::bedrock_mantle_blank(BetaProvider::BedrockMantle, "")]
    #[case::vertex_ai_blank(BetaProvider::VertexAi, "")]
    #[case::anthropic_whitespace(BetaProvider::Anthropic, " , ")]
    #[case::bedrock_whitespace(BetaProvider::Bedrock, " , ")]
    #[case::bedrock_mantle_whitespace(BetaProvider::BedrockMantle, " , ")]
    #[case::vertex_ai_whitespace(BetaProvider::VertexAi, " , ")]
    fn filter_removes_blank_beta_headers(#[case] provider: BetaProvider, #[case] value: &str) {
        assert_eq!(
            beta_headers_after(
                BetaPolicy::Filter(provider),
                &[
                    ("anthropic-beta", value),
                    ("anthropic-version", "2023-06-01")
                ],
            ),
            headers(&[("anthropic-version", "2023-06-01")])
        );
    }

    #[rstest]
    #[case::anthropic(BetaProvider::Anthropic)]
    #[case::azure_ai(BetaProvider::AzureAi)]
    #[case::bedrock_converse(BetaProvider::BedrockConverse)]
    #[case::bedrock(BetaProvider::Bedrock)]
    #[case::vertex_ai(BetaProvider::VertexAi)]
    fn filter_removes_every_rejected_beta(#[case] provider: BetaProvider) {
        let rejected: BetaSet = AnthropicBeta::KNOWN
            .into_iter()
            .filter(|beta| beta.on(provider).is_none())
            .collect();
        assert!(!rejected.is_empty());
        assert_eq!(
            beta_headers_after(
                BetaPolicy::Filter(provider),
                &[
                    ("anthropic-beta", &rejected.to_string()),
                    ("x-api-key", "k")
                ],
            ),
            headers(&[("x-api-key", "k")])
        );
    }

    #[test]
    fn filter_sends_each_beta_under_the_name_the_provider_expects() {
        for provider in BetaProvider::ALL {
            for beta in AnthropicBeta::KNOWN {
                let expected = match beta.on(*provider) {
                    Some(mapped) => beta_header(&mapped),
                    None => headers(&[("x-api-key", "k")]),
                };
                assert_eq!(
                    BetaPolicy::Filter(*provider).apply(beta_header(&beta)),
                    expected,
                    "{beta} on {provider}"
                );
            }
        }
    }

    #[test]
    fn filter_merges_every_casing_into_one_sorted_header() {
        let kept = AnthropicBeta::KNOWN
            .into_iter()
            .filter(|beta| beta.on(BetaProvider::Anthropic).as_ref() == Some(beta))
            .take(2)
            .collect::<Vec<_>>();
        let [first, second] = kept.as_slice() else {
            panic!("the anthropic column accepts at least two betas unchanged")
        };
        assert_eq!(
            beta_headers_after(
                BetaPolicy::Filter(BetaProvider::Anthropic),
                &[
                    ("ANTHROPIC-BETA", second.as_str()),
                    (
                        "anthropic-beta",
                        &format!("example-beta-2099-01-01,{first}")
                    ),
                ]
            ),
            headers(&[(
                "anthropic-beta",
                &BetaSet::from_iter([first.clone(), second.clone()]).to_string()
            )])
        );
    }

    #[rstest]
    #[case::forward(
        BetaPolicy::Forward,
        &[("Anthropic-Beta", "example-beta-2099-01-01"), ("x-api-key", "k")]
    )]
    #[case::drop(BetaPolicy::Drop, &[("x-api-key", "k")])]
    fn forward_and_drop_ignore_the_config(
        #[case] policy: BetaPolicy,
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(
            beta_headers_after(
                policy,
                &[
                    ("Anthropic-Beta", "example-beta-2099-01-01"),
                    ("x-api-key", "k")
                ]
            ),
            headers(expected)
        );
    }

    #[rstest]
    #[case::anthropic(BetaPolicy::Filter(BetaProvider::Anthropic))]
    #[case::azure_ai(BetaPolicy::Filter(BetaProvider::AzureAi))]
    #[case::bedrock_converse(BetaPolicy::Filter(BetaProvider::BedrockConverse))]
    #[case::bedrock(BetaPolicy::Filter(BetaProvider::Bedrock))]
    #[case::bedrock_mantle(BetaPolicy::Filter(BetaProvider::BedrockMantle))]
    #[case::vertex_ai(BetaPolicy::Filter(BetaProvider::VertexAi))]
    #[case::databricks(BetaPolicy::Filter(BetaProvider::Databricks))]
    #[case::forward(BetaPolicy::Forward)]
    #[case::drop(BetaPolicy::Drop)]
    fn headers_without_a_beta_are_untouched(#[case] policy: BetaPolicy) {
        let input = headers(&[("anthropic-version", "2023-06-01")]);
        assert_eq!(policy.apply(input.clone()), input);
    }
}
