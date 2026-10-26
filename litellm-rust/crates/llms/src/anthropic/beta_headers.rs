use http::HeaderMap;
use litellm_llms_types::providers::anthropic::{BETA_HEADER, BetaProvider, BetaSet};

use crate::{
    Error,
    anthropic::common_utils::{existing_betas, with_betas},
};

/// How the `anthropic-beta` header is treated before a request leaves for a host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BetaPolicy {
    /// Send the header untouched and let the host decide.
    Forward,
    /// Keep only the betas the provider accepts, under the names it expects.
    Filter(BetaProvider),
    /// Remove every `anthropic-beta` header.
    Drop,
}

impl BetaPolicy {
    pub fn apply(self, mut headers: HeaderMap) -> Result<HeaderMap, Error> {
        let provider = match self {
            Self::Forward => return Ok(headers),
            Self::Drop => {
                headers.remove(BETA_HEADER);
                return Ok(headers);
            }
            Self::Filter(provider) => provider,
        };
        let accepted: BetaSet = existing_betas(&headers)
            .iter()
            .filter_map(|beta| beta.on(provider))
            .collect();
        headers.remove(BETA_HEADER);
        if accepted.is_empty() {
            return Ok(headers);
        }
        with_betas(headers, &accepted)
    }
}

#[cfg(test)]
mod tests {
    use crate::anthropic::beta_headers::BetaPolicy;
    use litellm_http::request::header_map as headers;
    use litellm_llms_types::providers::anthropic::BetaProvider;
    use rstest::rstest;

    #[rstest]
    fn forward_keeps_headers_as_is() {
        let input = headers(&[
            (
                "Anthropic-Beta",
                "example-beta-2099-01-01,web-search-2025-03-05",
            ),
            ("x-api-key", "k"),
        ]);
        assert_eq!(BetaPolicy::Forward.apply(input.clone()), Ok(input));
    }

    #[rstest]
    fn drop_removes_every_casing() {
        assert_eq!(
            BetaPolicy::Drop.apply(headers(&[
                ("anthropic-beta", "a"),
                ("x-api-key", "k"),
                ("ANTHROPIC-BETA", "b"),
            ])),
            Ok(headers(&[("x-api-key", "k")]))
        );
    }

    #[rstest]
    #[case::bedrock(BetaProvider::Bedrock, Some("tool-search-tool-2025-10-19"))]
    #[case::bedrock_mantle(BetaProvider::BedrockMantle, Some("tool-search-tool-2025-10-19"))]
    #[case::vertex_ai(BetaProvider::VertexAi, Some("tool-search-tool-2025-10-19"))]
    #[case::anthropic(BetaProvider::Anthropic, Some("advanced-tool-use-2025-11-20"))]
    #[case::azure_ai(BetaProvider::AzureAi, Some("advanced-tool-use-2025-11-20"))]
    #[case::databricks(BetaProvider::Databricks, None)]
    #[case::bedrock_converse(BetaProvider::BedrockConverse, None)]
    fn filter_renames_per_host(
        #[case] provider: BetaProvider,
        #[case] expected_beta: Option<&str>,
    ) {
        let expected = match expected_beta {
            Some(beta) => headers(&[("x-api-key", "k"), ("anthropic-beta", beta)]),
            None => headers(&[("x-api-key", "k")]),
        };
        assert_eq!(
            BetaPolicy::Filter(provider).apply(headers(&[
                ("Anthropic-Beta", "advanced-tool-use-2025-11-20"),
                ("x-api-key", "k"),
            ])),
            Ok(expected)
        );
    }

    #[rstest]
    #[case::azure_ai(BetaProvider::AzureAi, "fast-mode-2026-02-01,example-beta-2099-01-01")]
    #[case::anthropic(BetaProvider::Anthropic, "bash_20241022")]
    #[case::vertex_ai(BetaProvider::VertexAi, " , ")]
    fn filter_removes_header_when_nothing_survives(
        #[case] provider: BetaProvider,
        #[case] beta_value: &str,
    ) {
        assert_eq!(
            BetaPolicy::Filter(provider).apply(headers(&[
                ("x-api-key", "k"),
                ("anthropic-beta", beta_value)
            ])),
            Ok(headers(&[("x-api-key", "k")]))
        );
    }

    #[rstest]
    #[case::anthropic(
        BetaProvider::Anthropic,
        "web-search-2025-03-05",
        "oauth-2025-04-20,web-search-2025-03-05,example-beta-2099-01-01",
        "oauth-2025-04-20,web-search-2025-03-05"
    )]
    #[case::vertex_ai_renames_and_deduplicates(
        BetaProvider::VertexAi,
        "advanced-tool-use-2025-11-20",
        "tool-search-tool-2025-10-19",
        "tool-search-tool-2025-10-19"
    )]
    fn filter_deduplicates_and_sorts(
        #[case] provider: BetaProvider,
        #[case] first_beta: &str,
        #[case] second_beta: &str,
        #[case] expected_beta: &str,
    ) {
        assert_eq!(
            BetaPolicy::Filter(provider).apply(headers(&[
                ("ANTHROPIC-BETA", first_beta),
                ("x-api-key", "k"),
                ("anthropic-beta", second_beta),
            ])),
            Ok(headers(&[
                ("x-api-key", "k"),
                ("anthropic-beta", expected_beta)
            ]))
        );
    }

    #[rstest]
    #[case::forward(BetaPolicy::Forward)]
    #[case::drop(BetaPolicy::Drop)]
    #[case::filter(BetaPolicy::Filter(BetaProvider::Bedrock))]
    fn headers_without_a_beta_are_untouched(#[case] policy: BetaPolicy) {
        let input = headers(&[("x-api-key", "k"), ("anthropic-version", "2023")]);
        assert_eq!(policy.apply(input.clone()), Ok(input));
    }
}
