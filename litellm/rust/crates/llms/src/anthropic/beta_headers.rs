use std::{
    collections::HashMap,
    sync::{Arc, OnceLock, PoisonError, RwLock},
};

use litellm_http::request::{with_header, without_headers};
use litellm_llms_types::providers::anthropic::{AnthropicBeta, BETA_HEADER, BetaProvider, BetaSet};
use serde::Deserialize;

use crate::{anthropic::common_utils::existing_betas, base_llm::auth::Headers};

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
    pub fn apply(self, headers: Headers) -> Headers {
        self.apply_keeping(headers, &BetaSet::default())
    }

    /// Python's `update_headers_with_filtered_beta` with `keep`: a beta the routed model
    /// accepts is sent as the caller sent it, after the betas the filter kept.
    pub fn apply_keeping(self, headers: Headers, keep: &BetaSet) -> Headers {
        let provider = match self {
            Self::Forward => return headers,
            Self::Drop => return without_headers(headers, &[BETA_HEADER]),
            Self::Filter(provider) => provider,
        };
        let sent = existing_betas(&headers);
        let accepted: BetaSet = sent
            .iter()
            .filter(|beta| !keep.contains(beta))
            .filter_map(|beta| on_loaded_config(beta, provider))
            .collect();
        let kept: Vec<&AnthropicBeta> = sent
            .iter()
            .filter(|beta| keep.contains(beta) && !accepted.contains(beta))
            .collect();
        let headers = without_headers(headers, &[BETA_HEADER]);
        if accepted.is_empty() && kept.is_empty() {
            return headers;
        }
        let value = accepted
            .iter()
            .chain(kept)
            .map(AnthropicBeta::as_str)
            .collect::<Vec<_>>()
            .join(",");
        with_header(headers, BETA_HEADER, value)
    }
}

pub fn accepted_or_unlisted(betas: BetaSet, provider: BetaProvider) -> BetaSet {
    if betas.is_empty() {
        return betas;
    }
    let loaded = loaded_config();
    betas
        .into_iter()
        .filter_map(|beta| match (loaded.as_deref(), beta) {
            (Some(config), beta) => config.cell(&beta, provider).unwrap_or(Some(beta)),
            (None, AnthropicBeta::Other(raw)) => Some(AnthropicBeta::Other(raw)),
            (None, known) => known.on(provider),
        })
        .collect()
}

/// The beta map Python's `anthropic_beta_headers_manager` loaded, remote or bundled. Once the
/// host installs it, the `anthropic-beta` filters read it in place of the compiled columns.
#[derive(Debug, Default, Deserialize)]
pub struct BetaHeadersConfig(HashMap<String, ConfigEntry>);

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ConfigEntry {
    Column(HashMap<String, Option<String>>),
    Other(serde::de::IgnoredAny),
}

impl BetaHeadersConfig {
    fn column(&self, name: &str) -> Option<&HashMap<String, Option<String>>> {
        match self.0.get(name)? {
            ConfigEntry::Column(column) => Some(column),
            ConfigEntry::Other(_) => None,
        }
    }

    /// `None` when the provider's header columns do not list `beta`, `Some(None)` when they mark
    /// it unsupported.
    fn cell(&self, beta: &AnthropicBeta, provider: BetaProvider) -> Option<Option<AnthropicBeta>> {
        let canonical = self
            .column("provider_aliases")
            .and_then(|aliases| aliases.get(provider.as_ref()).cloned().flatten())
            .unwrap_or_else(|| provider.to_string());
        [format!("{canonical}_header_only"), canonical]
            .iter()
            .find_map(|name| self.column(name)?.get(beta.as_str()))
            .map(|name| {
                name.as_deref()
                    .map(|name| name.parse().unwrap_or_else(|never| match never {}))
            })
    }
}

static LOADED_CONFIG: RwLock<Option<Arc<BetaHeadersConfig>>> = RwLock::new(None);
static REQUEST_LOAD: OnceLock<fn()> = OnceLock::new();

/// Registers how a filter that runs before the beta map is installed asks the host to load it.
pub fn on_beta_headers_config_needed(request_load: fn()) {
    let _ = REQUEST_LOAD.set(request_load);
}

pub fn install_beta_headers_config(config: BetaHeadersConfig) {
    *LOADED_CONFIG
        .write()
        .unwrap_or_else(PoisonError::into_inner) = Some(Arc::new(config));
}

fn loaded_config() -> Option<Arc<BetaHeadersConfig>> {
    let loaded = LOADED_CONFIG
        .read()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    if loaded.is_none()
        && let Some(request_load) = REQUEST_LOAD.get()
    {
        request_load();
    }
    loaded
}

/// Python's `filter_and_transform_beta_headers` for one beta: the loaded config's cell, or the
/// compiled column when no config was installed.
pub(crate) fn on_loaded_config(
    beta: &AnthropicBeta,
    provider: BetaProvider,
) -> Option<AnthropicBeta> {
    match loaded_config() {
        Some(config) => config.cell(beta, provider).flatten(),
        None => beta.on(provider),
    }
}

#[cfg(test)]
mod tests {
    use crate::anthropic::beta_headers::{BetaHeadersConfig, BetaPolicy};
    use litellm_llms_types::providers::anthropic::{AnthropicBeta, BetaProvider, BetaSet};
    use rstest::rstest;
    use serde_json::json;

    fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    #[rstest]
    fn forward_keeps_headers_as_is() {
        let input = headers(&[
            (
                "Anthropic-Beta",
                "example-beta-2099-01-01,web-search-2025-03-05",
            ),
            ("x-api-key", "k"),
        ]);
        assert_eq!(BetaPolicy::Forward.apply(input.clone()), input);
    }

    #[rstest]
    fn drop_removes_every_casing() {
        assert_eq!(
            BetaPolicy::Drop.apply(headers(&[
                ("anthropic-beta", "a"),
                ("x-api-key", "k"),
                ("ANTHROPIC-BETA", "b"),
            ])),
            headers(&[("x-api-key", "k")])
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
            expected
        );
    }

    #[rstest]
    #[case::azure_ai(
        BetaProvider::AzureAi,
        "advisor-tool-2026-03-01,example-beta-2099-01-01"
    )]
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
            headers(&[("x-api-key", "k")])
        );
    }

    #[rstest]
    #[case::kept_after_the_filtered_betas(
        "per-turn-control-2026-07-01,advanced-tool-use-2025-11-20,example-beta-2099-01-01",
        Some("tool-search-tool-2025-10-19,per-turn-control-2026-07-01")
    )]
    #[case::kept_alone(
        "per-turn-control-2026-07-01,example-beta-2099-01-01",
        Some("per-turn-control-2026-07-01")
    )]
    #[case::not_sent("advanced-tool-use-2025-11-20", Some("tool-search-tool-2025-10-19"))]
    #[case::nothing_survives("example-beta-2099-01-01", None)]
    fn filter_sends_a_kept_beta_as_sent_after_the_filtered_ones(
        #[case] sent: &str,
        #[case] expected: Option<&str>,
    ) {
        let keep: BetaSet = "per-turn-control-2026-07-01"
            .parse()
            .unwrap_or_else(|never| match never {});
        assert_eq!(
            BetaPolicy::Filter(BetaProvider::Bedrock).apply_keeping(
                headers(&[("x-api-key", "k"), ("anthropic-beta", sent)]),
                &keep
            ),
            [("x-api-key", "k")]
                .into_iter()
                .chain(expected.map(|beta| ("anthropic-beta", beta)))
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect::<Vec<_>>()
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
            headers(&[("x-api-key", "k"), ("anthropic-beta", expected_beta)])
        );
    }

    #[rstest]
    #[case::forward(BetaPolicy::Forward)]
    #[case::drop(BetaPolicy::Drop)]
    #[case::filter(BetaPolicy::Filter(BetaProvider::Bedrock))]
    fn headers_without_a_beta_are_untouched(#[case] policy: BetaPolicy) {
        let input = headers(&[("x-api-key", "k"), ("anthropic-version", "2023")]);
        assert_eq!(policy.apply(input.clone()), input);
    }

    #[rstest]
    #[case::renamed(
        "advanced-tool-use-2025-11-20",
        BetaProvider::AzureAi,
        Some(Some("tool-search-tool-2025-10-19"))
    )]
    #[case::marked_unsupported("web-search-2025-03-05", BetaProvider::AzureAi, Some(None))]
    #[case::unlisted("oauth-2025-04-20", BetaProvider::AzureAi, None)]
    #[case::header_only_column_wins(
        "web-search-2025-03-05",
        BetaProvider::VertexAi,
        Some(Some("web-search-2025-03-05"))
    )]
    #[case::aliased_column("web-search-2025-03-05", BetaProvider::Databricks, Some(None))]
    fn loaded_config_cell_follows_python_column_lookup(
        #[case] beta: &str,
        #[case] provider: BetaProvider,
        #[case] expected: Option<Option<&str>>,
    ) {
        let config: BetaHeadersConfig = serde_json::from_value(json!({
            "description": "remote",
            "provider_aliases": {"databricks": "azure_ai"},
            "azure_ai": {
                "advanced-tool-use-2025-11-20": "tool-search-tool-2025-10-19",
                "web-search-2025-03-05": null,
            },
            "vertex_ai": {"web-search-2025-03-05": null},
            "vertex_ai_header_only": {"web-search-2025-03-05": "web-search-2025-03-05"},
        }))
        .unwrap();
        let beta: AnthropicBeta = beta.parse().unwrap_or_else(|never| match never {});
        assert_eq!(
            config.cell(&beta, provider),
            expected
                .map(|name| name.map(|name| name.parse().unwrap_or_else(|never| match never {})))
        );
    }
}
