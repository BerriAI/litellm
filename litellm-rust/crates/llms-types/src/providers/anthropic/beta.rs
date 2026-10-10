use std::{
    cmp::Ordering,
    collections::BTreeSet,
    convert::Infallible,
    fmt,
    hash::{Hash, Hasher},
    str::FromStr,
};

use strum::VariantArray;

/// A provider column of `litellm/anthropic_beta_headers_config.json`: which betas a host accepts
/// and under which name.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    strum::AsRefStr,
    strum::Display,
    strum::EnumString,
    strum::VariantArray,
)]
pub enum BetaProvider {
    #[strum(serialize = "anthropic")]
    Anthropic,
    #[strum(serialize = "azure_ai")]
    AzureAi,
    #[strum(serialize = "bedrock_converse")]
    BedrockConverse,
    #[strum(serialize = "bedrock")]
    Bedrock,
    #[strum(serialize = "bedrock_mantle")]
    BedrockMantle,
    #[strum(serialize = "vertex_ai")]
    VertexAi,
    #[strum(serialize = "databricks")]
    Databricks,
}

/// One value of the `anthropic-beta` header. Equality, ordering and hashing follow the wire
/// string, so a value parsed from a caller's header never disagrees with the matching variant.
#[derive(Clone, Debug, strum::AsRefStr, strum::Display, strum::EnumString)]
pub enum AnthropicBeta {
    #[strum(serialize = "oauth-2025-04-20")]
    Oauth20250420,
    #[strum(serialize = "web-fetch-2025-09-10")]
    WebFetch20250910,
    #[strum(serialize = "web-search-2025-03-05")]
    WebSearch20250305,
    #[strum(serialize = "context-management-2025-06-27")]
    ContextManagement20250627,
    #[strum(serialize = "compact-2026-01-12")]
    Compact20260112,
    #[strum(serialize = "compact-2026-09-04")]
    Compact20260904,
    #[strum(serialize = "structured-outputs-2025-11-13")]
    StructuredOutputs20251113,
    #[strum(serialize = "advanced-tool-use-2025-11-20")]
    AdvancedToolUse20251120,
    #[strum(serialize = "fast-mode-2026-02-01")]
    FastMode20260201,
    #[strum(serialize = "advisor-tool-2026-03-01")]
    AdvisorTool20260301,
    #[strum(serialize = "per-turn-control-2026-07-01")]
    PerTurnControl20260701,
    #[strum(serialize = "dangerous-tool-use-2026-09-03")]
    DangerousToolUse20260903,
    #[strum(serialize = "bash_20241022")]
    Bash20241022,
    #[strum(serialize = "bash_20250124")]
    Bash20250124,
    #[strum(serialize = "claude-code-20250219")]
    ClaudeCode20250219,
    #[strum(serialize = "code-execution-2025-08-25")]
    CodeExecution20250825,
    #[strum(serialize = "computer-use-2025-01-24")]
    ComputerUse20250124,
    #[strum(serialize = "computer-use-2025-11-24")]
    ComputerUse20251124,
    #[strum(serialize = "context-1m-2025-08-07")]
    Context1m20250807,
    #[strum(serialize = "effort-2025-11-24")]
    Effort20251124,
    #[strum(serialize = "files-api-2025-04-14")]
    FilesApi20250414,
    #[strum(serialize = "fine-grained-tool-streaming-2025-05-14")]
    FineGrainedToolStreaming20250514,
    #[strum(serialize = "interleaved-thinking-2025-05-14")]
    InterleavedThinking20250514,
    #[strum(serialize = "mcp-client-2025-04-04")]
    McpClient20250404,
    #[strum(serialize = "mcp-client-2025-11-20")]
    McpClient20251120,
    #[strum(serialize = "mcp-servers-2025-12-04")]
    McpServers20251204,
    #[strum(serialize = "mid-conversation-output-config-2026-07-01")]
    MidConversationOutputConfig20260701,
    #[strum(serialize = "mid-conversation-tool-changes-2026-07-01")]
    MidConversationToolChanges20260701,
    #[strum(serialize = "output-128k-2025-02-19")]
    Output128k20250219,
    #[strum(serialize = "prompt-caching-scope-2026-01-05")]
    PromptCachingScope20260105,
    #[strum(serialize = "skills-2025-10-02")]
    Skills20251002,
    #[strum(serialize = "structured-output-2024-03-01")]
    StructuredOutput20240301,
    #[strum(serialize = "text_editor_20241022")]
    TextEditor20241022,
    #[strum(serialize = "text_editor_20250124")]
    TextEditor20250124,
    #[strum(serialize = "thinking-binding-controls-2026-08-01")]
    ThinkingBindingControls20260801,
    #[strum(serialize = "thinking-display-updates-2026-08-18")]
    ThinkingDisplayUpdates20260818,
    #[strum(serialize = "token-efficient-tools-2025-02-19")]
    TokenEfficientTools20250219,
    #[strum(serialize = "tool-examples-2025-10-29")]
    ToolExamples20251029,
    #[strum(serialize = "tool-search-tool-2025-10-19")]
    ToolSearchTool20251019,
    #[strum(serialize = "inline-tools-2026-09-15")]
    InlineTools20260915,
    #[strum(default, transparent)]
    Other(String),
}

impl AnthropicBeta {
    pub const KNOWN: [Self; 40] = [
        Self::Oauth20250420,
        Self::WebFetch20250910,
        Self::WebSearch20250305,
        Self::ContextManagement20250627,
        Self::Compact20260112,
        Self::Compact20260904,
        Self::StructuredOutputs20251113,
        Self::AdvancedToolUse20251120,
        Self::FastMode20260201,
        Self::AdvisorTool20260301,
        Self::PerTurnControl20260701,
        Self::DangerousToolUse20260903,
        Self::Bash20241022,
        Self::Bash20250124,
        Self::ClaudeCode20250219,
        Self::CodeExecution20250825,
        Self::ComputerUse20250124,
        Self::ComputerUse20251124,
        Self::Context1m20250807,
        Self::Effort20251124,
        Self::FilesApi20250414,
        Self::FineGrainedToolStreaming20250514,
        Self::InterleavedThinking20250514,
        Self::McpClient20250404,
        Self::McpClient20251120,
        Self::McpServers20251204,
        Self::MidConversationOutputConfig20260701,
        Self::MidConversationToolChanges20260701,
        Self::Output128k20250219,
        Self::PromptCachingScope20260105,
        Self::Skills20251002,
        Self::StructuredOutput20240301,
        Self::TextEditor20241022,
        Self::TextEditor20250124,
        Self::ThinkingBindingControls20260801,
        Self::ThinkingDisplayUpdates20260818,
        Self::TokenEfficientTools20250219,
        Self::ToolExamples20251029,
        Self::ToolSearchTool20251019,
        Self::InlineTools20260915,
    ];

    pub fn as_str(&self) -> &str {
        self.as_ref()
    }

    pub fn on(&self, provider: BetaProvider) -> Option<Self> {
        let resolved = match self {
            Self::Other(raw) => raw.parse().unwrap(),
            _ => self.clone(),
        };
        if resolved.rejected_by().contains(&provider) {
            return None;
        }
        Some(match (resolved, provider) {
            (
                Self::AdvancedToolUse20251120,
                BetaProvider::Bedrock | BetaProvider::BedrockMantle | BetaProvider::VertexAi,
            ) => Self::ToolSearchTool20251019,
            (beta, _) => beta,
        })
    }

    fn rejected_by(&self) -> &'static [BetaProvider] {
        match self {
            Self::AdvancedToolUse20251120 | Self::ContextManagement20250627 => {
                &[BetaProvider::BedrockConverse]
            }
            Self::AdvisorTool20260301 | Self::Compact20260904 => &[
                BetaProvider::AzureAi,
                BetaProvider::BedrockConverse,
                BetaProvider::Bedrock,
                BetaProvider::BedrockMantle,
                BetaProvider::VertexAi,
                BetaProvider::Databricks,
            ],
            Self::Bash20241022
            | Self::Bash20250124
            | Self::McpServers20251204
            | Self::StructuredOutput20240301
            | Self::TextEditor20241022
            | Self::TextEditor20250124 => BetaProvider::VARIANTS,
            Self::ClaudeCode20250219 | Self::ToolExamples20251029 => &[
                BetaProvider::Anthropic,
                BetaProvider::AzureAi,
                BetaProvider::BedrockConverse,
                BetaProvider::Bedrock,
                BetaProvider::VertexAi,
                BetaProvider::Databricks,
            ],
            Self::CodeExecution20250825
            | Self::FilesApi20250414
            | Self::McpClient20250404
            | Self::McpClient20251120
            | Self::PromptCachingScope20260105
            | Self::Skills20251002
            | Self::WebFetch20250910 => &[
                BetaProvider::BedrockConverse,
                BetaProvider::Bedrock,
                BetaProvider::BedrockMantle,
                BetaProvider::VertexAi,
            ],
            Self::Compact20260112 => &[BetaProvider::AzureAi, BetaProvider::BedrockConverse],
            Self::ComputerUse20250124 | Self::ComputerUse20251124 | Self::Context1m20250807 => &[],
            Self::DangerousToolUse20260903 => {
                &[BetaProvider::BedrockConverse, BetaProvider::Databricks]
            }
            Self::Effort20251124 => &[BetaProvider::VertexAi],
            Self::FastMode20260201 | Self::Oauth20250420 => &[
                BetaProvider::AzureAi,
                BetaProvider::BedrockConverse,
                BetaProvider::Bedrock,
                BetaProvider::BedrockMantle,
                BetaProvider::VertexAi,
            ],
            Self::FineGrainedToolStreaming20250514 => {
                &[BetaProvider::AzureAi, BetaProvider::VertexAi]
            }
            Self::InterleavedThinking20250514 | Self::WebSearch20250305 => {
                &[BetaProvider::BedrockConverse, BetaProvider::Bedrock]
            }
            Self::MidConversationOutputConfig20260701
            | Self::MidConversationToolChanges20260701 => &[
                BetaProvider::AzureAi,
                BetaProvider::BedrockConverse,
                BetaProvider::BedrockMantle,
                BetaProvider::VertexAi,
                BetaProvider::Databricks,
            ],
            Self::Output128k20250219 | Self::TokenEfficientTools20250219 => &[
                BetaProvider::AzureAi,
                BetaProvider::BedrockConverse,
                BetaProvider::Bedrock,
                BetaProvider::VertexAi,
            ],
            Self::PerTurnControl20260701 => &[
                BetaProvider::BedrockConverse,
                BetaProvider::Bedrock,
                BetaProvider::Databricks,
            ],
            Self::StructuredOutputs20251113 => &[BetaProvider::Bedrock, BetaProvider::VertexAi],
            Self::ThinkingBindingControls20260801 => &[BetaProvider::AzureAi],
            Self::ThinkingDisplayUpdates20260818 => &[BetaProvider::Databricks],
            Self::ToolSearchTool20251019 => &[
                BetaProvider::Anthropic,
                BetaProvider::AzureAi,
                BetaProvider::BedrockConverse,
                BetaProvider::Databricks,
            ],
            Self::InlineTools20260915 => &[
                BetaProvider::AzureAi,
                BetaProvider::BedrockConverse,
                BetaProvider::Bedrock,
                BetaProvider::BedrockMantle,
                BetaProvider::Databricks,
            ],
            Self::Other(_) => BetaProvider::VARIANTS,
        }
    }
}

impl PartialEq for AnthropicBeta {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for AnthropicBeta {}

impl Hash for AnthropicBeta {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl PartialOrd for AnthropicBeta {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AnthropicBeta {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

/// The values of one `anthropic-beta` header: sorted, deduplicated, comma-joined on the wire.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BetaSet(BTreeSet<AnthropicBeta>);

impl BetaSet {
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn contains(&self, beta: &AnthropicBeta) -> bool {
        self.0.contains(beta)
    }

    pub fn iter(&self) -> impl Iterator<Item = &AnthropicBeta> {
        self.0.iter()
    }

    pub fn union(self, other: Self) -> Self {
        self.0.into_iter().chain(other.0).collect()
    }
}

impl FromIterator<AnthropicBeta> for BetaSet {
    fn from_iter<I: IntoIterator<Item = AnthropicBeta>>(betas: I) -> Self {
        Self(betas.into_iter().collect())
    }
}

impl IntoIterator for BetaSet {
    type Item = AnthropicBeta;
    type IntoIter = std::collections::btree_set::IntoIter<AnthropicBeta>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl FromStr for BetaSet {
    type Err = Infallible;

    fn from_str(header: &str) -> Result<Self, Infallible> {
        Ok(header
            .split(',')
            .map(str::trim)
            .filter(|piece| !piece.is_empty())
            .map(|piece| AnthropicBeta::from_str(piece).unwrap_or_else(|never| match never {}))
            .collect())
    }
}

impl fmt::Display for BetaSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut betas = self.0.iter();
        let Some(first) = betas.next() else {
            return Ok(());
        };
        f.write_str(first.as_str())?;
        betas.try_for_each(|beta| write!(f, ",{beta}"))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use indexmap::IndexMap;
    use rstest::rstest;

    use super::*;

    fn beta_headers_config() -> IndexMap<String, serde_json::Value> {
        serde_json::from_str(include_str!(
            "../../../../../../litellm/anthropic_beta_headers_config.json"
        ))
        .unwrap()
    }

    fn set(header: &str) -> BetaSet {
        header.parse().unwrap_or_else(|never| match never {})
    }

    #[rstest]
    fn every_known_beta_parses_back_to_itself(
        #[values(
            AnthropicBeta::Oauth20250420,
            AnthropicBeta::WebFetch20250910,
            AnthropicBeta::WebSearch20250305,
            AnthropicBeta::ContextManagement20250627,
            AnthropicBeta::Compact20260112,
            AnthropicBeta::Compact20260904,
            AnthropicBeta::StructuredOutputs20251113,
            AnthropicBeta::AdvancedToolUse20251120,
            AnthropicBeta::FastMode20260201,
            AnthropicBeta::AdvisorTool20260301,
            AnthropicBeta::PerTurnControl20260701,
            AnthropicBeta::DangerousToolUse20260903,
            AnthropicBeta::Bash20241022,
            AnthropicBeta::Bash20250124,
            AnthropicBeta::ClaudeCode20250219,
            AnthropicBeta::CodeExecution20250825,
            AnthropicBeta::ComputerUse20250124,
            AnthropicBeta::ComputerUse20251124,
            AnthropicBeta::Context1m20250807,
            AnthropicBeta::Effort20251124,
            AnthropicBeta::FilesApi20250414,
            AnthropicBeta::FineGrainedToolStreaming20250514,
            AnthropicBeta::InterleavedThinking20250514,
            AnthropicBeta::McpClient20250404,
            AnthropicBeta::McpClient20251120,
            AnthropicBeta::McpServers20251204,
            AnthropicBeta::MidConversationOutputConfig20260701,
            AnthropicBeta::MidConversationToolChanges20260701,
            AnthropicBeta::Output128k20250219,
            AnthropicBeta::PromptCachingScope20260105,
            AnthropicBeta::Skills20251002,
            AnthropicBeta::StructuredOutput20240301,
            AnthropicBeta::TextEditor20241022,
            AnthropicBeta::TextEditor20250124,
            AnthropicBeta::ThinkingBindingControls20260801,
            AnthropicBeta::ThinkingDisplayUpdates20260818,
            AnthropicBeta::TokenEfficientTools20250219,
            AnthropicBeta::ToolExamples20251029,
            AnthropicBeta::ToolSearchTool20251019,
            AnthropicBeta::InlineTools20260915
        )]
        beta: AnthropicBeta,
    ) {
        let parsed: AnthropicBeta = beta.as_str().parse().unwrap();
        assert!(!matches!(parsed, AnthropicBeta::Other(_)));
        assert_eq!(parsed, beta);
        assert!(AnthropicBeta::KNOWN.contains(&beta));
    }

    #[rstest]
    fn known_betas_are_exactly_the_config_keys() {
        let config = beta_headers_config();
        let config_keys: BTreeSet<String> = config
            .iter()
            .filter(|(provider, _)| provider.as_str() != "description")
            .flat_map(|(_, column)| {
                column
                    .as_object()
                    .into_iter()
                    .flat_map(|values| values.keys().cloned())
            })
            .collect();
        let known_keys: BTreeSet<String> = AnthropicBeta::KNOWN
            .iter()
            .map(|beta| beta.as_str().to_string())
            .collect();
        assert_eq!(known_keys, config_keys);
    }

    #[rstest]
    #[case::anthropic(BetaProvider::Anthropic)]
    #[case::azure_ai(BetaProvider::AzureAi)]
    #[case::bedrock_converse(BetaProvider::BedrockConverse)]
    #[case::bedrock(BetaProvider::Bedrock)]
    #[case::bedrock_mantle(BetaProvider::BedrockMantle)]
    #[case::vertex_ai(BetaProvider::VertexAi)]
    #[case::databricks(BetaProvider::Databricks)]
    fn provider_columns_match_beta_provider_variants(#[case] provider: BetaProvider) {
        let config = beta_headers_config();
        let config_columns: Vec<String> = config
            .keys()
            .filter(|column| column.as_str() != "description")
            .cloned()
            .collect();
        let beta_provider_columns: Vec<String> = BetaProvider::VARIANTS
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(config_columns, beta_provider_columns);
        assert_eq!(
            provider.to_string().parse::<BetaProvider>().unwrap(),
            provider
        );
    }

    #[rstest]
    fn on_matches_every_config_cell() {
        let config = beta_headers_config();
        for provider in BetaProvider::VARIANTS {
            for beta in &AnthropicBeta::KNOWN {
                let expected = config
                    .get(provider.as_ref())
                    .and_then(serde_json::Value::as_object)
                    .and_then(|column| column.get(beta.as_str()))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string);
                let actual = beta.on(*provider).map(|name| name.to_string());
                assert_eq!(actual, expected, "provider {provider}, beta {beta}");
                if matches!(beta, AnthropicBeta::AdvancedToolUse20251120)
                    && matches!(
                        provider,
                        BetaProvider::Bedrock
                            | BetaProvider::BedrockMantle
                            | BetaProvider::VertexAi
                    )
                {
                    let parsed: AnthropicBeta = actual.as_deref().unwrap().parse().unwrap();
                    assert!(AnthropicBeta::KNOWN.contains(&parsed));
                    assert!(!matches!(parsed, AnthropicBeta::Other(_)));
                }
            }
        }
    }

    #[rstest]
    #[case::renamed(
        "advanced-tool-use-2025-11-20",
        BetaProvider::Bedrock,
        Some(AnthropicBeta::ToolSearchTool20251019)
    )]
    #[case::kept(
        "oauth-2025-04-20",
        BetaProvider::Anthropic,
        Some(AnthropicBeta::Oauth20250420)
    )]
    #[case::rejected("effort-2025-11-24", BetaProvider::VertexAi, None)]
    #[case::unknown("example-beta-2099-01-01", BetaProvider::Anthropic, None)]
    fn on_resolves_known_spellings_held_as_other(
        #[case] raw: &str,
        #[case] provider: BetaProvider,
        #[case] expected: Option<AnthropicBeta>,
    ) {
        let actual = AnthropicBeta::Other(raw.to_string()).on(provider);
        assert_eq!(actual, expected);
        assert!(!matches!(actual, Some(AnthropicBeta::Other(_))));
    }

    #[rstest]
    #[case::anthropic(BetaProvider::Anthropic)]
    #[case::azure_ai(BetaProvider::AzureAi)]
    #[case::bedrock_converse(BetaProvider::BedrockConverse)]
    #[case::bedrock(BetaProvider::Bedrock)]
    #[case::bedrock_mantle(BetaProvider::BedrockMantle)]
    #[case::vertex_ai(BetaProvider::VertexAi)]
    #[case::databricks(BetaProvider::Databricks)]
    fn unknown_beta_is_rejected_by_every_provider(#[case] provider: BetaProvider) {
        assert_eq!(
            AnthropicBeta::Other("example-beta-2099-01-01".into()).on(provider),
            None
        );
    }

    #[test]
    fn unknown_values_are_kept_verbatim() {
        let parsed: AnthropicBeta = "claude-code-20250219".parse().unwrap();
        assert_eq!(
            parsed,
            AnthropicBeta::Other("claude-code-20250219".to_string())
        );
        assert_eq!(parsed.to_string(), "claude-code-20250219");
    }

    #[test]
    fn a_known_value_spelled_as_other_is_the_same_beta() {
        let spelled_out = AnthropicBeta::Other("compact-2026-01-12".to_string());
        assert_eq!(spelled_out, AnthropicBeta::Compact20260112);
        assert_eq!(
            spelled_out.cmp(&AnthropicBeta::Compact20260112),
            Ordering::Equal
        );
        assert_eq!(
            BetaSet::from_iter([spelled_out, AnthropicBeta::Compact20260112]).to_string(),
            "compact-2026-01-12"
        );
    }

    #[rstest]
    #[case::empty("", "")]
    #[case::blank_pieces(" , ,", "")]
    #[case::single("b", "b")]
    #[case::sorted("c,a", "a,c")]
    #[case::trimmed_and_deduplicated("b, a ,b", "a,b")]
    #[case::blank_pieces_skipped("a,,b", "a,b")]
    #[case::known_and_unknown_sort_together(
        "web-search-2025-03-05,claude-code-20250219,fast-mode-2026-02-01",
        "claude-code-20250219,fast-mode-2026-02-01,web-search-2025-03-05"
    )]
    fn header_values_round_trip_sorted_and_deduplicated(#[case] header: &str, #[case] wire: &str) {
        assert_eq!(set(header).to_string(), wire);
        assert_eq!(set(header).is_empty(), wire.is_empty());
    }

    #[rstest]
    #[case::disjoint("a,c", "b", "a,b,c")]
    #[case::overlapping("a,b", "b,c", "a,b,c")]
    #[case::empty_right("a", "", "a")]
    #[case::empty_left("", "a", "a")]
    fn union_merges_both_sides(#[case] left: &str, #[case] right: &str, #[case] wire: &str) {
        assert_eq!(set(left).union(set(right)).to_string(), wire);
    }

    #[test]
    fn contains_matches_by_wire_value() {
        let betas = set("oauth-2025-04-20,claude-code-20250219");
        assert!(betas.contains(&AnthropicBeta::Oauth20250420));
        assert!(betas.contains(&AnthropicBeta::Other("claude-code-20250219".into())));
        assert!(!betas.contains(&AnthropicBeta::FastMode20260201));
    }
}
