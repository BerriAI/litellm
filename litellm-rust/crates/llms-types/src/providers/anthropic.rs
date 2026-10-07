use std::{
    cmp::Ordering,
    collections::BTreeSet,
    convert::Infallible,
    fmt,
    hash::{Hash, Hasher},
    str::FromStr,
};

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
    #[strum(default, transparent)]
    Other(String),
}

impl AnthropicBeta {
    pub const KNOWN: [Self; 39] = [
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
    ];

    pub fn as_str(&self) -> &str {
        self.as_ref()
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
    use rstest::rstest;

    use super::*;

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
            AnthropicBeta::ClaudeCode20250219,
            AnthropicBeta::Bash20241022,
            AnthropicBeta::ToolSearchTool20251019
        )]
        beta: AnthropicBeta,
    ) {
        let parsed: AnthropicBeta = beta.as_str().parse().unwrap();
        assert!(!matches!(parsed, AnthropicBeta::Other(_)));
        assert_eq!(parsed, beta);
        assert!(AnthropicBeta::KNOWN.contains(&beta));
    }

    #[test]
    fn unknown_values_are_kept_verbatim() {
        let parsed: AnthropicBeta = "example-beta-2099-01-01".parse().unwrap();
        assert_eq!(
            parsed,
            AnthropicBeta::Other("example-beta-2099-01-01".to_string())
        );
        assert_eq!(parsed.to_string(), "example-beta-2099-01-01");
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
        "web-search-2025-03-05,example-beta-2099-01-01,fast-mode-2026-02-01",
        "example-beta-2099-01-01,fast-mode-2026-02-01,web-search-2025-03-05"
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
        let betas = set("oauth-2025-04-20,example-beta-2099-01-01");
        assert!(betas.contains(&AnthropicBeta::Oauth20250420));
        assert!(betas.contains(&AnthropicBeta::Other("example-beta-2099-01-01".into())));
        assert!(!betas.contains(&AnthropicBeta::FastMode20260201));
    }
}
