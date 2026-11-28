//! The coding agent a request comes from, told by its `User-Agent` as the proxy tells it
//! (`is_claude_code_user_agent`, `is_codex_user_agent`, `is_github_copilot_chat_user_agent`).

use axum::http::{HeaderMap, header::USER_AGENT};

/// Claude Code's API calls (`claude-cli/`) and its own fetches (`claude-code/`).
const CLAUDE_CODE: [&str; 2] = ["claude-cli/", "claude-code/"];
const COPILOT_CHAT: &str = "GitHubCopilotChat/";
const CODEX: &str = "codex";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CodingAgent {
    ClaudeCode,
    Codex,
    CopilotChat,
}

impl CodingAgent {
    pub(crate) fn of(user_agent: &str) -> Option<Self> {
        if CLAUDE_CODE
            .iter()
            .any(|prefix| user_agent.starts_with(prefix))
        {
            return Some(Self::ClaudeCode);
        }
        if user_agent.starts_with(COPILOT_CHAT) {
            return Some(Self::CopilotChat);
        }
        user_agent
            .split_at_checked(CODEX.len())
            .is_some_and(|(name, rest)| {
                name.eq_ignore_ascii_case(CODEX) && rest.starts_with(['-', '_', ' ', '/'])
            })
            .then_some(Self::Codex)
    }

    /// The agent the request's `User-Agent` names.
    pub(crate) fn sent(headers: &HeaderMap) -> Option<Self> {
        Self::of(headers.get(USER_AGENT)?.to_str().ok()?)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::claude_codes_api_call(
        "claude-cli/2.1.0 (external, cli)",
        Some(CodingAgent::ClaudeCode)
    )]
    #[case::claude_codes_own_fetch("claude-code/2.1.0", Some(CodingAgent::ClaudeCode))]
    #[case::claude_code_later_in_the_value("agent claude-cli/2.1.0", None)]
    #[case::codex("codex_cli_rs/0.40.0", Some(CodingAgent::Codex))]
    #[case::codex_in_any_case("Codex-tui/0.40.0", Some(CodingAgent::Codex))]
    #[case::a_name_that_only_starts_like_codex("codexa/1.0", None)]
    #[case::codex_alone("codex", None)]
    #[case::copilot_chat("GitHubCopilotChat/0.26.7", Some(CodingAgent::CopilotChat))]
    #[case::copilot_chat_in_another_case("githubcopilotchat/0.26.7", None)]
    #[case::another_client("curl/8.7.1", None)]
    fn a_coding_agent_is_matched_as_the_proxy_matches_it(
        #[case] user_agent: &str,
        #[case] agent: Option<CodingAgent>,
    ) {
        assert_eq!(CodingAgent::of(user_agent), agent);
    }
}
