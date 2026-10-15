use litellm_llms_types::{
    formats::messages::{ContentBlockType, MessagesRequest, ThinkingConfig, ThinkingDisplay},
    providers::anthropic::{AnthropicBeta, BetaSet},
    recognized::Recognized,
};
use serde_json::Value;

include!("beta_detection_generated.rs");

fn thinking_display(request: &MessagesRequest) -> Option<&ThinkingDisplay> {
    match request.params.thinking.as_ref()?.known()? {
        ThinkingConfig::Enabled(thinking) => thinking.display.as_ref()?.known(),
        ThinkingConfig::Adaptive(thinking) => thinking.display.as_ref()?.known(),
        ThinkingConfig::Disabled(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde::Deserialize;
    use std::collections::BTreeSet;

    #[derive(Deserialize)]
    struct Case {
        name: String,
        request: MessagesRequest,
        expected: BTreeSet<String>,
    }

    #[rstest]
    #[case::baseline("baseline")]
    #[case::adaptive_updates("adaptive_updates")]
    #[case::enabled_updates("enabled_updates")]
    #[case::disabled_updates("disabled_updates")]
    #[case::summarized_thinking("summarized_thinking")]
    #[case::omitted_thinking("omitted_thinking")]
    #[case::missing_display("missing_display")]
    #[case::unknown_thinking("unknown_thinking")]
    #[case::null_thinking("null_thinking")]
    #[case::scalar_thinking("scalar_thinking")]
    #[case::wrong_display_type("wrong_display_type")]
    #[case::tool_addition("tool_addition")]
    #[case::tool_removal("tool_removal")]
    #[case::user_tool_change("user_tool_change")]
    #[case::wrong_target("wrong_target")]
    #[case::wrong_action("wrong_action")]
    #[case::split_messages("split_messages")]
    #[case::split_blocks("split_blocks")]
    #[case::text_content("text_content")]
    #[case::null_target("null_target")]
    #[case::combined("combined")]
    #[case::later_tool_change("later_tool_change")]
    fn shared_requests_select_the_expected_betas(#[case] name: &str) {
        let cases: Vec<Case> = serde_json::from_str(include_str!(
            "../../../../../tests/fixtures/anthropic_beta_detection.json"
        ))
        .unwrap();
        let case = cases.into_iter().find(|case| case.name == name).unwrap();
        let actual: BTreeSet<String> = detect_anthropic_betas(&case.request)
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(actual, case.expected, "{}", case.name);
    }
}
