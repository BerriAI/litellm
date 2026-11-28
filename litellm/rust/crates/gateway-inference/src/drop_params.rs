//! Whether a call drops what its provider rejects, decided here and nowhere else.
//!
//! The narrowest `drop_params` written decides: the request's own, then the deployment's, then
//! `litellm_settings.drop_params`. With none written, a coding agent's call drops and any other
//! call does not. A value that is no flag, `null` included, is not written.
//!
//! The proxy differs in four places, and none is kept:
//! - It reads `litellm.drop_params or drop_params`, so the setting on keeps dropping over a
//!   request or a deployment that writes `false`
//! - It writes a coding agent's default into the request, so the default wins over a
//!   deployment's own `drop_params: false`
//! - It lays the request over the deployment key by key, so a request's `null` or a value that
//!   is no flag turns the deployment's `true` off, and turns a coding agent's default off
//! - It reads the setting for a coding agent's default from the config alone, where
//!   `LITELLM_DROP_PARAMS=false` turns the default off here too, since the config crate reads
//!   that variable as the key's value

use litellm_python_compat::{flags::lax_bool, json::from_json};
use serde_json::Value;

use crate::coding_agent::CodingAgent;

/// Each place a call's `drop_params` is written, narrowest first, `None` where it was not.
pub(crate) struct Sources {
    pub request: Option<bool>,
    pub deployment: Option<bool>,
    /// `litellm_settings.drop_params`, with its variable under it.
    pub global: Option<bool>,
    pub client: Option<CodingAgent>,
}

pub(crate) fn drops(sources: &Sources) -> bool {
    sources
        .request
        .or(sources.deployment)
        .or(sources.global)
        .unwrap_or(sources.client.is_some())
}

/// The flag a request body's `drop_params` writes, read as the config's flags are.
pub(crate) fn written(value: &Value) -> Option<bool> {
    lax_bool(&from_json(value.clone()))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    const AGENT: Option<CodingAgent> = Some(CodingAgent::ClaudeCode);

    #[rstest]
    #[case::nothing_written(None, None, None, None, false)]
    #[case::the_setting(None, None, Some(true), None, true)]
    #[case::the_deployment_over_a_setting_that_is_off(None, Some(true), Some(false), None, true)]
    #[case::the_deployment_over_a_setting_that_is_on(None, Some(false), Some(true), None, false)]
    #[case::the_request_over_a_deployment_that_is_off(Some(true), Some(false), None, None, true)]
    #[case::the_request_over_a_deployment_that_is_on(Some(false), Some(true), None, None, false)]
    #[case::the_request_over_a_setting_that_is_on(Some(false), None, Some(true), None, false)]
    #[case::a_coding_agent(None, None, None, AGENT, true)]
    #[case::codex(None, None, None, Some(CodingAgent::Codex), true)]
    #[case::copilot_chat(None, None, None, Some(CodingAgent::CopilotChat), true)]
    #[case::a_coding_agent_under_a_setting_that_is_off(None, None, Some(false), AGENT, false)]
    #[case::a_coding_agent_to_a_deployment_that_is_off(None, Some(false), None, AGENT, false)]
    #[case::a_coding_agent_turning_it_off_for_its_call(Some(false), None, None, AGENT, false)]
    fn the_narrowest_drop_params_written_decides(
        #[case] request: Option<bool>,
        #[case] deployment: Option<bool>,
        #[case] global: Option<bool>,
        #[case] client: Option<CodingAgent>,
        #[case] expected: bool,
    ) {
        let sources = Sources {
            request,
            deployment,
            global,
            client,
        };

        assert_eq!(drops(&sources), expected);
    }

    #[rstest]
    #[case::on(json!(true), Some(true))]
    #[case::off(json!(false), Some(false))]
    #[case::a_flag_word(json!(" yes "), Some(true))]
    #[case::a_word_that_is_no_flag(json!("maybe"), None)]
    #[case::a_number_that_is_no_flag(json!(2), None)]
    #[case::null(json!(null), None)]
    fn a_requests_drop_params_is_written_only_as_a_flag(
        #[case] value: Value,
        #[case] flag: Option<bool>,
    ) {
        assert_eq!(written(&value), flag);
    }
}
