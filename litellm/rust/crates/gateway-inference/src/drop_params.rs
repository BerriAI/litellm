//! Whether a call drops what its provider rejects, decided here and nowhere else.
//!
//! The narrowest `drop_params` written decides: the request's own, then the deployment's, then
//! `litellm_settings.drop_params`. With none written the call does not drop. A value that is no
//! flag, `null` included, is not written.
//!
//! The proxy differs in two places, and neither is kept:
//! - It reads `litellm.drop_params or drop_params`, so the setting on keeps dropping over a
//!   request or a deployment that writes `false`
//! - It lays the request over the deployment key by key, so a request's `null` or a value that
//!   is no flag turns the deployment's `true` off

use litellm_python_compat::{flags::lax_bool, json::from_json};
use serde_json::Value;

/// Each place a call's `drop_params` is written, narrowest first, `None` where it was not.
pub(crate) struct Sources {
    pub request: Option<bool>,
    pub deployment: Option<bool>,
    /// `litellm_settings.drop_params`, with its variable under it.
    pub global: Option<bool>,
}

pub(crate) fn drops(sources: &Sources) -> bool {
    sources
        .request
        .or(sources.deployment)
        .or(sources.global)
        .unwrap_or(false)
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

    #[rstest]
    #[case::nothing_written(None, None, None, false)]
    #[case::the_setting(None, None, Some(true), true)]
    #[case::the_deployment_over_a_setting_that_is_off(None, Some(true), Some(false), true)]
    #[case::the_deployment_over_a_setting_that_is_on(None, Some(false), Some(true), false)]
    #[case::the_request_over_a_deployment_that_is_off(Some(true), Some(false), None, true)]
    #[case::the_request_over_a_deployment_that_is_on(Some(false), Some(true), None, false)]
    #[case::the_request_over_a_setting_that_is_on(Some(false), None, Some(true), false)]
    fn the_narrowest_drop_params_written_decides(
        #[case] request: Option<bool>,
        #[case] deployment: Option<bool>,
        #[case] global: Option<bool>,
        #[case] expected: bool,
    ) {
        let sources = Sources {
            request,
            deployment,
            global,
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
