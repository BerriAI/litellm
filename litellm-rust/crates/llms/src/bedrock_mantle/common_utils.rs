use litellm_auth_aws::constants::{
    AWS_ACCESS_KEY_ID, AWS_BEARER_TOKEN_BEDROCK, AWS_EXTERNAL_ID, AWS_PROFILE_NAME, AWS_REGION,
    AWS_REGION_NAME, AWS_ROLE_NAME, AWS_SECRET_ACCESS_KEY, AWS_SESSION_NAME, AWS_SESSION_TOKEN,
    AWS_STS_ENDPOINT, AWS_WEB_IDENTITY_TOKEN,
};

use crate::bedrock::messages::mantle::transformation::{
    BEDROCK_MANTLE_API_BASE_ENV, MantleSettings,
};

pub const BEDROCK_MANTLE_API_KEY_ENV: &str = "BEDROCK_MANTLE_API_KEY";
pub const BEDROCK_MANTLE_REGION_ENV: &str = "BEDROCK_MANTLE_REGION";
pub const BEDROCK_MANTLE_DEFAULT_REGION: &str = "us-east-1";

pub const BEDROCK_MANTLE_SETTINGS: MantleSettings = MantleSettings {
    bearer_token_envs: &[BEDROCK_MANTLE_API_KEY_ENV, AWS_BEARER_TOKEN_BEDROCK],
    api_base_env: BEDROCK_MANTLE_API_BASE_ENV,
    region_envs: &[BEDROCK_MANTLE_REGION_ENV, AWS_REGION_NAME, AWS_REGION],
    default_region: BEDROCK_MANTLE_DEFAULT_REGION,
    secret_names: &[
        BEDROCK_MANTLE_API_KEY_ENV,
        AWS_BEARER_TOKEN_BEDROCK,
        BEDROCK_MANTLE_API_BASE_ENV,
        BEDROCK_MANTLE_REGION_ENV,
        AWS_REGION_NAME,
        AWS_REGION,
        AWS_ACCESS_KEY_ID,
        AWS_SECRET_ACCESS_KEY,
        AWS_SESSION_TOKEN,
        AWS_SESSION_NAME,
        AWS_PROFILE_NAME,
        AWS_ROLE_NAME,
        AWS_WEB_IDENTITY_TOKEN,
        AWS_STS_ENDPOINT,
        AWS_EXTERNAL_ID,
    ],
};

pub fn is_mantle_claude_model(model: &str) -> bool {
    model.to_ascii_lowercase().contains("claude")
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::claude("anthropic.claude-test", true)]
    #[case::mixed_case("us-east-1/Anthropic.CLAUDE-test", true)]
    #[case::other_family("openai.gpt-oss-120b", false)]
    fn only_claude_models_use_messages(#[case] model: &str, #[case] expected: bool) {
        assert_eq!(is_mantle_claude_model(model), expected);
    }
}
