use crate::{
    Error,
    base_llm::auth::{Headers, ValidatedEnvironment},
    openai::common_utils,
};

const DEFAULT_API_BASE: &str = "https://api.openai.com";
const API_BASE_ENVS: [&str; 2] = ["OPENAI_BASE_URL", "OPENAI_API_BASE"];
const PATH: &str = "/v1/decisions";

pub fn secret_names(api_key: Option<&str>, api_base: Option<&str>) -> &'static [&'static str] {
    common_utils::secret_names(api_key, api_base)
}

pub fn complete_url(api_base: Option<&str>, lookup: &dyn Fn(&str) -> Option<String>) -> String {
    let base = api_base
        .filter(|base| !base.is_empty())
        .map(str::to_owned)
        .or_else(|| API_BASE_ENVS.into_iter().find_map(lookup))
        .unwrap_or_else(|| DEFAULT_API_BASE.into());
    let base = base.trim_end_matches('/');
    format!("{}{PATH}", base.strip_suffix("/v1").unwrap_or(base))
}

pub fn environment(
    headers: Headers,
    api_key: Option<&str>,
    lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<ValidatedEnvironment, Error> {
    common_utils::bearer_environment(headers, api_key, lookup)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn env(name: &str) -> Option<String> {
        (name == "OPENAI_API_BASE").then(|| "http://env-base/v1/".into())
    }

    #[rstest]
    #[case::default(None, no_env, "https://api.openai.com/v1/decisions")]
    #[case::base_with_version(Some("http://mock/d/1/v1"), no_env, "http://mock/d/1/v1/decisions")]
    #[case::base_without_version(Some("http://mock/d/1/"), no_env, "http://mock/d/1/v1/decisions")]
    #[case::environment(None, env, "http://env-base/v1/decisions")]
    #[case::deployment_wins(Some("http://mock"), env, "http://mock/v1/decisions")]
    fn the_url_keeps_one_version_segment(
        #[case] api_base: Option<&str>,
        #[case] lookup: fn(&str) -> Option<String>,
        #[case] expected: &str,
    ) {
        assert_eq!(complete_url(api_base, &lookup), expected);
    }
}
