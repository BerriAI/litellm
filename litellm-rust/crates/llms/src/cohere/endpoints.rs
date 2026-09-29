use litellm_core_utils::{
    ApiUrlError,
    url_utils::{ApiUrl, Complete, EndpointTarget, Mount},
};

pub const DEFAULT_API_BASE: &str = "https://api.cohere.com";

crate::provider_endpoints! {
    pub enum CohereEndpoint { Parse => POST "/v2/parse", }
}

pub fn legacy_target(value: &str) -> Result<EndpointTarget, ApiUrlError> {
    let parsed = ApiUrl::<Complete>::parse_exact(value)?;
    let route = CohereEndpoint::Parse.path();
    let path = parsed.as_url().path().trim_end_matches('/');
    if route.is_suffix_of(path) {
        let mut normalized = parsed.into_url();
        let path = normalized
            .path()
            .strip_suffix('/')
            .unwrap_or(normalized.path())
            .to_owned();
        normalized.set_path(&path);
        return Ok(EndpointTarget::Exact(ApiUrl::parse_exact(
            normalized.as_str(),
        )?));
    }
    if path.ends_with("/v2") {
        let mount = ApiUrl::<Mount>::parse_mount(value)?;
        return Ok(EndpointTarget::Exact(
            mount.resolve_segments(["parse"], &[])?,
        ));
    }
    Ok(EndpointTarget::Mount(ApiUrl::parse_mount(value)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_llm::endpoint::ProviderEndpoint;
    use rstest::rstest;

    #[rstest]
    #[case::root("https://gateway.test", "/v2/parse")]
    #[case::version("https://gateway.test/v2?tenant=1", "/v2/parse")]
    #[case::gateway("https://gateway.test/team", "/team/v2/parse")]
    #[case::complete("https://gateway.test/team/v2/parse?tenant=1", "/team/v2/parse")]
    #[case::complete_with_trailing_slash("https://gateway.test/v2/parse/?tenant=1", "/v2/parse")]
    fn legacy_policy(#[case] value: &str, #[case] path: &str) {
        let target = legacy_target(value).unwrap();
        let resolved = CohereEndpoint::Parse.resolve(&target).unwrap();
        assert_eq!(resolved.url().as_url().path(), path);
        assert_eq!(resolved.method(), &reqwest::Method::POST);
        assert_eq!(
            resolved.url().as_url().query(),
            reqwest::Url::parse(value).unwrap().query()
        );
    }

    #[rstest]
    fn explicit_targets_bypass_legacy_interpretation() {
        let mount =
            EndpointTarget::Mount(ApiUrl::<Mount>::parse_mount("https://gateway.test/v2").unwrap());
        assert_eq!(
            CohereEndpoint::Parse
                .resolve(&mount)
                .unwrap()
                .url()
                .as_url()
                .path(),
            "/v2/v2/parse"
        );
        let exact = ApiUrl::parse_exact("https://gateway.test/custom?sig=a%2fb").unwrap();
        assert_eq!(
            CohereEndpoint::Parse
                .resolve(&EndpointTarget::Exact(exact.clone()))
                .unwrap()
                .url(),
            &exact
        );
    }
}
