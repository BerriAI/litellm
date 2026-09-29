use litellm_core_utils::{
    ApiUrlError,
    url_utils::{ApiUrl, Complete, EndpointTarget, Mount},
};

pub const DEFAULT_API_BASE: &str = "https://api.mistral.ai/v1";

crate::provider_endpoints! {
    pub enum MistralEndpoint { Ocr => POST path(litellm_llms_types::providers::mistral::batches::MistralBatchEndpoint::Ocr.path()), }
}

pub fn legacy_target(value: &str) -> Result<EndpointTarget, ApiUrlError> {
    let parsed = ApiUrl::<Complete>::parse_exact(value)?;
    let route = MistralEndpoint::Ocr.path();
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
    if path.ends_with("/v1") {
        let mount = ApiUrl::<Mount>::parse_mount(value)?;
        return Ok(EndpointTarget::Exact(mount.resolve_segments(["ocr"], &[])?));
    }
    Ok(EndpointTarget::Mount(ApiUrl::parse_mount(value)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_llm::endpoint::ProviderEndpoint;
    use rstest::rstest;

    #[rstest]
    #[case::root("https://gateway.test", "/v1/ocr")]
    #[case::version("https://gateway.test/v1?tenant=1", "/v1/ocr")]
    #[case::gateway("https://gateway.test/team", "/team/v1/ocr")]
    #[case::complete("https://gateway.test/team/v1/ocr?tenant=1", "/team/v1/ocr")]
    #[case::complete_with_trailing_slash("https://gateway.test/v1/ocr/?tenant=1", "/v1/ocr")]
    fn legacy_policy(#[case] value: &str, #[case] path: &str) {
        let target = legacy_target(value).unwrap();
        let resolved = MistralEndpoint::Ocr.resolve(&target).unwrap();
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
            EndpointTarget::Mount(ApiUrl::<Mount>::parse_mount("https://gateway.test/v1").unwrap());
        assert_eq!(
            MistralEndpoint::Ocr
                .resolve(&mount)
                .unwrap()
                .url()
                .as_url()
                .path(),
            "/v1/v1/ocr"
        );
        let exact = ApiUrl::parse_exact("https://gateway.test/custom?sig=a%2fb").unwrap();
        assert_eq!(
            MistralEndpoint::Ocr
                .resolve(&EndpointTarget::Exact(exact.clone()))
                .unwrap()
                .url(),
            &exact
        );
    }
}
