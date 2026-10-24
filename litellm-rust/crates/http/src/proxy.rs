use hyper_util::client::proxy::matcher::Matcher;
use litellm_http_types::EnvironmentProxies;

pub(crate) fn matcher(
    proxies: &EnvironmentProxies,
) -> impl Fn(&reqwest::Url) -> bool + Send + Sync + use<> {
    let matcher = Matcher::builder()
        .all(proxies.all())
        .http(proxies.http())
        .https(proxies.https())
        .no(proxies.no())
        .build();
    move |url| {
        url.as_str()
            .parse::<http::Uri>()
            .is_ok_and(|uri| matcher.intercept(&uri).is_some())
    }
}

pub(crate) fn reqwest_proxies(proxies: &EnvironmentProxies) -> Vec<reqwest::Proxy> {
    let no_proxy = reqwest::NoProxy::from_string(proxies.no());
    [
        reqwest::Proxy::http(proxies.http()),
        reqwest::Proxy::https(proxies.https()),
        reqwest::Proxy::all(proxies.all()),
    ]
    .into_iter()
    .filter_map(Result::ok)
    .map(|proxy| proxy.no_proxy(no_proxy.clone()))
    .collect()
}

#[cfg(feature = "mcp")]
pub(crate) fn mcp_proxies(proxies: &EnvironmentProxies) -> Vec<reqwest_mcp::Proxy> {
    let no_proxy = reqwest_mcp::NoProxy::from_string(proxies.no());
    [
        reqwest_mcp::Proxy::http(proxies.http()),
        reqwest_mcp::Proxy::https(proxies.https()),
        reqwest_mcp::Proxy::all(proxies.all()),
    ]
    .into_iter()
    .filter_map(Result::ok)
    .map(|proxy| proxy.no_proxy(no_proxy.clone()))
    .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn env_of(values: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            values
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    fn url(value: &str) -> reqwest::Url {
        reqwest::Url::parse(value).unwrap()
    }

    #[rstest]
    #[case::http_only(&[("HTTP_PROXY", "http://proxy:3128")], "http://api.test/", true)]
    #[case::http_proxy_skips_https(&[("HTTP_PROXY", "http://proxy:3128")], "https://api.test/", false)]
    #[case::all_covers_https(&[("ALL_PROXY", "http://proxy:3128")], "https://api.test/", true)]
    #[case::lowercase(&[("https_proxy", "http://proxy:3128")], "https://api.test/", true)]
    #[case::no_proxy_bypass(&[("HTTPS_PROXY", "http://proxy:3128"), ("NO_PROXY", "api.test")], "https://api.test/", false)]
    #[case::cgi_still_proxies_https(&[("REQUEST_METHOD", "GET"), ("HTTPS_PROXY", "http://proxy:3128")], "https://api.test/", true)]
    fn proxies_follow_the_injected_environment(
        #[case] env: &'static [(&'static str, &'static str)],
        #[case] target: &str,
        #[case] expected: bool,
    ) {
        let proxies = EnvironmentProxies::from_environment(&env_of(env));
        assert_eq!(matcher(&proxies)(&url(target)), expected);
    }

    #[test]
    fn an_empty_environment_proxies_nothing() {
        assert!(reqwest_proxies(&EnvironmentProxies::default()).is_empty());
    }
}
