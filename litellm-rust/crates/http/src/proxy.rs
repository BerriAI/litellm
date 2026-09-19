use hyper_util::client::proxy::matcher::Matcher;
use litellm_core_utils::settings::Lookup;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EnvironmentProxies {
    all: String,
    http: String,
    https: String,
    no: String,
}

impl EnvironmentProxies {
    pub fn from_environment(env: &impl Lookup) -> Self {
        let lowercase_first = |upper: Option<&str>, lower: &str| {
            env.get(lower)
                .or_else(|| upper.and_then(|name| env.truthy(name)))
                .unwrap_or_default()
        };
        let is_cgi = env.get("REQUEST_METHOD").is_some();
        Self {
            all: lowercase_first(Some("ALL_PROXY"), "all_proxy"),
            http: lowercase_first((!is_cgi).then_some("HTTP_PROXY"), "http_proxy"),
            https: lowercase_first(Some("HTTPS_PROXY"), "https_proxy"),
            no: lowercase_first(Some("NO_PROXY"), "no_proxy"),
        }
    }

    pub fn apply_to(&self, url: &reqwest::Url) -> bool {
        let matcher = Matcher::builder()
            .all(self.all.clone())
            .http(self.http.clone())
            .https(self.https.clone())
            .no(self.no.clone())
            .build();
        url.as_str()
            .parse::<http::Uri>()
            .is_ok_and(|uri| matcher.intercept(&uri).is_some())
    }

    pub(crate) fn reqwest_proxies(&self) -> Vec<reqwest::Proxy> {
        let no_proxy = reqwest::NoProxy::from_string(&self.no);
        [
            reqwest::Proxy::http(self.http.as_str()),
            reqwest::Proxy::https(self.https.as_str()),
            reqwest::Proxy::all(self.all.as_str()),
        ]
        .into_iter()
        .filter_map(Result::ok)
        .map(|proxy| proxy.no_proxy(no_proxy.clone()))
        .collect()
    }
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
    fn proxies_follow_the_injected_environment(
        #[case] env: &'static [(&'static str, &'static str)],
        #[case] target: &str,
        #[case] expected: bool,
    ) {
        let proxies = EnvironmentProxies::from_environment(&env_of(env));
        assert_eq!(proxies.apply_to(&url(target)), expected);
    }

    #[rstest]
    #[case::lowercase_wins(&[("HTTPS_PROXY", "http://upper:3128"), ("https_proxy", "http://lower:3128")], &[("https_proxy", "http://lower:3128")])]
    #[case::empty_uppercase_falls_through_to_lowercase(&[("HTTPS_PROXY", ""), ("https_proxy", "http://lower:3128")], &[("https_proxy", "http://lower:3128")])]
    #[case::empty_uppercase_alone_is_unset(&[("HTTPS_PROXY", "")], &[])]
    #[case::empty_lowercase_clears_the_uppercase_value(&[("HTTPS_PROXY", "http://upper:3128"), ("https_proxy", "")], &[])]
    #[case::lowercase_no_proxy_wins(&[("NO_PROXY", "upper.test"), ("no_proxy", "lower.test")], &[("no_proxy", "lower.test")])]
    #[case::cgi_forgets_the_client_settable_http_proxy(&[("REQUEST_METHOD", "GET"), ("HTTP_PROXY", "http://attacker:3128")], &[])]
    #[case::cgi_keeps_lowercase_http_proxy(&[("REQUEST_METHOD", "GET"), ("HTTP_PROXY", "http://attacker:3128"), ("http_proxy", "http://lower:3128")], &[("http_proxy", "http://lower:3128")])]
    #[case::cgi_keeps_every_other_variable(&[("REQUEST_METHOD", "GET"), ("HTTPS_PROXY", "http://proxy:3128"), ("ALL_PROXY", "http://all:3128"), ("NO_PROXY", "internal.test")], &[("HTTPS_PROXY", "http://proxy:3128"), ("ALL_PROXY", "http://all:3128"), ("NO_PROXY", "internal.test")])]
    fn variables_resolve_like_urllib_getproxies_environment(
        #[case] env: &'static [(&'static str, &'static str)],
        #[case] equivalent: &'static [(&'static str, &'static str)],
    ) {
        assert_eq!(
            EnvironmentProxies::from_environment(&env_of(env)),
            EnvironmentProxies::from_environment(&env_of(equivalent))
        );
    }

    #[test]
    fn a_cgi_request_still_proxies_https_through_the_configured_proxy() {
        let proxies = EnvironmentProxies::from_environment(&env_of(&[
            ("REQUEST_METHOD", "GET"),
            ("HTTPS_PROXY", "http://proxy:3128"),
        ]));
        assert!(proxies.apply_to(&url("https://api.test/")));
    }

    #[test]
    fn an_empty_environment_proxies_nothing() {
        let proxies = EnvironmentProxies::from_environment(&env_of(&[]));
        assert_eq!(proxies, EnvironmentProxies::default());
        assert!(proxies.reqwest_proxies().is_empty());
    }
}
