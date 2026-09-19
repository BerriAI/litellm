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
        if env.get("REQUEST_METHOD").is_some() {
            return Self::default();
        }
        let first = |upper: &str, lower: &str| {
            env.get(upper)
                .or_else(|| env.get(lower))
                .unwrap_or_default()
        };
        Self {
            all: first("ALL_PROXY", "all_proxy"),
            http: first("HTTP_PROXY", "http_proxy"),
            https: first("HTTPS_PROXY", "https_proxy"),
            no: first("NO_PROXY", "no_proxy"),
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
    #[case::cgi_ignores_everything(&[("HTTPS_PROXY", "http://proxy:3128"), ("REQUEST_METHOD", "GET")], "https://api.test/", false)]
    #[case::uppercase_wins_even_when_empty(&[("HTTPS_PROXY", ""), ("https_proxy", "http://proxy:3128")], "https://api.test/", false)]
    fn proxies_follow_the_injected_environment(
        #[case] env: &'static [(&'static str, &'static str)],
        #[case] target: &str,
        #[case] expected: bool,
    ) {
        let proxies = EnvironmentProxies::from_environment(&env_of(env));
        assert_eq!(proxies.apply_to(&url(target)), expected);
    }

    #[test]
    fn an_empty_environment_proxies_nothing() {
        let proxies = EnvironmentProxies::from_environment(&env_of(&[]));
        assert_eq!(proxies, EnvironmentProxies::default());
        assert!(proxies.reqwest_proxies().is_empty());
    }
}
