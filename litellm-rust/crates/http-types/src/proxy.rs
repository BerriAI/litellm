use litellm_core_utils::settings::Lookup;
use veil::Redact;

#[derive(Clone, Redact, Default, PartialEq, Eq, Hash)]
pub struct EnvironmentProxies {
    #[redact]
    all: String,
    #[redact]
    http: String,
    #[redact]
    https: String,
    no: String,
}

impl EnvironmentProxies {
    pub fn from_environment(env: &impl Lookup) -> Self {
        Self::resolve(env, cfg!(windows))
    }

    fn resolve(env: &impl Lookup, names_ignore_case: bool) -> Self {
        let lowercase_first = |upper: Option<&str>, lower: &str| {
            env.get(lower)
                .or_else(|| upper.and_then(|name| env.truthy(name)))
                .unwrap_or_default()
        };
        let is_cgi = env.get("REQUEST_METHOD").is_some();
        Self {
            all: lowercase_first(Some("ALL_PROXY"), "all_proxy"),
            http: if is_cgi && names_ignore_case {
                String::new()
            } else {
                lowercase_first((!is_cgi).then_some("HTTP_PROXY"), "http_proxy")
            },
            https: lowercase_first(Some("HTTPS_PROXY"), "https_proxy"),
            no: lowercase_first(Some("NO_PROXY"), "no_proxy"),
        }
    }

    pub fn all(&self) -> &str {
        &self.all
    }

    pub fn http(&self) -> &str {
        &self.http
    }

    pub fn https(&self) -> &str {
        &self.https
    }

    pub fn no(&self) -> &str {
        &self.no
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
    fn cgi_drops_http_proxy_entirely_where_variable_names_ignore_case() {
        let windows_env = |name: &str| match name.to_ascii_uppercase().as_str() {
            "REQUEST_METHOD" => Some("GET".to_string()),
            "HTTP_PROXY" => Some("http://attacker:3128".to_string()),
            "HTTPS_PROXY" => Some("http://proxy:3128".to_string()),
            _ => None,
        };
        let ignoring_case = EnvironmentProxies::resolve(&windows_env, true);
        assert_eq!(ignoring_case.http(), "");
        assert_eq!(ignoring_case.https(), "http://proxy:3128");
        assert_eq!(
            EnvironmentProxies::resolve(&windows_env, false).http(),
            "http://attacker:3128"
        );
    }

    #[test]
    fn debug_output_hides_proxy_credentials_but_shows_which_variables_are_set() {
        let proxies = EnvironmentProxies::from_environment(&env_of(&[
            ("HTTPS_PROXY", "http://operator:hunter2@proxy.corp:3128"),
            ("NO_PROXY", "internal.test"),
        ]));
        let debug = format!("{proxies:?}");
        assert!(!debug.contains("hunter2") && !debug.contains("operator"));
        assert!(debug.contains("internal.test"));
        assert_ne!(debug, format!("{:?}", EnvironmentProxies::default()));
    }

    #[test]
    fn an_empty_environment_proxies_nothing() {
        assert_eq!(
            EnvironmentProxies::from_environment(&env_of(&[])),
            EnvironmentProxies::default()
        );
    }
}
