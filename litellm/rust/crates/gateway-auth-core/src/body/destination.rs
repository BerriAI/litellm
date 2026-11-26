//! Whether a URL-valued fallback target only sends to hosts the admin allowed, as
//! `provider_url_destination_candidates` and `is_url_destination_allowed_by_host` in
//! `litellm_core_utils/url_utils.py` decide it. URLs are split the way Python's `urlsplit` splits
//! them, so the host checked is the host the provider call would reach.

/// Whether every HTTP destination `value` could send to is a host `allowed_hosts` lists.
pub(crate) fn url_destination_allowed(value: &str, allowed_hosts: &[String]) -> bool {
    destination_candidates(value)
        .filter(|candidate| is_http(candidate))
        .all(|candidate| allowed_by_host(candidate, allowed_hosts))
}

/// `provider_url_destination_candidates`: each comma-separated part, and what follows its first
/// `/`, since a provider prefix such as `openai/` comes off before the call. Control characters
/// come off with whitespace, so nothing a provider would strip hides a URL here.
fn destination_candidates(value: &str) -> impl Iterator<Item = &str> {
    value
        .split(',')
        .map(|part| part.trim_matches(|c: char| c.is_whitespace() || c.is_control()))
        .flat_map(|part| [part, part.split_once('/').map_or("", |(_, rest)| rest)])
        .filter(|candidate| !candidate.is_empty())
}

fn is_http(candidate: &str) -> bool {
    let lower = candidate.to_ascii_lowercase();
    lower.starts_with("http://") || lower.starts_with("https://")
}

/// `is_url_destination_allowed_by_host`.
fn allowed_by_host(url: &str, allowed_hosts: &[String]) -> bool {
    let Some(Origin {
        scheme: Some(scheme),
        host,
        port,
    }) = Origin::parse(url)
    else {
        return false;
    };
    let effective_port = port
        .filter(|port| *port != 0)
        .unwrap_or(default_port(&scheme));
    allowed_hosts
        .iter()
        .filter_map(|entry| allowlist_entry(entry))
        .any(|entry| {
            entry.host == host
                && entry
                    .scheme
                    .as_ref()
                    .is_none_or(|allowed| *allowed == scheme)
                && entry.port.is_none_or(|allowed| allowed == effective_port)
        })
}

/// `_parse_url_destination_allowlist_entry`: a bare host, a host and port, or an origin. A path
/// is ignored, so an admin can paste an `api_base`.
fn allowlist_entry(entry: &str) -> Option<Origin> {
    let entry = entry.trim();
    if entry.is_empty() {
        return None;
    }
    let parsed = if entry.contains("://") {
        Origin::parse(entry).filter(|origin| origin.scheme.is_some())?
    } else {
        Origin::parse(&format!("//{entry}"))?
    };
    let port = match (&parsed.scheme, parsed.port) {
        (Some(scheme), None) => Some(default_port(scheme)),
        (_, port) => port,
    };
    Some(Origin { port, ..parsed })
}

fn default_port(scheme: &str) -> u16 {
    if scheme == "https" { 443 } else { 80 }
}

/// The parts of a URL the check compares.
#[derive(Debug, PartialEq, Eq)]
struct Origin {
    /// `http` or `https`, or `None` for a URL written without a scheme.
    scheme: Option<String>,
    /// Lower case, without a trailing dot.
    host: String,
    port: Option<u16>,
}

impl Origin {
    /// `urlsplit` with its `hostname` and `port`. `None` for anything the Python check refuses:
    /// another scheme, a user or password, no host, a port that is not a number up to 65535, or
    /// a bracketed host that does not close.
    fn parse(url: &str) -> Option<Self> {
        let url: String = url
            .chars()
            .filter(|c| !matches!(c, '\t' | '\r' | '\n'))
            .collect();
        let (scheme, rest) = match split_scheme(&url) {
            Some((scheme, rest)) => (Some(scheme), rest),
            None => (None, url.as_str()),
        };
        if scheme
            .as_deref()
            .is_some_and(|scheme| scheme != "http" && scheme != "https")
        {
            return None;
        }
        let netloc = rest.strip_prefix("//")?;
        let netloc = netloc.split(['/', '?', '#']).next().unwrap_or_default();
        if netloc.contains('@') {
            return None;
        }
        let (host, port) = match netloc.strip_prefix('[') {
            Some(bracketed) => {
                let (host, after) = bracketed.split_once(']')?;
                (host, after.strip_prefix(':'))
            }
            None => match netloc.split_once(':') {
                Some((host, port)) => (host, Some(port)),
                None => (netloc, None),
            },
        };
        let port = match port.filter(|port| !port.is_empty()) {
            None => None,
            Some(port) if port.bytes().all(|byte| byte.is_ascii_digit()) => {
                Some(port.parse::<u16>().ok()?)
            }
            Some(_) => return None,
        };
        let host = host.to_lowercase();
        let host = host.trim_end_matches('.');
        if host.is_empty() {
            return None;
        }
        Some(Self {
            scheme,
            host: host.to_owned(),
            port,
        })
    }
}

/// The scheme `urlsplit` reads: letters, digits, `+`, `-` and `.` before the first `:`, starting
/// with a letter, in lower case.
fn split_scheme(url: &str) -> Option<(String, &str)> {
    let (scheme, rest) = url.split_once(':')?;
    let valid = scheme
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic())
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    valid.then(|| (scheme.to_ascii_lowercase(), rest))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn hosts(entries: &[&str]) -> Vec<String> {
        entries.iter().map(|entry| (*entry).to_owned()).collect()
    }

    #[rstest]
    #[case::a_model_name("gpt-4o", &[], true)]
    #[case::a_provider_prefixed_model("openai/gpt-4o", &[], true)]
    #[case::an_unlisted_url("https://attacker.example/v1", &[], false)]
    #[case::behind_a_provider_prefix("openai/https://attacker.example/v1", &[], false)]
    #[case::in_a_list("gpt-4o, https://attacker.example", &[], false)]
    #[case::behind_a_control_character("\u{1c}https://attacker.example", &[], false)]
    #[case::a_listed_host("https://api.own.example/v1", &["api.own.example"], true)]
    #[case::listed_in_another_case("HTTPS://API.Own.Example./v1", &["api.own.example"], true)]
    #[case::a_listed_origin_and_its_default_port("https://api.own.example:443/v1", &["https://api.own.example/v1"], true)]
    #[case::another_scheme_than_listed("http://api.own.example", &["https://api.own.example"], false)]
    #[case::another_port_than_listed("https://api.own.example:8443", &["api.own.example:443"], false)]
    #[case::the_listed_port("https://api.own.example:8443", &["api.own.example:8443"], true)]
    #[case::port_zero_is_the_default("https://api.own.example:0", &["https://api.own.example"], true)]
    #[case::a_user_before_a_listed_host("https://attacker.example@api.own.example", &["api.own.example"], false)]
    #[case::a_port_that_is_no_number("https://api.own.example:x", &["api.own.example"], false)]
    #[case::a_port_too_large("https://api.own.example:70000", &["api.own.example"], false)]
    #[case::a_listed_ipv6_host("http://[::1]:8000/v1", &["[::1]:8000"], true)]
    #[case::an_unclosed_bracket("http://[::1/v1", &["[::1]"], false)]
    #[case::a_listed_host_with_a_path_only_listed("https://other.example/api.own.example", &["api.own.example"], false)]
    #[case::an_entry_with_a_user("https://api.own.example", &["user@api.own.example"], false)]
    #[case::an_entry_with_another_scheme("https://api.own.example", &["ftp://api.own.example"], false)]
    fn a_url_valued_target_may_only_reach_a_listed_host(
        #[case] value: &str,
        #[case] allowed: &[&str],
        #[case] expected: bool,
    ) {
        assert_eq!(url_destination_allowed(value, &hosts(allowed)), expected);
    }
}
