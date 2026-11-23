//! How Python's `requests` reads its proxy and CA bundle from the process environment on every
//! request, for token clients that stand in for a Python library sending through `requests`.

use std::{
    collections::BTreeMap,
    net::Ipv4Addr,
    path::{Path, PathBuf},
};

use litellm_env::{Environment, variable};
use url::Url;

use crate::proxy::EnvironmentProxies;

const CA_BUNDLE_VARIABLES: [&str; 2] = ["REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"];

/// What `requests` raises when the CA bundle its environment names does not exist.
#[derive(Debug, thiserror::Error)]
#[error("Could not find a suitable TLS CA certificate bundle, invalid path: {0}")]
pub struct MissingCaBundle(pub String);

/// `requests`' `merge_environment_settings` and `HTTPAdapter.cert_verify`: the first non-empty CA
/// bundle variable, or `None` for `requests`' own default bundle. `SSL_CERT_FILE` is never read.
pub fn ca_bundle(environment: &Environment) -> Result<Option<PathBuf>, MissingCaBundle> {
    let bundle = CA_BUNDLE_VARIABLES
        .iter()
        .find_map(|name| variable(environment, name).filter(|value| !value.is_empty()));
    match bundle {
        None => Ok(None),
        Some(path) if Path::new(path).exists() => Ok(Some(path.into())),
        Some(path) => Err(MissingCaBundle(path.to_owned())),
    }
}

/// The proxies a client sends `url` through when `requests` picked `proxy` for it.
pub fn proxies_for(url: &Url, proxy: Option<String>) -> EnvironmentProxies {
    let variable = format!("{}_proxy", url.scheme());
    EnvironmentProxies::from_environment(&|name: &str| {
        (name == variable).then(|| proxy.clone()).flatten()
    })
}

/// The proxy `requests` sends `url` through: urllib's environment proxies for its scheme, or `all`,
/// unless `requests`' `should_bypass_proxies` skips them for the host.
pub fn requests_proxy(environment: &Environment, url: &Url) -> Option<String> {
    if bypasses_proxies(environment, url) {
        return None;
    }
    let proxies = environment_proxies(environment);
    proxies
        .get(url.scheme())
        .or_else(|| proxies.get("all"))
        .cloned()
}

/// urllib's `getproxies_environment`: every `*_proxy` variable in any case, then the lowercase
/// spellings again, where an empty one removes the scheme. A CGI request drops `HTTP_PROXY`.
fn environment_proxies(environment: &Environment) -> BTreeMap<String, String> {
    let any_case: BTreeMap<String, String> = environment
        .iter()
        .filter(|(_, value)| !value.is_empty())
        .filter_map(|(name, value)| {
            let scheme = name.to_lowercase().strip_suffix("_proxy")?.to_owned();
            Some((scheme, value.clone()))
        })
        .filter(|(scheme, _)| scheme != "http" || variable(environment, "REQUEST_METHOD").is_none())
        .collect();
    environment
        .iter()
        .filter_map(|(name, value)| Some((name.strip_suffix("_proxy")?.to_lowercase(), value)))
        .fold(any_case, |proxies, (scheme, value)| {
            proxies
                .into_iter()
                .filter(|(kept, _)| *kept != scheme)
                .chain((!value.is_empty()).then(|| (scheme.clone(), value.clone())))
                .collect()
        })
}

fn bypasses_proxies(environment: &Environment, url: &Url) -> bool {
    let Some(hostname) = url.host_str().map(|host| {
        host.trim_start_matches('[')
            .trim_end_matches(']')
            .to_lowercase()
    }) else {
        return true;
    };
    let no_proxy = variable(environment, "no_proxy")
        .filter(|value| !value.is_empty())
        .or_else(|| variable(environment, "NO_PROXY"))
        .unwrap_or_default()
        .replace(' ', "");
    let entries = || no_proxy.split(',').filter(|entry| !entry.is_empty());
    let listed = match hostname.parse::<Ipv4Addr>() {
        Ok(address) => entries().any(|entry| match cidr(entry) {
            Some((network, mask)) => address.to_bits() & mask == network.to_bits() & mask,
            None => entry == hostname,
        }),
        Err(_) => {
            let with_port = url
                .port()
                .map_or_else(|| hostname.clone(), |port| format!("{hostname}:{port}"));
            entries()
                .map(|entry| entry.trim_start_matches('.'))
                .any(|entry| {
                    let dotted = format!(".{entry}");
                    hostname == entry
                        || with_port == entry
                        || hostname.ends_with(&dotted)
                        || with_port.ends_with(&dotted)
                })
        }
    };
    listed || bypasses_environment(&environment_proxies(environment), &hostname)
}

/// `requests`' `is_valid_cidr` and `dotted_netmask`.
fn cidr(entry: &str) -> Option<(Ipv4Addr, u32)> {
    let (address, bits) = entry.split_once('/')?;
    let bits = bits
        .parse::<u32>()
        .ok()
        .filter(|bits| (1..=32).contains(bits))?;
    Some((address.parse().ok()?, u32::MAX << (32 - bits)))
}

/// urllib's `proxy_bypass_environment`, over the `no` entry of its environment proxies.
fn bypasses_environment(proxies: &BTreeMap<String, String>, host: &str) -> bool {
    let Some(no_proxy) = proxies.get("no") else {
        return false;
    };
    if no_proxy == "*" {
        return true;
    }
    let host_only = split_port(host);
    no_proxy
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| name.trim_start_matches('.').to_lowercase())
        .any(|name| {
            let dotted = format!(".{name}");
            host_only == name
                || host == name
                || host_only.ends_with(&dotted)
                || host.ends_with(&dotted)
        })
}

/// urllib's `_splitport`, which reads the text after the last colon as a port when it is all digits.
fn split_port(host: &str) -> &str {
    host.rsplit_once(':')
        .filter(|(_, port)| port.bytes().all(|byte| byte.is_ascii_digit()))
        .map_or(host, |(host_only, _)| host_only)
}
