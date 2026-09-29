use std::marker::PhantomData;

use url::Url;

use crate::ApiUrlError;
use litellm_llms_types::endpoint::EndpointPath;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mount;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Base;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Complete;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiUrl<State> {
    url: Url,
    state: PhantomData<State>,
}

impl ApiUrl<Base> {
    pub fn parse(value: &str) -> Result<Self, ApiUrlError> {
        Ok(Self {
            url: parse_http(value)?,
            state: PhantomData,
        })
    }

    pub fn complete_path(mut self, target: &[&str]) -> Result<ApiUrl<Complete>, ApiUrlError> {
        let existing: Vec<String> = self
            .url
            .path_segments()
            .ok_or(ApiUrlError::CannotBeBase)?
            .filter(|segment| !segment.is_empty())
            .map(str::to_string)
            .collect();
        let overlap = (0..=existing.len().min(target.len()))
            .rev()
            .find(|&length| {
                existing[existing.len() - length..]
                    .iter()
                    .map(String::as_str)
                    .eq(target[..length].iter().copied())
            })
            .unwrap_or(0);
        self.url
            .path_segments_mut()
            .map_err(|()| ApiUrlError::CannotBeBase)?
            .pop_if_empty()
            .extend(target[overlap..].iter().copied());
        Ok(ApiUrl {
            url: self.url,
            state: PhantomData,
        })
    }
}

impl ApiUrl<Complete> {
    pub fn parse_exact(value: &str) -> Result<Self, ApiUrlError> {
        Ok(Self {
            url: parse_http(value)?,
            state: PhantomData,
        })
    }

    pub fn as_url(&self) -> &Url {
        &self.url
    }

    pub fn into_url(self) -> Url {
        self.url
    }

    pub fn into_string(self) -> String {
        self.url.into()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EndpointTarget {
    Mount(ApiUrl<Mount>),
    Exact(ApiUrl<Complete>),
}

#[derive(Clone, Copy, Debug)]
pub enum QueryPolicy {
    Default,
    Replace,
    Reject,
}

#[derive(Clone, Copy, Debug)]
pub struct QueryParameter<'a> {
    pub key: &'a str,
    pub value: &'a str,
    pub policy: QueryPolicy,
}

fn parse_http(value: &str) -> Result<Url, ApiUrlError> {
    let url = Url::parse(value.trim())?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || url.cannot_be_a_base()
    {
        return Err(ApiUrlError::InvalidTransport);
    }
    Ok(url)
}

pub fn check_segment(value: &str) -> Result<&str, ApiUrlError> {
    if value.is_empty() || matches!(value, "." | "..") {
        return Err(ApiUrlError::InvalidSegment);
    }
    Ok(value)
}

impl ApiUrl<Mount> {
    pub fn parse_mount(value: &str) -> Result<Self, ApiUrlError> {
        Ok(Self {
            url: parse_http(value)?,
            state: PhantomData,
        })
    }

    pub fn resolve(&self, path: EndpointPath) -> Result<ApiUrl<Complete>, ApiUrlError> {
        self.resolve_segments(path.segments(), &[])
    }

    pub fn resolve_segments<'a>(
        &self,
        segments: impl IntoIterator<Item = &'a str>,
        query: &[QueryParameter<'_>],
    ) -> Result<ApiUrl<Complete>, ApiUrlError> {
        Ok(ApiUrl {
            url: resolve_url(&self.url, segments, query)?,
            state: PhantomData,
        })
    }
}

fn resolve_url<'a>(
    mount: &Url,
    segments: impl IntoIterator<Item = &'a str>,
    query: &[QueryParameter<'_>],
) -> Result<Url, ApiUrlError> {
    let checked = segments
        .into_iter()
        .map(check_segment)
        .collect::<Result<Vec<_>, _>>()?;
    let mut url = mount.clone();
    url.path_segments_mut()
        .map_err(|()| ApiUrlError::CannotBeBase)?
        .pop_if_empty()
        .extend(checked);
    for parameter in query {
        let exists = url.query_pairs().any(|(key, _)| key == parameter.key);
        match (parameter.policy, exists) {
            (QueryPolicy::Default, true) => continue,
            (QueryPolicy::Reject, true) => {
                return Err(ApiUrlError::QueryConflict(parameter.key.into()));
            }
            (QueryPolicy::Replace, true) => {
                let retained = url
                    .query_pairs()
                    .filter(|(key, _)| key != parameter.key)
                    .map(|(key, value)| (key.into_owned(), value.into_owned()))
                    .collect::<Vec<_>>();
                url.set_query(None);
                url.query_pairs_mut().extend_pairs(retained);
            }
            _ => {}
        }
        url.query_pairs_mut()
            .append_pair(parameter.key, parameter.value);
    }
    Ok(url)
}

/// ```
/// use litellm_core_utils::url_utils::{WebSocketUrl, Mount};
/// let mount = WebSocketUrl::<Mount>::parse_mount("wss://gateway.test/team").unwrap();
/// let complete = mount.resolve_segments(["responses"], &[]).unwrap();
/// assert_eq!(complete.as_url().path(), "/team/responses");
/// ```
/// ```compile_fail
/// use litellm_core_utils::url_utils::{WebSocketUrl, Complete};
/// let complete = WebSocketUrl::<Complete>::parse_exact("wss://gateway.test/custom").unwrap();
/// complete.resolve_segments(["responses"], &[]);
/// ```
/// ```compile_fail
/// use litellm_core_utils::url_utils::{WebSocketUrl, Complete};
/// let mut complete = WebSocketUrl::<Complete>::parse_exact("wss://gateway.test/custom").unwrap();
/// complete.as_url().set_scheme("https");
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WebSocketUrl<State> {
    url: Url,
    state: PhantomData<State>,
}

fn parse_websocket(value: &str) -> Result<Url, ApiUrlError> {
    let url = Url::parse(value.trim())?;
    if !matches!(url.scheme(), "ws" | "wss") || url.host_str().is_none() || url.cannot_be_a_base() {
        return Err(ApiUrlError::InvalidTransport);
    }
    Ok(url)
}

impl WebSocketUrl<Mount> {
    pub fn parse_mount(value: &str) -> Result<Self, ApiUrlError> {
        Ok(Self {
            url: parse_websocket(value)?,
            state: PhantomData,
        })
    }

    pub fn resolve_segments<'a>(
        &self,
        segments: impl IntoIterator<Item = &'a str>,
        query: &[QueryParameter<'_>],
    ) -> Result<WebSocketUrl<Complete>, ApiUrlError> {
        Ok(WebSocketUrl {
            url: resolve_url(&self.url, segments, query)?,
            state: PhantomData,
        })
    }
}

impl WebSocketUrl<Complete> {
    pub fn parse_exact(value: &str) -> Result<Self, ApiUrlError> {
        Ok(Self {
            url: parse_websocket(value)?,
            state: PhantomData,
        })
    }
    pub fn as_url(&self) -> &Url {
        &self.url
    }
    pub fn into_url(self) -> Url {
        self.url
    }
}

impl EndpointTarget {
    pub fn resolve(&self, path: EndpointPath) -> Result<ApiUrl<Complete>, ApiUrlError> {
        match self {
            Self::Mount(mount) => mount.resolve(path),
            Self::Exact(url) => Ok(url.clone()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::root("https://gateway.test", "https://gateway.test/v1/ocr")]
    #[case::version(
        "https://gateway.test/v1?tenant=a",
        "https://gateway.test/v1/ocr?tenant=a"
    )]
    #[case::complete("https://gateway.test/v1/ocr", "https://gateway.test/v1/ocr")]
    fn unmigrated_adapters_keep_legacy_completion(#[case] value: &str, #[case] expected: &str) {
        let complete = ApiUrl::<Base>::parse(value)
            .unwrap()
            .complete_path(&["v1", "ocr"])
            .unwrap();
        assert_eq!(complete.as_url().as_str(), expected);
    }

    #[rstest]
    #[case::gateway("https://gateway.test/team", "/team/v2/parse")]
    #[case::explicit_version_mount("https://gateway.test/v2", "/v2/v2/parse")]
    #[case::encoded_prefix("https://gateway.test/a%2Fb%20c", "/a%2Fb%20c/v2/parse")]
    fn appends_entire_route_without_changing_mount(#[case] value: &str, #[case] expected: &str) {
        let mount = ApiUrl::<Mount>::parse_mount(value).unwrap();
        let parse = mount.resolve(EndpointPath::new("/v2/parse")).unwrap();
        let other = mount.resolve(EndpointPath::new("/other")).unwrap();
        assert_eq!(parse.as_url().path(), expected);
        assert!(!other.as_url().path().contains("parse"));
        assert_eq!(
            mount.resolve(EndpointPath::new("/v2/parse")).unwrap(),
            parse
        );
    }

    #[rstest]
    #[case::slash("a/b", "a%2Fb")]
    #[case::percent("a%2Fb", "a%252Fb")]
    #[case::space("a b", "a%20b")]
    #[case::unicode("é", "%C3%A9")]
    #[case::backslash("a\\b", "a%5Cb")]
    #[case::structure("a?b#c", "a%3Fb%23c")]
    fn dynamic_identifiers_are_one_encoded_segment(#[case] id: &str, #[case] encoded: &str) {
        let url = ApiUrl::<Mount>::parse_mount("https://gateway.test/p%2Fq")
            .unwrap()
            .resolve_segments([id], &[])
            .unwrap();
        assert_eq!(url.as_url().path(), format!("/p%2Fq/{encoded}"));
        assert_eq!(url.as_url().query(), None);
        assert_eq!(url.as_url().fragment(), None);
    }

    #[rstest]
    #[case::dot(".")]
    #[case::parent("..")]
    #[case::empty("")]
    fn rejects_invalid_dynamic_segments(#[case] id: &str) {
        let mount = ApiUrl::<Mount>::parse_mount("https://gateway.test").unwrap();
        assert_eq!(
            mount.resolve_segments([id], &[]),
            Err(ApiUrlError::InvalidSegment)
        );
    }

    #[rstest]
    #[case::default(QueryPolicy::Default, vec!["old", "older"])]
    #[case::replace(QueryPolicy::Replace, vec!["new"])]
    fn query_policy_preserves_unrelated_repeated_pairs(
        #[case] policy: QueryPolicy,
        #[case] expected: Vec<&str>,
    ) {
        let mount =
            ApiUrl::<Mount>::parse_mount("https://gateway.test?key=old&keep=1&key=older&keep=2")
                .unwrap();
        let url = mount
            .resolve_segments(
                ["route"],
                &[QueryParameter {
                    key: "key",
                    value: "new",
                    policy,
                }],
            )
            .unwrap();
        let values = |key: &str| {
            url.as_url()
                .query_pairs()
                .filter(|(name, _)| name == key)
                .map(|(_, value)| value.into_owned())
                .collect::<Vec<_>>()
        };
        assert_eq!(values("key"), expected);
        assert_eq!(values("keep"), ["1", "2"]);
    }

    #[rstest]
    fn query_conflicts_are_errors() {
        let mount = ApiUrl::<Mount>::parse_mount("https://gateway.test?key=old").unwrap();
        assert_eq!(
            mount.resolve_segments(
                ["route"],
                &[QueryParameter {
                    key: "key",
                    value: "new",
                    policy: QueryPolicy::Reject
                }]
            ),
            Err(ApiUrlError::QueryConflict("key".into()))
        );
    }

    #[rstest]
    fn exact_target_preserves_raw_query() {
        let url =
            ApiUrl::parse_exact("https://gateway.test/custom?sig=a%2fb+%20&key=1&key=2").unwrap();
        let actual = EndpointTarget::Exact(url.clone())
            .resolve(EndpointPath::new("/v2/parse"))
            .unwrap();
        assert_eq!(actual, url);
        assert_eq!(actual.as_url().query(), Some("sig=a%2fb+%20&key=1&key=2"));
    }

    #[rstest]
    #[case::ftp("ftp://gateway.test")]
    #[case::opaque("mailto:somebody@example.test")]
    #[case::relative("/route")]
    fn rejects_non_http_destinations(#[case] value: &str) {
        assert!(ApiUrl::<Mount>::parse_mount(value).is_err());
        assert!(ApiUrl::<Complete>::parse_exact(value).is_err());
    }
}
