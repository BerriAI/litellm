#![allow(
    dead_code,
    reason = "used by the OCR architecture in the next stacked PR"
)]

use std::marker::PhantomData;

use thiserror::Error;
use url::Url;

#[derive(Debug, Error)]
pub(crate) enum ApiUrlError {
    #[error("invalid URL: {0}")]
    Parse(#[from] url::ParseError),
    #[error("URL cannot be used as a base")]
    CannotBeBase,
    #[error("unsupported URL scheme: {0}")]
    Scheme(String),
}

pub(crate) struct Base;
pub(crate) struct Complete;

pub(crate) struct ApiUrl<State> {
    url: Url,
    state: PhantomData<State>,
}

impl ApiUrl<Base> {
    pub(crate) fn parse(value: &str) -> Result<Self, ApiUrlError> {
        Ok(Self {
            url: Url::parse(value.trim())?,
            state: PhantomData,
        })
    }

    pub(crate) fn parse_with_default_scheme(
        value: &str,
        default_scheme: &str,
    ) -> Result<Self, ApiUrlError> {
        let value = value.trim();
        if value.contains("://") {
            return Self::parse(value);
        }
        Self::parse(&format!("{default_scheme}://{value}"))
    }

    pub(crate) fn with_scheme(mut self, scheme: &str) -> Result<Self, ApiUrlError> {
        self.url
            .set_scheme(scheme)
            .map_err(|()| ApiUrlError::Scheme(scheme.to_string()))?;
        Ok(self)
    }

    pub(crate) fn scheme(&self) -> &str {
        self.url.scheme()
    }

    pub(crate) fn truncate_path_after(mut self, segment: &str) -> Result<Self, ApiUrlError> {
        let segments: Vec<String> = self
            .url
            .path_segments()
            .ok_or(ApiUrlError::CannotBeBase)?
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect();
        let Some(index) = segments.iter().position(|value| value == segment) else {
            return Ok(self);
        };
        self.url
            .path_segments_mut()
            .map_err(|()| ApiUrlError::CannotBeBase)?
            .clear()
            .extend(segments[..=index].iter().map(String::as_str));
        Ok(self)
    }

    pub(crate) fn complete_path(self, target: &[&str]) -> Result<ApiUrl<Complete>, ApiUrlError> {
        self.complete_path_with_aliases(target, &[])
    }

    pub(crate) fn complete_path_with_aliases(
        mut self,
        target: &[&str],
        aliases: &[&[&str]],
    ) -> Result<ApiUrl<Complete>, ApiUrlError> {
        let existing: Vec<String> = self
            .url
            .path_segments()
            .ok_or(ApiUrlError::CannotBeBase)?
            .filter(|segment| !segment.is_empty())
            .map(str::to_string)
            .collect();
        let ends_with = |suffix: &[&str]| {
            existing.len() >= suffix.len()
                && existing[existing.len() - suffix.len()..]
                    .iter()
                    .map(String::as_str)
                    .eq(suffix.iter().copied())
        };
        let already_complete = aliases.iter().any(|suffix| ends_with(suffix));
        let overlap = if already_complete {
            target.len()
        } else {
            (0..=existing.len().min(target.len()))
                .rev()
                .find(|&length| {
                    existing[existing.len() - length..]
                        .iter()
                        .map(String::as_str)
                        .eq(target[..length].iter().copied())
                })
                .unwrap_or(0)
        };
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
    pub(crate) fn has_query_key(&self, key: &str) -> bool {
        self.url.query_pairs().any(|(name, _)| name == key)
    }

    pub(crate) fn append_query_pair(mut self, key: &str, value: &str) -> Self {
        self.url.query_pairs_mut().append_pair(key, value);
        self
    }

    pub(crate) fn append_query_pairs<'a>(
        mut self,
        pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Self {
        self.url.query_pairs_mut().extend_pairs(pairs);
        self
    }

    pub(crate) fn into_string(self) -> String {
        self.url.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_appends_only_the_missing_path_suffix() {
        for (base, expected) in [
            ("https://example.test", "https://example.test/v1/ocr"),
            ("https://example.test/v1", "https://example.test/v1/ocr"),
            ("https://example.test/v1/ocr", "https://example.test/v1/ocr"),
        ] {
            let actual = ApiUrl::parse(base)
                .and_then(|url| url.complete_path(&["v1", "ocr"]))
                .map(|url| url.into_string())
                .expect("url builds");
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn completion_places_paths_before_queries_and_encodes_query_values() {
        let actual = ApiUrl::parse("https://example.test/v1?tenant=a")
            .and_then(|url| url.complete_path(&["v1", "ocr"]))
            .map(|url| {
                url.append_query_pair("model", "name with spaces")
                    .into_string()
            })
            .expect("url builds");
        assert_eq!(
            actual,
            "https://example.test/v1/ocr?tenant=a&model=name+with+spaces"
        );
    }
}
