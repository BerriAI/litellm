use std::marker::PhantomData;

use thiserror::Error;
use url::Url;

#[derive(Debug, Error)]
pub(crate) enum ApiUrlError {
    #[error("invalid URL: {0}")]
    Parse(#[from] url::ParseError),
    #[error("URL cannot be used as a base")]
    CannotBeBase,
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
}
