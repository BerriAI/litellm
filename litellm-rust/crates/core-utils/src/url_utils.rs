use std::marker::PhantomData;

use percent_encoding::{AsciiSet, utf8_percent_encode};
use url::Url;

#[derive(Debug, thiserror::Error)]
pub enum ApiUrlError {
    #[error("invalid URL: {0}")]
    Parse(#[from] url::ParseError),
    #[error("URL cannot be used as a base")]
    CannotBeBase,
}

pub struct Base;
pub struct Complete;

pub struct ApiUrl<State> {
    url: Url,
    state: PhantomData<State>,
}

impl ApiUrl<Base> {
    pub fn parse(value: &str) -> Result<Self, ApiUrlError> {
        Ok(Self {
            url: Url::parse(value.trim())?,
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
    pub fn append_query_pairs<'a>(
        mut self,
        pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Self {
        self.url.query_pairs_mut().extend_pairs(pairs);
        self
    }

    /// Like [`Self::append_query_pairs`], but percent-encodes keys and values with `escape`
    /// instead of form encoding, so callers choose which reserved characters stay literal.
    /// An existing query is kept; an empty result leaves no dangling `?`.
    pub fn append_query_pairs_escaped<'a>(
        mut self,
        pairs: impl IntoIterator<Item = (&'a str, &'a str)>,
        escape: &'static AsciiSet,
    ) -> Self {
        let appended = pairs.into_iter().map(|(key, value)| {
            format!(
                "{}={}",
                utf8_percent_encode(key, escape),
                utf8_percent_encode(value, escape)
            )
        });
        let query = self
            .url
            .query()
            .filter(|existing| !existing.is_empty())
            .map(str::to_owned)
            .into_iter()
            .chain(appended)
            .collect::<Vec<_>>()
            .join("&");
        self.url
            .set_query(Some(query.as_str()).filter(|query| !query.is_empty()));
        self
    }

    pub fn into_string(self) -> String {
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
    fn completion_places_paths_before_queries() {
        let actual = ApiUrl::parse("https://example.test/v1?tenant=a")
            .and_then(|url| url.complete_path(&["v1", "ocr"]))
            .map(|url| url.into_string())
            .expect("url builds");
        assert_eq!(actual, "https://example.test/v1/ocr?tenant=a");
    }

    #[test]
    fn appended_query_pairs_are_encoded() {
        let actual = ApiUrl::parse("https://example.test")
            .and_then(|url| url.complete_path(&["analyze"]))
            .map(|url| {
                url.append_query_pairs([("model", "name with spaces")])
                    .into_string()
            })
            .expect("url builds");
        assert_eq!(
            actual,
            "https://example.test/analyze?model=name+with+spaces"
        );
    }

    const KEEP_COMMAS: &AsciiSet = &percent_encoding::NON_ALPHANUMERIC.remove(b',');

    #[rstest::rstest]
    #[case::fresh_query("https://example.test", &[("pages", "1,2")], "?pages=1,2")]
    #[case::existing_query_kept(
        "https://example.test?tenant=a",
        &[("pages", "1,2")],
        "?tenant=a&pages=1,2"
    )]
    #[case::bare_question_mark("https://example.test?", &[("pages", "1")], "?pages=1")]
    #[case::no_pairs_no_query("https://example.test", &[], "")]
    #[case::no_pairs_keeps_query("https://example.test?tenant=a", &[], "?tenant=a")]
    #[case::outside_set_escaped(
        "https://example.test",
        &[("q", "a b&c=d+e%2C#")],
        "?q=a%20b%26c%3Dd%2Be%252C%23"
    )]
    #[case::keys_escaped_too("https://example.test", &[("a&b", "1")], "?a%26b=1")]
    #[case::non_ascii_as_utf8("https://example.test", &[("q", "é")], "?q=%C3%A9")]
    fn escaped_query_pairs(
        #[case] base: &str,
        #[case] pairs: &[(&str, &str)],
        #[case] expected_query: &str,
    ) {
        let actual = ApiUrl::parse(base)
            .and_then(|url| url.complete_path(&["analyze"]))
            .map(|url| {
                url.append_query_pairs_escaped(pairs.iter().copied(), KEEP_COMMAS)
                    .into_string()
            })
            .expect("url builds");
        assert_eq!(
            actual,
            format!("https://example.test/analyze{expected_query}")
        );
    }
}
