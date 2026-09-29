use std::marker::PhantomData;

use url::Url;

use crate::ApiUrlError;

pub struct Base;
pub struct Complete;

pub struct ApiUrl<State> {
    url: Url,
    state: PhantomData<State>,
}

impl ApiUrl<Base> {
    pub fn parse(value: &str) -> Result<Self, ApiUrlError> {
        Self::from_url(Url::parse(value.trim())?)
    }

    pub fn from_url(url: Url) -> Result<Self, ApiUrlError> {
        if !matches!(url.scheme(), "http" | "https" | "ws" | "wss") {
            return Err(ApiUrlError::Scheme(url.scheme().into()));
        }
        if url.cannot_be_a_base() || url.host().is_none() {
            return Err(ApiUrlError::CannotBeBase);
        }
        Ok(Self {
            url,
            state: PhantomData,
        })
    }

    pub fn append_path(mut self, segments: &[&str]) -> Result<ApiUrl<Complete>, ApiUrlError> {
        if segments
            .iter()
            .any(|segment| matches!(*segment, "." | ".."))
        {
            return Err(ApiUrlError::DotSegment);
        }
        self.url
            .path_segments_mut()
            .map_err(|()| ApiUrlError::CannotBeBase)?
            .pop_if_empty()
            .extend(segments.iter().copied());
        Ok(ApiUrl {
            url: self.url,
            state: PhantomData,
        })
    }

    pub fn complete_path(mut self, target: &[&str]) -> Result<ApiUrl<Complete>, ApiUrlError> {
        if target.iter().any(|segment| matches!(*segment, "." | "..")) {
            return Err(ApiUrlError::DotSegment);
        }
        let existing: Vec<Vec<u8>> = self
            .url
            .path_segments()
            .ok_or(ApiUrlError::CannotBeBase)?
            .map(|segment| percent_encoding::percent_decode_str(segment).collect())
            .collect();
        let existing = existing.strip_suffix(&[Vec::new()]).unwrap_or(&existing);
        let overlap = (0..=existing.len().min(target.len()))
            .rev()
            .find(|&length| {
                existing[existing.len() - length..]
                    .iter()
                    .map(Vec::as_slice)
                    .eq(target[..length].iter().map(|segment| segment.as_bytes()))
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
}

impl<State> ApiUrl<State> {
    pub fn into_url(self) -> Url {
        self.url
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case::root("https://example.test", "https://example.test/v1/ocr")]
    #[case::version_prefix("https://example.test/v1", "https://example.test/v1/ocr")]
    #[case::complete("https://example.test/v1/ocr", "https://example.test/v1/ocr")]
    fn completion_appends_only_the_missing_path_suffix(#[case] base: &str, #[case] expected: &str) {
        let actual = ApiUrl::parse(base)
            .and_then(|url| url.complete_path(&["v1", "ocr"]))
            .map(|url| url.into_url().to_string())
            .expect("url builds");
        assert_eq!(actual, expected);
    }

    #[rstest::rstest]
    fn completion_places_paths_before_queries() {
        let actual = ApiUrl::parse("https://example.test/v1?tenant=a")
            .and_then(|url| url.complete_path(&["v1", "ocr"]))
            .map(|url| url.into_url().to_string())
            .expect("url builds");
        assert_eq!(actual, "https://example.test/v1/ocr?tenant=a");
    }

    #[rstest::rstest]
    fn appended_query_pairs_are_encoded() {
        let actual = ApiUrl::parse("https://example.test")
            .and_then(|url| url.complete_path(&["analyze"]))
            .map(|url| {
                url.append_query_pairs([("model", "name with spaces")])
                    .into_url()
                    .to_string()
            })
            .expect("url builds");
        assert_eq!(
            actual,
            "https://example.test/analyze?model=name+with+spaces"
        );
    }
    #[rstest]
    #[case::query_fragment(
        "https://[::1]:8080/prefix/v1?tenant=a#part",
        "https://[::1]:8080/prefix/v1/ocr?tenant=a#part"
    )]
    #[case::complete_query(
        "https://example.test/prefix/v1/ocr?tenant=a#part",
        "https://example.test/prefix/v1/ocr?tenant=a#part"
    )]
    #[case::interior_empty(
        "https://example.test/prefix//v1",
        "https://example.test/prefix//v1/ocr"
    )]
    fn completion_preserves_components(#[case] input: &str, #[case] expected: &str) {
        let url = ApiUrl::from_url(Url::parse(input).unwrap())
            .unwrap()
            .complete_path(&["v1", "ocr"])
            .unwrap()
            .into_url();
        assert_eq!(url.as_str(), expected);
    }

    #[rstest]
    #[case::slash("a/b", "a%2Fb")]
    #[case::percent("a%2Fb", "a%252Fb")]
    #[case::delimiters("a?b#c", "a%3Fb%23c")]
    #[case::unicode("한글", "%ED%95%9C%EA%B8%80")]
    fn raw_segments_are_encoded_once(#[case] raw: &str, #[case] encoded: &str) {
        let url = ApiUrl::parse("https://example.test/root?tenant=a#f")
            .unwrap()
            .append_path(&[raw])
            .unwrap()
            .into_url();
        assert_eq!(url.path(), format!("/root/{encoded}"));
        assert_eq!(url.query(), Some("tenant=a"));
        assert_eq!(url.fragment(), Some("f"));
        let completed = ApiUrl::from_url(url.clone())
            .unwrap()
            .complete_path(&[raw])
            .unwrap()
            .into_url();
        assert_eq!(completed, url);
        let appended = ApiUrl::from_url(url)
            .unwrap()
            .append_path(&[raw])
            .unwrap()
            .into_url();
        assert_eq!(appended.path(), format!("/root/{encoded}/{encoded}"));
    }

    #[rstest]
    #[case::relative("relative/path")]
    #[case::opaque("mailto:name@example.test")]
    #[case::unsupported("ftp://example.test/path")]
    fn invalid_api_bases_are_rejected(#[case] input: &str) {
        assert!(ApiUrl::parse(input).is_err());
    }

    #[rstest]
    #[case::current(".")]
    #[case::parent("..")]
    fn resource_dot_segments_are_rejected(#[case] segment: &str) {
        assert!(matches!(
            ApiUrl::parse("https://example.test")
                .unwrap()
                .append_path(&[segment]),
            Err(ApiUrlError::DotSegment)
        ));
        assert!(matches!(
            ApiUrl::parse("https://example.test")
                .unwrap()
                .complete_path(&[segment]),
            Err(ApiUrlError::DotSegment)
        ));
    }
    #[rstest]
    fn completion_recognizes_equivalent_percent_encoding() {
        let input = "https://example.test/root/a%2fb?tenant=a#f";
        let url = ApiUrl::parse(input)
            .unwrap()
            .complete_path(&["a/b"])
            .unwrap()
            .into_url();
        assert_eq!(url.as_str(), input);
    }
}
