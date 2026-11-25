use bytes::Bytes;
use futures_util::{future::BoxFuture, stream::BoxStream};

use crate::Error;

pub struct TransportRequest {
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Bytes,
    pub stream: bool,
}

pub enum Delivered {
    Complete(String),
    Stream(BoxStream<'static, Result<Bytes, Error>>),
    Rejected { text: String, error: Error },
}

pub struct TransportResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub delivered: Delivered,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnencodableHeader {
    Sent(Vec<u8>),
    Received(Vec<u8>),
}

impl std::fmt::Display for UnencodableHeader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a header value is not ASCII, which Python's HTTP client cannot encode")
    }
}

impl std::error::Error for UnencodableHeader {}

impl UnencodableHeader {
    pub(crate) fn received(headers: &reqwest::header::HeaderMap) -> Option<Self> {
        headers
            .values()
            .find(|value| !value.as_bytes().is_ascii())
            .map(|value| Self::Received(value.as_bytes().to_vec()))
    }
}

/// A URL with a non-printable ASCII character, which Python's HTTP client refuses to send. It
/// holds the message the refusal is raised with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnsendableUrl(pub String);

impl std::fmt::Display for UnsendableUrl {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for UnsendableUrl {}

impl UnsendableUrl {
    pub(crate) fn of(url: &str) -> Option<Self> {
        let (position, character) = url
            .char_indices()
            .enumerate()
            .find(|(_, (_, character))| character.is_ascii() && !(' '..='~').contains(character))
            .map(|(position, (_, character))| (position, character))?;
        Some(Self(format!(
            "Invalid URL: it holds the non-printable ASCII character {character:?} at position {position}."
        )))
    }
}

pub trait Transport: Send + Sync {
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'static, Result<TransportResponse, Error>>;
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::newline("https://us-east5\n-aiplatform.googleapis.com/v1", true)]
    #[case::tab("https://h/\tpath", true)]
    #[case::delete("https://h/\u{7f}", true)]
    #[case::printable("https://us-east5-aiplatform.googleapis.com/v1?a=b c", false)]
    #[case::non_ascii("https://h/caf\u{e9}", false)]
    fn a_url_python_s_client_refuses_holds_a_non_printable_ascii_character(
        #[case] url: &str,
        #[case] refused: bool,
    ) {
        assert_eq!(UnsendableUrl::of(url).is_some(), refused);
    }

    #[rstest]
    fn the_refusal_names_the_character_and_its_position() {
        assert_eq!(
            UnsendableUrl::of("https://us-east5\n-aiplatform.googleapis.com/v1")
                .unwrap()
                .to_string(),
            "Invalid URL: it holds the non-printable ASCII character '\\n' at position 16."
        );
    }
}
