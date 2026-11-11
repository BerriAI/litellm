use std::io::Write;

use bytes::Bytes;
use flate2::write::{GzDecoder, ZlibDecoder};
use futures_util::StreamExt;
use reqwest::header::{CONTENT_ENCODING, HeaderMap};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

enum Decoder {
    Gzip(GzDecoder<Vec<u8>>),
    Deflate(ZlibDecoder<Vec<u8>>),
}

impl Decoder {
    fn detect(headers: &HeaderMap) -> Option<Self> {
        let encoding = headers.get(CONTENT_ENCODING)?.to_str().ok()?;
        match encoding.trim().to_ascii_lowercase().as_str() {
            "gzip" | "x-gzip" => Some(Self::Gzip(GzDecoder::new(Vec::new()))),
            "deflate" => Some(Self::Deflate(ZlibDecoder::new(Vec::new()))),
            _ => None,
        }
    }

    fn feed(&mut self, chunk: &[u8]) -> std::io::Result<Bytes> {
        let decoded = match self {
            Self::Gzip(decoder) => {
                decoder.write_all(chunk)?;
                decoder.get_mut()
            }
            Self::Deflate(decoder) => {
                decoder.write_all(chunk)?;
                decoder.get_mut()
            }
        };
        Ok(Bytes::from(std::mem::take(decoded)))
    }

    fn finish(self) -> std::io::Result<Bytes> {
        match self {
            Self::Gzip(decoder) => decoder.finish(),
            Self::Deflate(decoder) => decoder.finish(),
        }
        .map(Bytes::from)
    }
}

pub(crate) fn decoded(response: reqwest::Response) -> reqwest::Response {
    let Some(decoder) = Decoder::detect(response.headers()) else {
        return response;
    };
    let status = response.status();
    let version = response.version();
    let headers = response.headers().clone();
    let body = futures_util::stream::unfold(
        Some((response.bytes_stream().boxed(), decoder)),
        |state| async move {
            let (mut body, mut decoder) = state?;
            loop {
                let fed = match body.next().await {
                    Some(Ok(chunk)) => decoder.feed(&chunk).map_err(BoxError::from),
                    Some(Err(error)) => Err(BoxError::from(error)),
                    None => {
                        return match decoder.finish() {
                            Ok(rest) if rest.is_empty() => None,
                            Ok(rest) => Some((Ok(rest), None)),
                            Err(error) => Some((Err(BoxError::from(error)), None)),
                        };
                    }
                };
                match fed {
                    Ok(decoded) if decoded.is_empty() => {}
                    Ok(decoded) => return Some((Ok(decoded), Some((body, decoder)))),
                    Err(error) => return Some((Err(error), None)),
                }
            }
        },
    );
    let mut rebuilt = http::Response::new(reqwest::Body::wrap_stream(body));
    *rebuilt.status_mut() = status;
    *rebuilt.version_mut() = version;
    *rebuilt.headers_mut() = headers;
    reqwest::Response::from(rebuilt)
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::{Compression, write::GzEncoder, write::ZlibEncoder};
    use rstest::rstest;

    use super::decoded;

    fn encoded(encoding: &str, body: &[u8]) -> Vec<u8> {
        match encoding {
            "gzip" => {
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(body).unwrap();
                encoder.finish().unwrap()
            }
            "deflate" => {
                let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(body).unwrap();
                encoder.finish().unwrap()
            }
            _ => body.to_vec(),
        }
    }

    fn response(encoding: &str, body: Vec<u8>) -> reqwest::Response {
        let length = body.len().to_string();
        let mut response = http::Response::new(reqwest::Body::from(body));
        let headers = response.headers_mut();
        headers.insert("content-encoding", encoding.parse().unwrap());
        headers.insert("content-length", length.parse().unwrap());
        reqwest::Response::from(response)
    }

    #[rstest]
    #[case::gzip("gzip")]
    #[case::deflate("deflate")]
    #[case::unknown("br")]
    #[tokio::test]
    async fn decoded_bodies_keep_the_head_the_provider_sent(#[case] encoding: &str) {
        let body = br#"{"type":"message"}"#;
        let wire = encoded(encoding, body);
        let length = wire.len().to_string();
        let response = decoded(response(encoding, wire.clone()));

        assert_eq!(response.headers()["content-encoding"], encoding);
        assert_eq!(response.headers()["content-length"], length.as_str());
        let expected = if encoding == "br" {
            wire
        } else {
            body.to_vec()
        };
        assert_eq!(
            response.bytes().await.unwrap().as_ref(),
            expected.as_slice()
        );
    }
}
