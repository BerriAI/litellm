use bytes::Bytes;
use flate2::{Decompress, DecompressError, FlushDecompress, Status};
use futures_util::StreamExt;
use reqwest::header::{CONTENT_ENCODING, HeaderMap};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

const MAX_WINDOW_BITS: u8 = 15;
const MAX_DECOMPRESS_SIZE: usize = 1 << 16;

#[derive(Debug)]
pub struct DecodingError(String);

impl std::fmt::Display for DecodingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for DecodingError {}

impl From<DecompressError> for DecodingError {
    fn from(error: DecompressError) -> Self {
        Self(match (error.needs_dictionary(), error.message()) {
            (Some(_), _) => "Error 2 while decompressing data".to_owned(),
            (None, Some(message)) => format!("Error -3 while decompressing data: {message}"),
            (None, None) => "Error -3 while decompressing data: invalid input data".to_owned(),
        })
    }
}

enum Codec {
    Gzip,
    Deflate,
}

enum FirstAttempt {
    Unread,
    Reading(Vec<u8>),
    Over,
}

struct Inflater {
    zlib: Decompress,
    first_attempt: FirstAttempt,
    pending: Vec<u8>,
    saturated: bool,
    stalled: bool,
    finished: bool,
}

impl Inflater {
    fn new(codec: Codec) -> Self {
        let (zlib, first_attempt) = match codec {
            Codec::Gzip => (Decompress::new_gzip(MAX_WINDOW_BITS), FirstAttempt::Over),
            Codec::Deflate => (Decompress::new(true), FirstAttempt::Unread),
        };
        Self {
            zlib,
            first_attempt,
            pending: Vec::new(),
            saturated: false,
            stalled: false,
            finished: false,
        }
    }

    fn push(&mut self, data: &[u8]) {
        self.first_attempt = match self.first_attempt {
            FirstAttempt::Unread => FirstAttempt::Reading(data.to_vec()),
            FirstAttempt::Reading(_) | FirstAttempt::Over => FirstAttempt::Over,
        };
        if self.finished {
            return;
        }
        self.pending.extend_from_slice(data);
        self.stalled = false;
    }

    fn has_work(&self) -> bool {
        !self.finished && (self.saturated || (!self.pending.is_empty() && !self.stalled))
    }

    fn step(&mut self) -> Result<Vec<u8>, DecodingError> {
        match (self.inflate(), &mut self.first_attempt) {
            (Err(_), FirstAttempt::Reading(first)) => {
                self.pending = std::mem::take(first);
                self.first_attempt = FirstAttempt::Over;
                self.zlib = Decompress::new_with_window_bits(false, MAX_WINDOW_BITS);
                self.saturated = false;
                self.stalled = false;
                self.step()
            }
            (inflated, attempt) => {
                if self.pending.is_empty() && !self.saturated {
                    *attempt = FirstAttempt::Over;
                }
                inflated
            }
        }
    }

    fn inflate(&mut self) -> Result<Vec<u8>, DecodingError> {
        let mut output = Vec::with_capacity(MAX_DECOMPRESS_SIZE);
        let (read, written) = (self.zlib.total_in(), self.zlib.total_out());
        let status = self
            .zlib
            .decompress_vec(&self.pending, &mut output, FlushDecompress::Sync)?;
        self.pending.drain(..(self.zlib.total_in() - read) as usize);
        self.finished = matches!(status, Status::StreamEnd);
        if self.finished {
            self.pending.clear();
        }
        self.saturated = !self.finished && output.len() == MAX_DECOMPRESS_SIZE;
        self.stalled = self.zlib.total_in() == read && self.zlib.total_out() == written;
        Ok(output)
    }
}

struct Decoder(Vec<Inflater>);

impl Decoder {
    fn detect(headers: &HeaderMap) -> Option<Self> {
        let inflaters: Vec<Inflater> = header_pairs(headers)
            .into_iter()
            .filter(|(name, _)| name.as_str() == CONTENT_ENCODING.as_str())
            .flat_map(|(_, value)| {
                value
                    .split(',')
                    .map(|coding| coding.trim().to_lowercase())
                    .collect::<Vec<_>>()
            })
            .filter_map(|coding| match coding.as_str() {
                "gzip" => Some(Codec::Gzip),
                "deflate" => Some(Codec::Deflate),
                _ => None,
            })
            .rev()
            .map(Inflater::new)
            .collect();
        (!inflaters.is_empty()).then_some(Self(inflaters))
    }

    fn feed(&mut self, chunk: &[u8]) {
        self.0[0].push(chunk);
    }

    fn next(&mut self) -> Result<Option<Bytes>, DecodingError> {
        self.pull(self.0.len() - 1)
            .map(|decoded| decoded.map(Bytes::from))
    }

    fn pull(&mut self, layer: usize) -> Result<Option<Vec<u8>>, DecodingError> {
        loop {
            if self.0[layer].has_work() {
                let decoded = self.0[layer].step()?;
                if !decoded.is_empty() {
                    return Ok(Some(decoded));
                }
                continue;
            }
            let Some(outer) = layer.checked_sub(1) else {
                return Ok(None);
            };
            match self.pull(outer)? {
                Some(decoded) => self.0[layer].push(&decoded),
                None => return Ok(None),
            }
        }
    }
}

pub(crate) fn header_pairs(headers: &HeaderMap) -> Vec<(String, String)> {
    let utf8 = headers
        .values()
        .all(|value| std::str::from_utf8(value.as_bytes()).is_ok());
    headers
        .iter()
        .map(|(name, value)| {
            let bytes = value.as_bytes();
            let text = if utf8 {
                String::from_utf8_lossy(bytes).into_owned()
            } else {
                bytes.iter().copied().map(char::from).collect()
            };
            (name.to_string(), text)
        })
        .collect()
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
                match decoder.next() {
                    Ok(Some(decoded)) => return Some((Ok(decoded), Some((body, decoder)))),
                    Ok(None) => {}
                    Err(error) => return Some((Err(BoxError::from(error)), None)),
                }
                match body.next().await? {
                    Ok(chunk) => decoder.feed(&chunk),
                    Err(error) => return Some((Err(BoxError::from(error)), None)),
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
    use futures_util::TryStreamExt;
    use rstest::rstest;

    use reqwest::header::{HeaderMap, HeaderValue};

    use super::{decoded, header_pairs};

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

    fn raw_deflate(body: &[u8]) -> Vec<u8> {
        let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(body).unwrap();
        encoder.finish().unwrap()
    }

    fn chunked(encoding: &str, wire: Vec<u8>, size: usize) -> reqwest::Response {
        let chunks: Vec<Result<Vec<u8>, std::io::Error>> =
            wire.chunks(size).map(|chunk| Ok(chunk.to_vec())).collect();
        let mut response = http::Response::new(reqwest::Body::wrap_stream(
            futures_util::stream::iter(chunks),
        ));
        response
            .headers_mut()
            .insert("content-encoding", encoding.parse().unwrap());
        reqwest::Response::from(response)
    }

    async fn read(response: reqwest::Response) -> Result<Vec<u8>, String> {
        response
            .bytes()
            .await
            .map(|bytes| bytes.to_vec())
            .map_err(|error| {
                litellm_error::find::<super::DecodingError>(&error)
                    .map_or_else(|| error.to_string(), ToString::to_string)
            })
    }

    const BODY: &[u8] = br#"{"type":"message","content":[{"type":"text","text":"Paris"}]}"#;

    #[rstest]
    #[case::padded_upper_case(" GZIP ", encoded("gzip", BODY))]
    #[case::listed_with_an_unknown_coding("gzip, br", encoded("gzip", BODY))]
    #[case::applied_twice("gzip, gzip", encoded("gzip", &encoded("gzip", BODY)))]
    #[case::deflate_without_its_zlib_header("deflate", raw_deflate(BODY))]
    #[case::gzip_with_data_past_its_end("gzip", [encoded("gzip", BODY), b"\x1f\x8b trailing".to_vec()].concat())]
    #[tokio::test]
    async fn bodies_decode_as_httpx_decodes_them(
        #[case] encoding: &str,
        #[case] wire: Vec<u8>,
        #[values(7, 4096)] chunk: usize,
    ) {
        assert_eq!(
            read(decoded(chunked(encoding, wire, chunk))).await.unwrap(),
            BODY
        );
    }

    #[rstest]
    #[case::x_gzip_alias(" X-Gzip ")]
    #[case::unknown_only("br, identity")]
    #[tokio::test]
    async fn codings_httpx_has_no_decoder_for_pass_through(#[case] encoding: &str) {
        let wire = encoded("gzip", BODY);
        assert_eq!(
            read(decoded(chunked(encoding, wire.clone(), 4096)))
                .await
                .unwrap(),
            wire
        );
    }

    #[rstest]
    #[case::not_deflate("deflate", b"not deflate at all".to_vec(), "Error -3 while decompressing data: invalid block type")]
    #[case::bad_gzip_trailer(
        "gzip",
        { let wire = encoded("gzip", BODY); [&wire[..wire.len() - 8], &[0; 8]].concat() },
        "Error -3 while decompressing data: incorrect data check"
    )]
    #[case::not_gzip("gzip", b"\x1f\x8b\x08\x00not really gzip data at all".to_vec(), "Error -3 while decompressing data: invalid stored block lengths")]
    #[case::no_gzip_magic("gzip", b"plain text".to_vec(), "Error -3 while decompressing data: incorrect header check")]
    #[tokio::test]
    async fn corrupt_bodies_fail_with_zlib_s_error(
        #[case] encoding: &str,
        #[case] wire: Vec<u8>,
        #[case] expected: &str,
    ) {
        assert_eq!(
            read(decoded(chunked(encoding, wire, 4096)))
                .await
                .unwrap_err(),
            expected
        );
    }

    #[rstest]
    #[case::gzip_in_gzip("gzip, gzip")]
    #[case::deflate_in_gzip("deflate, gzip")]
    #[tokio::test]
    async fn a_nested_bomb_is_decoded_in_bounded_reads(#[case] encoding: &str) {
        let zeros = vec![0; 8 << 20];
        let inner = if encoding.starts_with("deflate") {
            encoded("deflate", &zeros)
        } else {
            encoded("gzip", &zeros)
        };
        let wire = encoded("gzip", &inner);
        let sizes: Vec<usize> = decoded(chunked(encoding, wire, 4096))
            .bytes_stream()
            .map_ok(|chunk| chunk.len())
            .try_collect()
            .await
            .unwrap();
        assert!(
            sizes.iter().all(|size| *size <= super::MAX_DECOMPRESS_SIZE),
            "{sizes:?}"
        );
        assert_eq!(sizes.iter().sum::<usize>(), zeros.len());
    }

    #[rstest]
    #[tokio::test]
    async fn a_truncated_body_yields_what_it_holds_without_an_error() {
        let wire = encoded("gzip", BODY);
        let decoded_body = read(decoded(chunked(
            "gzip",
            wire[..wire.len() - 8].to_vec(),
            4096,
        )))
        .await
        .unwrap();
        assert_eq!(decoded_body, BODY);
    }

    #[rstest]
    #[case::ascii(&[b"plain".as_slice()], &["plain"])]
    #[case::utf_8(&[b"plain".as_slice(), b"caf\xc3\xa9"], &["plain", "caf\u{e9}"])]
    #[case::latin_1_once_any_value_is_not_utf_8(
        &[b"caf\xc3\xa9".as_slice(), b"caf\xe9"],
        &["caf\u{c3}\u{a9}", "caf\u{e9}"]
    )]
    fn header_values_decode_as_httpx_decodes_them(
        #[case] values: &[&[u8]],
        #[case] expected: &[&str],
    ) {
        let headers: HeaderMap = values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                (
                    format!("x-fz-{index}")
                        .parse::<reqwest::header::HeaderName>()
                        .unwrap(),
                    HeaderValue::from_bytes(value).unwrap(),
                )
            })
            .collect();
        let decoded: Vec<String> = header_pairs(&headers)
            .into_iter()
            .map(|(_, value)| value)
            .collect();
        assert_eq!(decoded, expected);
    }
}
