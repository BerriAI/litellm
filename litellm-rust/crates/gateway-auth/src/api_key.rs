use axum::http::HeaderMap;

/// The key a request presents, read the way `get_api_key` in `user_api_key_auth.py` reads it.
#[derive(Debug, PartialEq, Eq)]
pub enum ApiKey {
    /// The first key header in Python's order, with its scheme stripped the way Python strips it.
    Token(String),
    /// No key header at all.
    Missing,
    /// A key header whose value is not visible ASCII, which Rust can't read the way Python does.
    Unreadable,
}

#[derive(Clone, Copy)]
enum Scheme {
    /// `_get_bearer_token_or_received_api_key`: a value without a known scheme is the key itself.
    Received,
    /// `get_bearer_token`: a value without a known scheme is no key.
    Bearer,
    /// Taken as sent.
    Raw,
}

/// Python's precedence. An empty header counts as absent, as FastAPI's `APIKeyHeader` returns
/// `None` for it.
const HEADERS: [(&str, Scheme); 6] = [
    ("x-litellm-api-key", Scheme::Received),
    ("authorization", Scheme::Bearer),
    ("api-key", Scheme::Raw),
    ("x-api-key", Scheme::Raw),
    ("x-goog-api-key", Scheme::Raw),
    ("ocp-apim-subscription-key", Scheme::Raw),
];

const SCHEMES: [&str; 3] = ["Bearer ", "Basic ", "bearer "];
const AWS_SIGNATURE: &str = "AWS4-HMAC-SHA256";
const AWS_CREDENTIAL: &str = "Credential=";

impl ApiKey {
    pub fn from_headers(headers: &HeaderMap) -> Self {
        let first = HEADERS.iter().find_map(|(name, scheme)| {
            let value = headers.get(*name)?;
            (!value.is_empty()).then_some((value.to_str(), *scheme))
        });
        match first {
            None => Self::Missing,
            Some((Err(_), _)) => Self::Unreadable,
            Some((Ok(value), scheme)) => Self::Token(scheme.token(value)),
        }
    }

    /// The token to plan on. A missing key plans as an empty token, like Python's.
    pub fn token(&self) -> Option<&str> {
        match self {
            Self::Token(token) => Some(token),
            Self::Missing => Some(""),
            Self::Unreadable => None,
        }
    }
}

impl Scheme {
    fn token(self, value: &str) -> String {
        let unparsed = match self {
            Self::Raw => return value.to_owned(),
            Self::Received => value,
            Self::Bearer => "",
        };
        // Python strips with `str.replace`, so every occurrence of the scheme goes, not only the prefix.
        if let Some(scheme) = SCHEMES.iter().find(|scheme| value.starts_with(**scheme)) {
            return value.replace(scheme, "");
        }
        match value.starts_with(AWS_SIGNATURE) {
            true => aws_credential(value).unwrap_or(unparsed).to_owned(),
            false => unparsed.to_owned(),
        }
    }
}

/// `re.search(r"Credential=Bearer\s+([^/\s,]+)")`, then `re.search(r"Credential=([^/\s,]+)")`.
fn aws_credential(value: &str) -> Option<&str> {
    let credentials = || {
        value
            .match_indices(AWS_CREDENTIAL)
            .map(|(at, _)| &value[at + AWS_CREDENTIAL.len()..])
    };
    credentials()
        .find_map(|rest| {
            let after = rest.strip_prefix("Bearer")?;
            let token = after.trim_start();
            (token.len() < after.len())
                .then_some(token)
                .and_then(credential_value)
        })
        .or_else(|| credentials().find_map(credential_value))
}

fn credential_value(rest: &str) -> Option<&str> {
    let end = rest
        .find(|c: char| c == '/' || c == ',' || c.is_whitespace())
        .unwrap_or(rest.len());
    (end > 0).then(|| &rest[..end])
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderValue, header::AUTHORIZATION};
    use rstest::rstest;

    use super::*;

    const AWS_BEARER: &str = "AWS4-HMAC-SHA256 Credential=Bearer sk-1/20240101/us-east-1/bedrock/aws4_request, SignedHeaders=host, Signature=abc";
    const AWS_PLAIN: &str =
        "AWS4-HMAC-SHA256 Credential=AKIA1/20240101/us-east-1/bedrock/aws4_request, Signature=abc";

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        pairs
            .iter()
            .map(|(name, value)| (*name, HeaderValue::from_str(value).unwrap()))
            .fold(HeaderMap::new(), |mut map, (name, value)| {
                map.append(name, value);
                map
            })
    }

    fn token(token: &str) -> ApiKey {
        ApiKey::Token(token.to_owned())
    }

    #[rstest]
    #[case::bearer(&[("authorization", "Bearer sk-1")], token("sk-1"))]
    #[case::lowercase_bearer(&[("authorization", "bearer sk-1")], token("sk-1"))]
    #[case::basic(&[("authorization", "Basic sk-1")], token("sk-1"))]
    #[case::every_occurrence_stripped(&[("authorization", "Bearer sk-Bearer 1")], token("sk-1"))]
    #[case::unknown_scheme_is_no_key(&[("authorization", "Token sk-1")], token(""))]
    #[case::aws_bearer(&[("authorization", AWS_BEARER)], token("sk-1"))]
    #[case::aws_credential(&[("authorization", AWS_PLAIN)], token("AKIA1"))]
    #[case::aws_without_credential(&[("authorization", "AWS4-HMAC-SHA256 Signature=abc")], token(""))]
    #[case::litellm_header_bearer(&[("x-litellm-api-key", "Bearer sk-1")], token("sk-1"))]
    #[case::litellm_header_raw(&[("x-litellm-api-key", "sk-1")], token("sk-1"))]
    #[case::litellm_header_aws_without_credential(&[("x-litellm-api-key", "AWS4-HMAC-SHA256 x")], token("AWS4-HMAC-SHA256 x"))]
    #[case::azure(&[("api-key", "sk-1")], token("sk-1"))]
    #[case::anthropic(&[("x-api-key", "sk-1")], token("sk-1"))]
    #[case::google(&[("x-goog-api-key", "sk-1")], token("sk-1"))]
    #[case::apim(&[("ocp-apim-subscription-key", "sk-1")], token("sk-1"))]
    #[case::raw_keeps_scheme(&[("x-api-key", "Bearer sk-1")], token("Bearer sk-1"))]
    #[case::litellm_header_wins(&[("authorization", "Bearer sk-2"), ("x-litellm-api-key", "sk-1")], token("sk-1"))]
    #[case::authorization_before_raw(&[("x-api-key", "sk-2"), ("authorization", "Bearer sk-1")], token("sk-1"))]
    #[case::azure_before_anthropic(&[("x-api-key", "sk-2"), ("api-key", "sk-1")], token("sk-1"))]
    #[case::empty_header_is_absent(&[("authorization", ""), ("x-api-key", "sk-1")], token("sk-1"))]
    #[case::first_of_duplicates(&[("authorization", "Bearer sk-1"), ("authorization", "Bearer sk-2")], token("sk-1"))]
    #[case::missing(&[("content-type", "application/json")], ApiKey::Missing)]
    fn reads_the_key_python_reads(
        #[case] pairs: &[(&'static str, &str)],
        #[case] expected: ApiKey,
    ) {
        assert_eq!(ApiKey::from_headers(&headers(pairs)), expected);
    }

    #[rstest]
    fn a_value_rust_cannot_read_is_unreadable() {
        let headers = HeaderMap::from_iter([(
            AUTHORIZATION,
            HeaderValue::from_bytes(b"Bearer sk-\xe9").unwrap(),
        )]);
        assert_eq!(ApiKey::from_headers(&headers), ApiKey::Unreadable);
        assert_eq!(ApiKey::Unreadable.token(), None);
    }

    #[rstest]
    fn a_missing_key_plans_as_empty() {
        assert_eq!(ApiKey::Missing.token(), Some(""));
    }
}
