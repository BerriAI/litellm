use jsonwebtoken::dangerous::insecure_decode_claims;
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq)]
pub struct Claims(pub Map<String, Value>);

pub fn is_jwt(token: &str) -> bool {
    token.split('.').count() == 3
}

/// `get_unverified_claims`: the payload of a well-formed token, signature unchecked.
pub fn unverified_claims(token: &str) -> Option<Claims> {
    if !is_jwt(token) {
        return None;
    }
    insecure_decode_claims(token).ok().map(Claims)
}

#[cfg(test)]
mod tests {
    use base64::{Engine, engine::general_purpose::URL_SAFE_NO_PAD};
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::test_support::unsigned_jwt;

    fn b64(value: &Value) -> String {
        URL_SAFE_NO_PAD.encode(serde_json::to_vec(value).unwrap())
    }

    fn raw_token(header: &str, payload: &str) -> String {
        format!("{header}.{payload}.AAAA")
    }

    #[rstest]
    #[case::bad_signature_still_read(&unsigned_jwt(&json!({"a": 1})), Some(json!({"a": 1})))]
    #[case::two_segments("e30.e30", None)]
    #[case::payload_not_object(&raw_token(&b64(&json!({"alg": "RS256"})), &b64(&json!([1]))), None)]
    #[case::detached_payload(&raw_token(&b64(&json!({"alg": "RS256", "b64": false, "crit": ["b64"]})), ""), None)]
    fn unverified_claims_read_the_payload(#[case] token: &str, #[case] expected: Option<Value>) {
        let expected = expected.map(|claims| match claims {
            Value::Object(map) => Claims(map),
            other => panic!("claims must be an object: {other}"),
        });
        assert_eq!(unverified_claims(token), expected);
    }
}
