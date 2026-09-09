use serde::Serialize;

use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestBodyPolicy {
    StructuredAtSend,
    StructuredAtBuild,
    SerializedAtBuild,
}

#[derive(Debug, PartialEq, Eq)]
pub struct WireBody {
    bytes: Vec<u8>,
}

impl WireBody {
    pub fn encode<T: Serialize>(value: &T, operation: &str) -> Result<Self, Error> {
        serde_json::to_vec(value)
            .map(|bytes| Self { bytes })
            .map_err(|error| {
                Error::InvalidRequest(format!("could not encode {operation}: {error}"))
            })
    }

    pub fn from_serialized(value: String) -> Self {
        Self {
            bytes: value.into_bytes(),
        }
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct AuthorizedBody {
    body: WireBody,
    headers: Vec<(String, String)>,
}

impl AuthorizedBody {
    pub(crate) fn new(body: WireBody, headers: Vec<(String, String)>) -> Self {
        Self { body, headers }
    }

    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    pub fn body(&self) -> &[u8] {
        self.body.as_bytes()
    }

    pub fn settle(self) -> SettledHttpRequest {
        SettledHttpRequest {
            body: self.body,
            headers: self.headers,
        }
    }

    pub fn settle_headers(self, headers: Vec<(String, String)>) -> SettledHttpRequest {
        SettledHttpRequest {
            body: self.body,
            headers,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum PreCallBody<T> {
    StructuredAtSend {
        callback: T,
    },
    StructuredAtBuild {
        callback: T,
        authorized: AuthorizedBody,
    },
    SerializedAtBuild {
        callback: String,
        authorized: AuthorizedBody,
    },
}

impl<T> PreCallBody<T> {
    pub fn policy(&self) -> RequestBodyPolicy {
        match self {
            Self::StructuredAtSend { .. } => RequestBodyPolicy::StructuredAtSend,
            Self::StructuredAtBuild { .. } => RequestBodyPolicy::StructuredAtBuild,
            Self::SerializedAtBuild { .. } => RequestBodyPolicy::SerializedAtBuild,
        }
    }

    pub fn structured_callback(&self) -> Option<&T> {
        match self {
            Self::StructuredAtSend { callback } | Self::StructuredAtBuild { callback, .. } => {
                Some(callback)
            }
            Self::SerializedAtBuild { .. } => None,
        }
    }

    pub fn authorized(&self) -> Option<&AuthorizedBody> {
        match self {
            Self::StructuredAtSend { .. } => None,
            Self::StructuredAtBuild { authorized, .. }
            | Self::SerializedAtBuild { authorized, .. } => Some(authorized),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct SettledHttpRequest {
    body: WireBody,
    headers: Vec<(String, String)>,
}

impl SettledHttpRequest {
    pub fn body(&self) -> &[u8] {
        self.body.as_bytes()
    }

    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    pub fn into_parts(self) -> (Vec<u8>, Vec<(String, String)>) {
        (self.body.into_bytes(), self.headers)
    }

    pub fn replace_headers(self, headers: Vec<(String, String)>) -> Self {
        Self {
            body: self.body,
            headers,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn build_time_capture_ignores_later_structured_mutation() {
        let mut callback = json!({"value": "before"});
        let wire = WireBody::encode(&callback, "test body").unwrap();
        let authorized = AuthorizedBody::new(wire, vec![]);
        callback["value"] = json!("after");

        let settled = authorized.settle();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(settled.body()).unwrap(),
            json!({"value": "before"})
        );
    }

    #[test]
    fn send_time_capture_reflects_in_place_mutation() {
        let mut callback = json!({"value": "before"});
        callback["value"] = json!("after");
        let settled =
            AuthorizedBody::new(WireBody::encode(&callback, "test body").unwrap(), vec![]);
        let settled = settled.settle();

        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(settled.body()).unwrap(),
            json!({"value": "after"})
        );
    }

    #[test]
    fn settling_headers_cannot_replace_authorized_bytes() {
        let authorized = AuthorizedBody::new(
            WireBody::from_serialized("{\"signed\":true}".to_string()),
            vec![("authorization".into(), "signature".into())],
        );
        let settled = authorized.settle_headers(vec![("x-callback".into(), "mutated".into())]);

        assert_eq!(settled.body(), br#"{"signed":true}"#);
        assert_eq!(
            settled.headers(),
            &[("x-callback".into(), "mutated".into())]
        );
    }

    #[test]
    fn replacing_build_time_callback_projection_does_not_replace_wire_body() {
        let authorized = AuthorizedBody::new(
            WireBody::encode(&json!({"wire": "original"}), "test body").unwrap(),
            vec![],
        );
        let mut pre_call = PreCallBody::StructuredAtBuild {
            callback: json!({"wire": "original"}),
            authorized,
        };
        let PreCallBody::StructuredAtBuild {
            callback,
            authorized,
        } = &mut pre_call
        else {
            unreachable!()
        };
        *callback = json!({"wire": "replacement"});

        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(authorized.body()).unwrap(),
            json!({"wire": "original"})
        );
    }

    #[test]
    fn authorization_and_transport_observe_the_same_owned_bytes() {
        let wire = WireBody::from_serialized("{ \"exact\": true }\n".to_string());
        let authorization_input = wire.as_bytes().to_vec();
        let settled = AuthorizedBody::new(
            wire,
            vec![(
                "x-authorized-length".into(),
                authorization_input.len().to_string(),
            )],
        )
        .settle();
        let (transport_body, headers) = settled.into_parts();

        assert_eq!(transport_body, authorization_input);
        assert_eq!(
            headers,
            vec![(
                "x-authorized-length".into(),
                transport_body.len().to_string()
            )]
        );
    }
}
