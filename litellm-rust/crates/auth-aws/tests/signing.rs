use std::collections::{BTreeMap, BTreeSet};
use std::time::SystemTime;

use litellm_auth_aws::{Credentials, SigV4Request, sign_v4, static_credentials};

const URI: &str = "https://sts.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15";
const SIGNING_TIME: SystemTime = SystemTime::UNIX_EPOCH;

fn authorization(
    method: &str,
    uri: &str,
    body: &[u8],
    headers: &BTreeMap<String, String>,
    region: &str,
    service: &str,
    credentials: &Credentials,
) -> String {
    sign_v4(
        SigV4Request {
            method,
            uri,
            body,
            headers,
            region,
            service,
            signing_time: SIGNING_TIME,
        },
        credentials,
    )
    .expect("signature")
    .remove("Authorization")
    .expect("authorization header")
}

#[test]
fn every_signed_request_dimension_changes_the_authorization() {
    let headers = BTreeMap::from([(
        "Content-Type".into(),
        "application/x-www-form-urlencoded".into(),
    )]);
    let changed_headers = BTreeMap::from([("Content-Type".into(), "application/json".into())]);
    let credentials = static_credentials("AKIDEXAMPLE", "secret");
    let baseline = authorization(
        "POST",
        URI,
        b"payload",
        &headers,
        "us-east-1",
        "sts",
        &credentials,
    );
    let signatures = BTreeSet::from([
        baseline.clone(),
        authorization(
            "GET",
            URI,
            b"payload",
            &headers,
            "us-east-1",
            "sts",
            &credentials,
        ),
        authorization(
            "POST",
            "https://sts.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-14",
            b"payload",
            &headers,
            "us-east-1",
            "sts",
            &credentials,
        ),
        authorization(
            "POST",
            URI,
            b"payloaD",
            &headers,
            "us-east-1",
            "sts",
            &credentials,
        ),
        authorization(
            "POST",
            URI,
            b"payload",
            &changed_headers,
            "us-east-1",
            "sts",
            &credentials,
        ),
        authorization(
            "POST",
            URI,
            b"payload",
            &headers,
            "us-west-2",
            "sts",
            &credentials,
        ),
        authorization(
            "POST",
            URI,
            b"payload",
            &headers,
            "us-east-1",
            "bedrock",
            &credentials,
        ),
    ]);

    assert_eq!(signatures.len(), 7);
}

#[test]
fn session_token_is_emitted_and_covered_by_the_signature() {
    let headers = BTreeMap::new();
    let first = Credentials::new(
        "AKIDEXAMPLE",
        "secret",
        Some("first-session".into()),
        None,
        "test",
    );
    let second = Credentials::new(
        "AKIDEXAMPLE",
        "secret",
        Some("second-session".into()),
        None,
        "test",
    );
    let signed = sign_v4(
        SigV4Request {
            method: "GET",
            uri: URI,
            body: b"",
            headers: &headers,
            region: "us-east-1",
            service: "sts",
            signing_time: SIGNING_TIME,
        },
        &first,
    )
    .expect("signature");
    let first_authorization = signed.get("Authorization").expect("authorization");
    let second_authorization =
        authorization("GET", URI, b"", &headers, "us-east-1", "sts", &second);

    assert_eq!(
        signed.get("X-Amz-Security-Token").map(String::as_str),
        Some("first-session")
    );
    assert!(first_authorization.contains("x-amz-security-token"));
    assert_ne!(first_authorization, &second_authorization);
}
