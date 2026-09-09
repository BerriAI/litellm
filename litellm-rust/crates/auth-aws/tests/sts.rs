use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::mpsc::{self, Receiver};
use std::thread::JoinHandle;
use std::time::Duration;

use litellm_auth_aws::{
    AssumeRoleRequest, Error, WebIdentityRequest, assume_role_credentials, static_credentials,
    web_identity_credentials,
};

const EXPIRATION: &str = "2035-01-02T03:04:05Z";

fn sts_server(response_body: String) -> (String, Receiver<String>, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener");
    let endpoint = format!("http://{}", listener.local_addr().expect("address"));
    let (sender, receiver) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("request");
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .expect("read timeout");
        let mut request = Vec::new();
        let mut buffer = [0; 4096];
        loop {
            let count = stream.read(&mut buffer).expect("request bytes");
            assert_ne!(count, 0, "request ended before its body arrived");
            request.extend_from_slice(&buffer[..count]);
            if request_is_complete(&request) {
                break;
            }
        }
        sender
            .send(String::from_utf8(request).expect("utf-8 request"))
            .expect("captured request");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/xml\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        stream.write_all(response.as_bytes()).expect("response");
    });
    (endpoint, receiver, handle)
}

fn request_is_complete(request: &[u8]) -> bool {
    let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
        return false;
    };
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case("content-length")
            .then(|| value.trim().parse::<usize>().expect("content length"))
    });
    request.len() >= header_end + 4 + content_length.unwrap_or(0)
}

fn credentials_xml() -> String {
    format!(
        "<Credentials><AccessKeyId>acquired-access-key</AccessKeyId><SecretAccessKey>acquired-secret-key</SecretAccessKey><SessionToken>acquired-session-token</SessionToken><Expiration>{EXPIRATION}</Expiration></Credentials>"
    )
}

#[tokio::test]
async fn assume_role_sends_the_exact_policy_inputs_and_parses_credentials() {
    let response_body = format!(
        "<AssumeRoleResponse xmlns=\"https://sts.amazonaws.com/doc/2011-06-15/\"><AssumeRoleResult>{}<AssumedRoleUser><Arn>arn:aws:sts::123456789012:assumed-role/demo/test-session</Arn><AssumedRoleId>id:test-session</AssumedRoleId></AssumedRoleUser></AssumeRoleResult><ResponseMetadata><RequestId>request-id</RequestId></ResponseMetadata></AssumeRoleResponse>",
        credentials_xml()
    );
    let (endpoint, request, server) = sts_server(response_body);
    let credentials = assume_role_credentials(AssumeRoleRequest {
        role: "arn:aws:iam::123456789012:role/path/demo".into(),
        session_name: "test-session".into(),
        region: Some("us-west-2".into()),
        endpoint: Some(endpoint),
        source_credentials: Some(static_credentials("source-access-key", "source-secret-key")),
        external_id: Some("external-value".into()),
    })
    .await
    .expect("assume role credentials");
    let request = request.recv().expect("captured request");
    server.join().expect("server");

    assert!(request.starts_with("POST / HTTP/1.1\r\n"));
    assert!(request.contains("Action=AssumeRole"));
    assert!(request.contains("RoleArn=arn%3Aaws%3Aiam%3A%3A123456789012%3Arole%2Fpath%2Fdemo"));
    assert!(request.contains("RoleSessionName=test-session"));
    assert!(request.contains("ExternalId=external-value"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: aws4-hmac-sha256 credential=source-access-key/")
    );
    assert_eq!(credentials.access_key_id(), "acquired-access-key");
    assert_eq!(credentials.session_token(), Some("acquired-session-token"));
}

#[tokio::test]
async fn web_identity_sends_the_token_and_parses_credentials() {
    let response_body = format!(
        "<AssumeRoleWithWebIdentityResponse xmlns=\"https://sts.amazonaws.com/doc/2011-06-15/\"><AssumeRoleWithWebIdentityResult>{}<AssumedRoleUser><Arn>arn:aws:sts::123456789012:assumed-role/demo/web-session</Arn><AssumedRoleId>id:web-session</AssumedRoleId></AssumedRoleUser><Audience>audience</Audience><Provider>provider</Provider><SubjectFromWebIdentityToken>subject</SubjectFromWebIdentityToken></AssumeRoleWithWebIdentityResult><ResponseMetadata><RequestId>request-id</RequestId></ResponseMetadata></AssumeRoleWithWebIdentityResponse>",
        credentials_xml()
    );
    let (endpoint, request, server) = sts_server(response_body);
    let credentials = web_identity_credentials(WebIdentityRequest {
        token: "header.payload.signature".into(),
        role: "arn:aws:iam::123456789012:role/demo".into(),
        session_name: "web-session".into(),
        region: Some("us-east-1".into()),
        endpoint: Some(endpoint),
    })
    .await
    .expect("web identity credentials");
    let request = request.recv().expect("captured request");
    server.join().expect("server");

    assert!(request.starts_with("POST / HTTP/1.1\r\n"));
    assert!(request.contains("Action=AssumeRoleWithWebIdentity"));
    assert!(request.contains("RoleArn=arn%3Aaws%3Aiam%3A%3A123456789012%3Arole%2Fdemo"));
    assert!(request.contains("RoleSessionName=web-session"));
    assert!(request.contains("WebIdentityToken=header.payload.signature"));
    assert!(!request.to_ascii_lowercase().contains("\r\nauthorization:"));
    assert_eq!(credentials.access_key_id(), "acquired-access-key");
    assert_eq!(credentials.session_token(), Some("acquired-session-token"));
}

#[tokio::test]
async fn web_identity_rejects_a_success_response_without_credentials() {
    let response_body = "<AssumeRoleWithWebIdentityResponse xmlns=\"https://sts.amazonaws.com/doc/2011-06-15/\"><AssumeRoleWithWebIdentityResult></AssumeRoleWithWebIdentityResult><ResponseMetadata><RequestId>request-id</RequestId></ResponseMetadata></AssumeRoleWithWebIdentityResponse>".to_string();
    let (endpoint, request, server) = sts_server(response_body);
    let error = web_identity_credentials(WebIdentityRequest {
        token: "header.payload.signature".into(),
        role: "arn:aws:iam::123456789012:role/demo".into(),
        session_name: "web-session".into(),
        region: Some("us-east-1".into()),
        endpoint: Some(endpoint),
    })
    .await
    .expect_err("missing credentials");
    request.recv().expect("captured request");
    server.join().expect("server");

    assert_eq!(error, Error::MissingWebIdentityCredentials);
}
