//! An external account whose `credential_source` is a file, with the file read under the
//! credentials file size limit on a blocking thread. Google's Rust SDK (`google_cloud_auth` 1.16)
//! reads such a file whole with a synchronous `read_to_string` on the runtime, so a file such as
//! `/dev/zero` would never finish. This module reads it bounded, takes the subject token from it
//! as the SDK and google-auth's `identity_pool` do, and leaves the STS exchange and any
//! impersonation to the SDK. It sits beside [`super::legacy`] rather than in it because the SDK
//! does support file sources; it only hardens the read. Delete it once the SDK bounds the read.

use std::sync::Arc;

use google_cloud_auth::credentials::subject_token::{self, SubjectToken, SubjectTokenProvider};

use super::{
    builders::{exchanged, programmatic},
    credentials::{BoundedRead, ExternalAccountInfo, SourceFormat, read_bounded_off_runtime},
    environment::GoogleEnvironment,
    failure::{GoogleLoginFailure, PythonException, UNLOADABLE_CREDENTIALS},
};
use crate::GcpTokenSource;

const JSON_FORMAT: &str = "json";

const UNREADABLE: &str = "Vertex: the external account's credential_source file cannot be read.";
const TOO_LARGE: &str =
    "Vertex: the external account's credential_source file is larger than any subject token.";
const NOT_TEXT: &str = "Vertex: the external account's credential_source file is not UTF-8 text.";
const NO_TOKEN: &str =
    "Vertex: the external account's credential_source file holds no subject token.";

/// How the file holds its subject token: the whole file, or one text field of a JSON object.
#[derive(Debug, PartialEq)]
enum Format {
    Text,
    Json(String),
}

/// The file an external account reads its subject token from.
#[derive(Debug)]
struct SubjectFile {
    path: String,
    format: Format,
}

/// The subject token file an external account reads, `None` for any other source.
pub(crate) fn subject_file(info: &ExternalAccountInfo) -> Option<&str> {
    let source = info.credential_source.as_ref()?;
    match (&source.environment_id, &source.url, &source.file) {
        (None, None, Some(path)) => Some(path),
        _ => None,
    }
}

/// The SDK's reading of `format`: a JSON field when its type is `json`, else the whole file. A
/// format without a type is refused as `unusable`, as the SDK refuses to build it.
fn format(
    format: Option<&SourceFormat>,
    unusable: PythonException,
) -> Result<Format, GoogleLoginFailure> {
    match format {
        None => Ok(Format::Text),
        Some(SourceFormat { kind: None, .. }) => {
            Err(GoogleLoginFailure::new(unusable, UNLOADABLE_CREDENTIALS))
        }
        Some(SourceFormat {
            kind: Some(kind),
            subject_token_field_name,
        }) => Ok(match kind.as_str() {
            JSON_FORMAT => Format::Json(subject_token_field_name.clone().unwrap_or_default()),
            _ => Format::Text,
        }),
    }
}

/// The external account at `path`, refused as `unusable` when it cannot be built.
pub(crate) fn load(
    info: &ExternalAccountInfo,
    path: &str,
    environment: &Arc<dyn GoogleEnvironment>,
    unusable: PythonException,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    let subject = SubjectFile {
        path: path.to_owned(),
        format: format(
            info.credential_source
                .as_ref()
                .and_then(|source| source.format.as_ref()),
            unusable,
        )?,
    };
    exchanged(
        info,
        programmatic(info, subject, unusable)?,
        environment,
        unusable,
    )
}

fn refresh_error(message: &str) -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::Refresh, message)
}

impl SubjectFile {
    /// The file's subject token, as `identity_pool._parse_token_data` takes it from the file's
    /// text: an empty token is no token.
    async fn read(&self) -> Result<String, GoogleLoginFailure> {
        let bytes = read_bounded_off_runtime(self.path.clone())
            .await
            .map_err(|read| {
                refresh_error(match read {
                    BoundedRead::Unreadable(_) => UNREADABLE,
                    BoundedRead::TooLarge => TOO_LARGE,
                })
            })?;
        let text = String::from_utf8(bytes).map_err(|_| refresh_error(NOT_TEXT))?;
        match &self.format {
            Format::Text => Some(text),
            Format::Json(field) => {
                serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&text)
                    .ok()
                    .and_then(|object| object.get(field)?.as_str().map(str::to_owned))
            }
        }
        .filter(|token| !token.is_empty())
        .ok_or_else(|| refresh_error(NO_TOKEN))
    }
}

impl SubjectTokenProvider for SubjectFile {
    type Error = GoogleLoginFailure;

    async fn subject_token(&self) -> Result<SubjectToken, GoogleLoginFailure> {
        self.read()
            .await
            .map(|token| subject_token::Builder::new(token).build())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::google::credentials::MAX_CREDENTIALS_FILE_BYTES;

    fn info(source: serde_json::Value) -> ExternalAccountInfo {
        serde_json::from_value(json!({"credential_source": source})).unwrap()
    }

    #[rstest]
    #[case::file(info(json!({"file": "/t"})), Some("/t"))]
    #[case::url(info(json!({"url": "https://example.com", "file": "/t"})), None)]
    #[case::aws(info(json!({"environment_id": "aws1", "file": "/t"})), None)]
    #[case::executable(info(json!({"executable": {"command": "/bin/true"}})), None)]
    fn only_a_file_source_is_read_here(
        #[case] info: ExternalAccountInfo,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(subject_file(&info), expected);
    }

    #[rstest]
    #[case::text(None, b"file-subject".to_vec(), Ok("file-subject"))]
    #[case::text_kept_whole(Some(json!({"type": "text"})), b" subject\n".to_vec(), Ok(" subject\n"))]
    #[case::unknown_type_is_text(Some(json!({"type": "xml"})), b"<subject/>".to_vec(), Ok("<subject/>"))]
    #[case::json_field(Some(json!({"type": "json", "subject_token_field_name": "token"})), br#"{"token": "json-subject"}"#.to_vec(), Ok("json-subject"))]
    #[case::json_missing_field(Some(json!({"type": "json", "subject_token_field_name": "token"})), br#"{"other": "secret-subject"}"#.to_vec(), Err(NO_TOKEN))]
    #[case::json_field_not_text(Some(json!({"type": "json", "subject_token_field_name": "token"})), br#"{"token": 1}"#.to_vec(), Err(NO_TOKEN))]
    #[case::json_without_a_field_name(Some(json!({"type": "json"})), br#"{"token": "secret-subject"}"#.to_vec(), Err(NO_TOKEN))]
    #[case::not_json(Some(json!({"type": "json", "subject_token_field_name": "token"})), b"secret-subject".to_vec(), Err(NO_TOKEN))]
    #[case::empty(None, Vec::new(), Err(NO_TOKEN))]
    #[case::not_utf8(None, vec![0xff, 0xfe], Err(NOT_TEXT))]
    #[case::oversize(None, vec![b'a'; MAX_CREDENTIALS_FILE_BYTES as usize + 1], Err(TOO_LARGE))]
    #[tokio::test]
    async fn the_subject_token_is_read_bounded_as_the_sdk_reads_it(
        #[case] format_source: Option<serde_json::Value>,
        #[case] contents: Vec<u8>,
        #[case] expected: Result<&str, &str>,
    ) {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), contents).unwrap();
        let account = info(json!({"file": file.path(), "format": format_source}));
        let subject = SubjectFile {
            path: file.path().to_str().unwrap().to_owned(),
            format: format(
                account.credential_source.as_ref().unwrap().format.as_ref(),
                PythonException::Malformed,
            )
            .unwrap(),
        };

        let read = subject.read().await;

        assert_eq!(
            read.as_deref()
                .map_err(|failure| (failure.exception, failure.message.as_str())),
            expected.map_err(|message| (PythonException::Refresh, message))
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_missing_file_is_a_refresh_error() {
        let subject = SubjectFile {
            path: "/nonexistent/subject".into(),
            format: Format::Text,
        };

        let refused = subject.read().await.unwrap_err();

        assert_eq!(refused, refresh_error(UNREADABLE));
    }

    #[rstest]
    fn a_format_without_a_type_is_refused_as_the_sdk_refuses_it() {
        let refused = format(
            Some(&SourceFormat::default()),
            PythonException::DefaultCredentials,
        )
        .unwrap_err();

        assert_eq!(
            refused,
            GoogleLoginFailure::new(PythonException::DefaultCredentials, UNLOADABLE_CREDENTIALS)
        );
    }
}
