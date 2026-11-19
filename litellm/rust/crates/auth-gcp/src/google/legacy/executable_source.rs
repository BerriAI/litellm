//! Backwards compatibility only, with Python google-auth's `pluggable.Credentials`: an external
//! account whose `credential_source` runs a credential helper executable. Google's Rust SDK
//! (`google_cloud_auth` 1.16) runs one too, but splits its command on whitespace and passes none
//! of the `GOOGLE_EXTERNAL_ACCOUNT_*` variables, so a helper that quotes an argument or reads its
//! audience from the environment fails. This module runs the helper as `pluggable.py` does, with
//! the `shlex` crate standing in for `shlex.split`, and hands its subject token to the SDK, which
//! makes the STS exchange and any impersonation. Delete it once the SDK splits the command
//! shell-style and sets those variables.

use std::{ops::RangeInclusive, process::Stdio, sync::Arc, time::Duration};

use google_cloud_auth::credentials::subject_token::{self, SubjectToken, SubjectTokenProvider};
use serde::Deserialize;

use crate::{
    GcpTokenSource,
    google::{
        builders::{exchanged, programmatic},
        credentials::{
            BoundedRead, ExecutableSource, ExternalAccountInfo, read_bounded_off_runtime,
            target_principal,
        },
        environment::GoogleEnvironment,
        failure::{GoogleLoginFailure, PythonException},
    },
};

const ALLOW_EXECUTABLES: &str = "GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES";
const EXECUTABLES_NOT_ALLOWED: &str = "Executables need to be explicitly allowed (set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES to '1') to run.";
const DEFAULT_TIMEOUT_MILLIS: u64 = 30_000;
const TIMEOUT_MILLIS: RangeInclusive<u64> = 5_000..=120_000;
const INTERACTIVE_TIMEOUT_MILLIS: RangeInclusive<u64> = 30_000..=1_800_000;
const SUPPORTED_VERSION: u64 = 1;
const JWT: &str = "urn:ietf:params:oauth:token-type:jwt";
const ID_TOKEN: &str = "urn:ietf:params:oauth:token-type:id_token";
const SAML2: &str = "urn:ietf:params:oauth:token-type:saml2";

/// A credential helper's answer, as `pluggable._parse_subject_token` reads it. The error `code`
/// and `message` only count by their presence, so nothing the helper prints reaches an error.
#[derive(Deserialize)]
struct HelperResponse {
    version: Option<u64>,
    success: Option<bool>,
    code: Option<serde::de::IgnoredAny>,
    message: Option<serde::de::IgnoredAny>,
    expiration_time: Option<i64>,
    token_type: Option<String>,
    id_token: Option<String>,
    saml_response: Option<String>,
}

fn failure(exception: PythonException, message: &str) -> GoogleLoginFailure {
    GoogleLoginFailure::new(exception, message)
}

impl HelperResponse {
    /// A helper's answer: text that is not JSON is a `ValueError`, JSON that is not an answer a
    /// `MalformedError`.
    fn parse(text: &[u8]) -> Result<Self, GoogleLoginFailure> {
        serde_json::from_slice(text).map_err(|error| match error.classify() {
            serde_json::error::Category::Data => failure(
                PythonException::Malformed,
                "The credential helper's response is not a JSON object of the expected shape.",
            ),
            _ => failure(
                PythonException::Value,
                "The credential helper's response is not JSON.",
            ),
        })
    }

    /// `pluggable._parse_subject_token` at `now`, in Unix seconds.
    fn subject_token(&self, now: i64) -> Result<String, GoogleLoginFailure> {
        let refresh = |message| failure(PythonException::Refresh, message);
        let malformed = |message| failure(PythonException::Malformed, message);
        let version = self
            .version
            .ok_or_else(|| malformed("The credential helper's response has no version."))?;
        if version > SUPPORTED_VERSION {
            return Err(refresh(
                "The credential helper answered with an unsupported version.",
            ));
        }
        match self.success {
            None => {
                return Err(malformed(
                    "The credential helper's response has no success.",
                ));
            }
            Some(false) if self.code.is_none() || self.message.is_none() => {
                return Err(malformed(
                    "The credential helper failed without an error code and message.",
                ));
            }
            Some(false) => return Err(refresh("The credential helper reported a failure.")),
            Some(true) => {}
        }
        if self.expiration_time.is_some_and(|expiry| expiry < now) {
            return Err(refresh("The credential helper's token has expired."));
        }
        let token = match self.token_type.as_deref() {
            None => {
                return Err(malformed(
                    "The credential helper's response has no token_type.",
                ));
            }
            Some(JWT | ID_TOKEN) => &self.id_token,
            Some(SAML2) => &self.saml_response,
            Some(_) => {
                return Err(refresh(
                    "The credential helper answered with an unsupported token_type.",
                ));
            }
        };
        token
            .clone()
            .ok_or_else(|| refresh("The credential helper's response holds no token."))
    }
}

/// A credential helper as `pluggable.Credentials` runs it, never interactively.
#[derive(veil::Redact)]
struct Helper {
    #[redact(with = "[REDACTED]")]
    command: String,
    timeout: Duration,
    output_file: Option<String>,
    audience: String,
    subject_token_type: String,
    impersonated_email: Option<String>,
}

fn unix_now() -> i64 {
    time::OffsetDateTime::now_utc().unix_timestamp()
}

/// `pluggable.Credentials.__init__`'s checks of the helper's configuration, refused as
/// `explicit` credentials raise them or as application default credentials do.
fn helper(
    info: &ExternalAccountInfo,
    executable: &ExecutableSource,
    explicit: bool,
) -> Result<Helper, GoogleLoginFailure> {
    let refused = |exception, message| {
        failure(
            match explicit {
                true => exception,
                false => PythonException::DefaultCredentials,
            },
            message,
        )
    };
    let command = executable
        .command
        .clone()
        .filter(|command| !command.is_empty())
        .ok_or_else(|| {
            refused(
                PythonException::Malformed,
                "The credential_source executable has no command.",
            )
        })?;
    let timeout_millis = executable
        .timeout_millis
        .filter(|millis| *millis != 0)
        .unwrap_or(DEFAULT_TIMEOUT_MILLIS);
    if !TIMEOUT_MILLIS.contains(&timeout_millis) {
        return Err(refused(
            PythonException::InvalidValue,
            "The credential_source executable timeout_millis must be between 5000 and 120000.",
        ));
    }
    if executable
        .interactive_timeout_millis
        .is_some_and(|millis| millis != 0 && !INTERACTIVE_TIMEOUT_MILLIS.contains(&millis))
    {
        return Err(refused(
            PythonException::InvalidValue,
            "The credential_source executable interactive_timeout_millis must be between 30000 and 1800000.",
        ));
    }
    Ok(Helper {
        command,
        timeout: Duration::from_millis(timeout_millis),
        output_file: executable.output_file.clone(),
        audience: info.audience.clone().unwrap_or_default(),
        subject_token_type: info.subject_token_type.clone().unwrap_or_default(),
        impersonated_email: info
            .service_account_impersonation_url
            .as_deref()
            .and_then(target_principal)
            .map(str::to_owned),
    })
}

/// `pluggable.Credentials`, with the STS exchange and impersonation left to Google's Rust SDK.
/// Executables run only when `GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES` is `1`.
pub(super) fn load(
    info: &ExternalAccountInfo,
    executable: &ExecutableSource,
    environment: &Arc<dyn GoogleEnvironment>,
    explicit: bool,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    let helper = helper(info, executable, explicit)?;
    if environment.os_env(ALLOW_EXECUTABLES).as_deref() != Some("1") {
        return Err(failure(PythonException::Malformed, EXECUTABLES_NOT_ALLOWED));
    }
    let unusable = match explicit {
        true => PythonException::Malformed,
        false => PythonException::DefaultCredentials,
    };
    exchanged(
        info,
        programmatic(info, helper, unusable)?,
        environment,
        unusable,
    )
}

impl Helper {
    /// The token a still-valid answer in `output_file` holds. A file that cannot be read or is
    /// over the credentials file limit is no cached answer; an answer that is malformed is
    /// refused.
    async fn cached(&self) -> Result<Option<String>, GoogleLoginFailure> {
        let Some(path) = &self.output_file else {
            return Ok(None);
        };
        let Ok(content) = read_bounded_off_runtime(path.clone()).await else {
            return Ok(None);
        };
        let response = match HelperResponse::parse(&content) {
            Ok(response) => response,
            Err(failure) if failure.exception == PythonException::Malformed => return Err(failure),
            Err(_) => return Ok(None),
        };
        match response.subject_token(unix_now()) {
            Ok(token) if response.expiration_time.is_some() => Ok(Some(token)),
            Ok(_) => Ok(None),
            Err(failure) if failure.exception == PythonException::Malformed => Err(failure),
            Err(_) => Ok(None),
        }
    }

    fn command(&self) -> Result<tokio::process::Command, GoogleLoginFailure> {
        let unusable = |message| failure(PythonException::Value, message);
        let words = shlex::split(&self.command).ok_or_else(|| {
            unusable("The credential_source executable command has an unclosed quote.")
        })?;
        let (program, arguments) = words.split_first().ok_or_else(|| {
            unusable("The credential_source executable command names no program.")
        })?;
        let mut command = tokio::process::Command::new(program);
        command
            .args(arguments)
            .env("GOOGLE_EXTERNAL_ACCOUNT_AUDIENCE", &self.audience)
            .env(
                "GOOGLE_EXTERNAL_ACCOUNT_TOKEN_TYPE",
                &self.subject_token_type,
            )
            .env(
                "GOOGLE_EXTERNAL_ACCOUNT_ID",
                self.impersonated_email.as_deref().unwrap_or_default(),
            )
            .env("GOOGLE_EXTERNAL_ACCOUNT_INTERACTIVE", "0")
            .env("GOOGLE_EXTERNAL_ACCOUNT_REVOKE", "0")
            .envs(
                self.impersonated_email
                    .iter()
                    .map(|email| ("GOOGLE_EXTERNAL_ACCOUNT_IMPERSONATED_EMAIL", email)),
            )
            .envs(
                self.output_file
                    .iter()
                    .map(|path| ("GOOGLE_EXTERNAL_ACCOUNT_OUTPUT_FILE", path)),
            )
            .stdin(Stdio::null())
            .kill_on_drop(true);
        Ok(command)
    }

    /// `pluggable.Credentials.retrieve_subject_token`: a cached answer, else the helper's own,
    /// read from `output_file` when it prints none.
    async fn run(&self) -> Result<String, GoogleLoginFailure> {
        if let Some(token) = self.cached().await? {
            return Ok(token);
        }
        let output = tokio::time::timeout(self.timeout, self.command()?.output())
            .await
            .map_err(|_| {
                failure(
                    PythonException::Refresh,
                    "The credential_source executable did not finish within its timeout.",
                )
            })?
            .map_err(|error| {
                GoogleLoginFailure::new(
                    PythonException::OS,
                    format!("The credential_source executable cannot be run: {error}"),
                )
            })?;
        if !output.status.success() {
            return Err(GoogleLoginFailure::new(
                PythonException::Refresh,
                format!(
                    "The credential_source executable exited with code {}.",
                    output.status.code().unwrap_or(-1),
                ),
            ));
        }
        let printed = [output.stdout, output.stderr].concat();
        let response = match (printed.is_empty(), &self.output_file) {
            (false, _) => HelperResponse::parse(&printed)?,
            (true, Some(path)) => {
                HelperResponse::parse(&read_bounded_off_runtime(path.clone()).await.map_err(|read| {
                    match read {
                        BoundedRead::Unreadable(_) => failure(
                            PythonException::OS,
                            "The credential_source executable's output_file cannot be read.",
                        ),
                        BoundedRead::TooLarge => failure(
                            PythonException::Refresh,
                            "The credential_source executable's output_file is larger than any credential helper answer.",
                        ),
                    }
                })?)?
            }
            (true, None) => {
                return Err(failure(
                    PythonException::Malformed,
                    "The credential_source executable printed no response.",
                ));
            }
        };
        response.subject_token(unix_now())
    }
}

impl SubjectTokenProvider for Helper {
    type Error = GoogleLoginFailure;

    async fn subject_token(&self) -> Result<SubjectToken, GoogleLoginFailure> {
        self.run()
            .await
            .map(|token| subject_token::Builder::new(token).build())
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;
    use crate::google::credentials::MAX_CREDENTIALS_FILE_BYTES;

    const AUDIENCE: &str =
        "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/q";
    const IMPERSONATION_URL: &str = "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken";
    /// Prints the variables a helper is given and its first argument as its token.
    const ECHO_ENVIRONMENT: &str = r#"printf '{"version": 1, "success": true, "token_type": "urn:ietf:params:oauth:token-type:jwt", "id_token": "%s|%s|%s|%s|%s|%s|%s|%s"}' "$GOOGLE_EXTERNAL_ACCOUNT_AUDIENCE" "$GOOGLE_EXTERNAL_ACCOUNT_TOKEN_TYPE" "$GOOGLE_EXTERNAL_ACCOUNT_ID" "$GOOGLE_EXTERNAL_ACCOUNT_INTERACTIVE" "$GOOGLE_EXTERNAL_ACCOUNT_REVOKE" "${GOOGLE_EXTERNAL_ACCOUNT_IMPERSONATED_EMAIL-unset}" "${GOOGLE_EXTERNAL_ACCOUNT_OUTPUT_FILE-unset}" "${1-none}""#;
    const PRINT_FRESH: &str = r#"echo '{"version": 1, "success": true, "token_type": "urn:ietf:params:oauth:token-type:jwt", "id_token": "fresh"}'"#;

    /// A scratch directory and the helper script written into it.
    struct Scratch {
        directory: tempfile::TempDir,
    }

    impl Scratch {
        fn script(&self, body: &str) -> PathBuf {
            let script = self.directory.path().join("helper.sh");
            std::fs::write(&script, format!("#!/bin/sh\n{body}\n")).unwrap();
            std::fs::set_permissions(
                &script,
                <std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o755),
            )
            .unwrap();
            script
        }

        fn output_file(&self) -> PathBuf {
            self.directory.path().join("cached.json")
        }
    }

    #[fixture]
    fn scratch() -> Scratch {
        Scratch {
            directory: tempfile::tempdir().unwrap(),
        }
    }

    fn info(impersonated: bool) -> ExternalAccountInfo {
        serde_json::from_value(json!({
            "audience": AUDIENCE,
            "subject_token_type": JWT,
            "service_account_impersonation_url": impersonated.then_some(IMPERSONATION_URL),
        }))
        .unwrap()
    }

    fn source(command: String, output_file: Option<&Path>) -> ExecutableSource {
        ExecutableSource {
            command: Some(command),
            output_file: output_file.map(|path| path.to_str().unwrap().to_owned()),
            ..ExecutableSource::default()
        }
    }

    /// The command line that runs the script at `path`, through `sh` so that no test execs a
    /// file another test's fork may still hold open for writing.
    fn quoted(path: &Path) -> String {
        format!("/bin/sh '{}'", path.display())
    }

    #[rstest]
    #[case::plain(false, false, "", "|0|0|unset|unset|none")]
    #[case::impersonating(
        true,
        false,
        "",
        "sa@p.iam.gserviceaccount.com|0|0|sa@p.iam.gserviceaccount.com|unset|none"
    )]
    #[case::with_output_file(false, true, "", "|0|0|unset|{output}|none")]
    #[case::single_quoted_argument(
        false,
        false,
        " 'two words' other",
        "|0|0|unset|unset|two words"
    )]
    #[case::double_quoted_argument(
        false,
        false,
        r#" "two  words" other"#,
        "|0|0|unset|unset|two  words"
    )]
    #[case::escaped_space(false, false, r" two\ words", "|0|0|unset|unset|two words")]
    #[tokio::test]
    async fn the_helper_gets_pluggable_s_environment_and_shell_split_arguments(
        scratch: Scratch,
        #[case] impersonated: bool,
        #[case] output_file: bool,
        #[case] arguments: &str,
        #[case] expected: &str,
    ) {
        let script = scratch.script(ECHO_ENVIRONMENT);
        let output = scratch.output_file();
        let helper = helper(
            &info(impersonated),
            &source(
                format!("{}{arguments}", quoted(&script)),
                output_file.then_some(output.as_path()),
            ),
            true,
        )
        .unwrap();

        let token = helper.run().await.unwrap();

        assert_eq!(
            token,
            format!("{AUDIENCE}|{JWT}|{expected}").replace("{output}", output.to_str().unwrap())
        );
    }

    fn cached_answer(extra: serde_json::Value) -> serde_json::Value {
        json!({"version": 1, "success": true, "token_type": JWT, "id_token": "cached"})
            .as_object()
            .unwrap()
            .clone()
            .into_iter()
            .chain(extra.as_object().unwrap().clone())
            .collect()
    }

    #[rstest]
    #[case::still_valid(json!({"expiration_time": unix_now() + 3600}), "cached")]
    #[case::expired(json!({"expiration_time": unix_now() - 10}), "fresh")]
    #[case::without_expiration(json!({}), "fresh")]
    #[case::unsuccessful(json!({"success": false, "code": "1", "message": "m", "expiration_time": unix_now() + 3600}), "fresh")]
    #[tokio::test]
    async fn a_still_valid_cached_answer_is_used_instead_of_running_the_helper(
        scratch: Scratch,
        #[case] extra: serde_json::Value,
        #[case] expected: &str,
    ) {
        let script = scratch.script(PRINT_FRESH);
        let output = scratch.output_file();
        std::fs::write(&output, cached_answer(extra).to_string()).unwrap();
        let helper = helper(&info(false), &source(quoted(&script), Some(&output)), true).unwrap();

        assert_eq!(helper.run().await.unwrap(), expected);
    }

    #[rstest]
    #[case::helper_prints_nothing("true", Err(PythonException::Refresh))]
    #[case::helper_prints_an_answer(PRINT_FRESH, Ok("fresh"))]
    #[tokio::test]
    async fn an_oversize_output_file_is_never_read_as_an_answer(
        scratch: Scratch,
        #[case] body: &str,
        #[case] expected: Result<&str, PythonException>,
    ) {
        let script = scratch.script(body);
        let output = scratch.output_file();
        std::fs::write(&output, vec![b' '; MAX_CREDENTIALS_FILE_BYTES as usize + 1]).unwrap();
        let helper = helper(&info(false), &source(quoted(&script), Some(&output)), true).unwrap();

        let outcome = helper.run().await;

        assert_eq!(
            outcome.as_deref().map_err(|failure| failure.exception),
            expected
        );
    }

    #[rstest]
    #[case::non_zero_exit(
        "echo secret-subject; echo secret-error >&2; exit 3",
        PythonException::Refresh,
        "The credential_source executable exited with code 3."
    )]
    #[case::timeout(
        "sleep 30",
        PythonException::Refresh,
        "The credential_source executable did not finish within its timeout."
    )]
    #[case::unsuccessful(r#"echo '{"version": 1, "success": false, "code": "secret-code", "message": "secret-message"}'"#, PythonException::Refresh, "The credential helper reported a failure.")]
    #[case::unsuccessful_without_a_code(
        r#"echo '{"version": 1, "success": false}'"#,
        PythonException::Malformed,
        "The credential helper failed without an error code and message."
    )]
    #[case::not_json(
        "echo secret-subject",
        PythonException::Value,
        "The credential helper's response is not JSON."
    )]
    #[case::no_version(
        r#"echo '{"success": true, "id_token": "secret-subject"}'"#,
        PythonException::Malformed,
        "The credential helper's response has no version."
    )]
    #[case::newer_version(
        r#"echo '{"version": 2, "success": true}'"#,
        PythonException::Refresh,
        "The credential helper answered with an unsupported version."
    )]
    #[case::expired(r#"echo '{"version": 1, "success": true, "token_type": "urn:ietf:params:oauth:token-type:jwt", "id_token": "secret-subject", "expiration_time": 1}'"#, PythonException::Refresh, "The credential helper's token has expired.")]
    #[case::unsupported_token_type(
        r#"echo '{"version": 1, "success": true, "token_type": "secret-type"}'"#,
        PythonException::Refresh,
        "The credential helper answered with an unsupported token_type."
    )]
    #[case::saml_without_a_response(r#"echo '{"version": 1, "success": true, "token_type": "urn:ietf:params:oauth:token-type:saml2", "id_token": "secret-subject"}'"#, PythonException::Refresh, "The credential helper's response holds no token.")]
    #[case::printed_nothing(
        "true",
        PythonException::Malformed,
        "The credential_source executable printed no response."
    )]
    #[tokio::test]
    async fn a_failed_helper_is_refused_without_its_output(
        scratch: Scratch,
        #[case] body: &str,
        #[case] exception: PythonException,
        #[case] message: &str,
    ) {
        let script = scratch.script(body);
        let helper = Helper {
            timeout: Duration::from_secs(2),
            ..helper(&info(false), &source(quoted(&script), None), true).unwrap()
        };

        let refused = helper.run().await.unwrap_err();

        assert_eq!(
            (refused.exception, refused.message.as_str()),
            (exception, message)
        );
    }

    #[rstest]
    #[case::unclosed_quote(
        "'/bin/true",
        "The credential_source executable command has an unclosed quote."
    )]
    #[case::blank("   ", "The credential_source executable command names no program.")]
    #[tokio::test]
    async fn an_unsplittable_command_is_a_value_error(
        #[case] command: &str,
        #[case] message: &str,
    ) {
        let helper = helper(&info(false), &source(command.into(), None), true).unwrap();

        let refused = helper.run().await.unwrap_err();

        assert_eq!(
            (refused.exception, refused.message.as_str()),
            (PythonException::Value, message)
        );
    }

    #[rstest]
    #[case::no_command(None, None, None, true, Some(PythonException::Malformed))]
    #[case::empty_command(Some(""), None, None, true, Some(PythonException::Malformed))]
    #[case::default_timeout(Some("/bin/true"), None, None, true, None)]
    #[case::zero_timeout_is_the_default(Some("/bin/true"), Some(0), None, true, None)]
    #[case::short_timeout(
        Some("/bin/true"),
        Some(4_999),
        None,
        true,
        Some(PythonException::InvalidValue)
    )]
    #[case::long_timeout(
        Some("/bin/true"),
        Some(120_001),
        None,
        true,
        Some(PythonException::InvalidValue)
    )]
    #[case::short_interactive_timeout(
        Some("/bin/true"),
        None,
        Some(29_999),
        true,
        Some(PythonException::InvalidValue)
    )]
    #[case::application_default(
        Some("/bin/true"),
        Some(1),
        None,
        false,
        Some(PythonException::DefaultCredentials)
    )]
    fn the_configuration_is_checked_as_pluggable_checks_it(
        #[case] command: Option<&str>,
        #[case] timeout_millis: Option<u64>,
        #[case] interactive_timeout_millis: Option<u64>,
        #[case] explicit: bool,
        #[case] refused: Option<PythonException>,
    ) {
        let executable = ExecutableSource {
            command: command.map(str::to_owned),
            timeout_millis,
            interactive_timeout_millis,
            output_file: None,
        };

        let checked = helper(&info(false), &executable, explicit);

        assert_eq!(checked.err().map(|failure| failure.exception), refused);
    }
}
