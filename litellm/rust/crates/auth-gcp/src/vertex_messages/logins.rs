//! `VertexBase`'s credential cache for the Vertex AI Messages login: credentials loaded once per
//! (identity, project), a token reused until google-auth would call it stale, a stale token
//! served while one background refresh replaces it, and one blocking refresh at a time per
//! login. The identity of a key file includes its inode, mtime and size, so a rotated file loads
//! anew and a replacement that does not load keeps the credentials loaded before it.

use std::{
    future::Future,
    os::unix::fs::MetadataExt,
    sync::{Arc, Mutex, PoisonError},
};

use litellm_secret_value::SecretValue;
use moka::future::Cache;
use sha2::{Digest, Sha256};
use time::OffsetDateTime;

use super::{
    credentials::{Expiry, Loaded, Minted, TokenState, refresh_rejected},
    failure::{GoogleLoginFailure, PythonException},
};

const MAX_CACHED_LOGINS: usize = 64;

/// `vertex_credentials_file_version`: `os.stat`'s inode, mtime in nanoseconds and size.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct FileVersion {
    inode: u64,
    modified: i128,
    size: u64,
}

pub(crate) fn file_version(path: &str) -> Option<FileVersion> {
    let metadata = std::fs::metadata(path).ok()?;
    Some(FileVersion {
        inode: metadata.ino(),
        modified: i128::from(metadata.mtime()) * 1_000_000_000 + i128::from(metadata.mtime_nsec()),
        size: metadata.size(),
    })
}

/// `_credential_cache_identity`: the credentials text, with the version of the key file it names
/// when it names one. Application default credentials are known only by the version of the
/// `GOOGLE_APPLICATION_CREDENTIALS` file.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(crate) struct Identity {
    text: Option<[u8; 32]>,
    version: Option<FileVersion>,
}

pub(crate) fn identity(
    credentials: Option<&str>,
    application_credentials: Option<&str>,
) -> Option<Identity> {
    match credentials {
        Some(text) => Some(Identity {
            text: Some(Sha256::digest(text).into()),
            version: file_version(text),
        }),
        None => Some(Identity {
            text: None,
            version: Some(file_version(application_credentials?)?),
        }),
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) struct LoginKey {
    pub(crate) identity: Option<Identity>,
    pub(crate) project: Option<String>,
}

impl LoginKey {
    /// `_is_other_version_of_key_file`.
    fn is_other_version_of(&self, key: &Self) -> bool {
        let (Some(candidate), Some(identity)) = (self.identity, key.identity) else {
            return false;
        };
        candidate != identity
            && candidate.version.is_some()
            && candidate.text == identity.text
            && self.project == key.project
    }
}

/// Loaded credentials, the project they resolved and the token they minted last.
pub(crate) struct Login {
    credentials: Loaded,
    project_id: String,
    token: Mutex<Minted>,
    refreshing: Arc<tokio::sync::Mutex<()>>,
}

impl Login {
    pub(crate) fn new(credentials: Loaded, project_id: String, token: Minted) -> Arc<Self> {
        Arc::new(Self {
            credentials,
            project_id,
            token: Mutex::new(token),
            refreshing: Arc::default(),
        })
    }

    fn current(&self) -> Minted {
        self.token
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn replace(&self, token: Minted) {
        *self.token.lock().unwrap_or_else(PoisonError::into_inner) = token;
    }

    /// `_schedule_background_refresh`: one refresh at a time, its failure left for the next
    /// call that finds the token expired.
    fn refresh_in_background(self: &Arc<Self>) {
        let Ok(refreshing) = self.refreshing.clone().try_lock_owned() else {
            return;
        };
        let login = self.clone();
        tokio::spawn(async move {
            if let Ok(token) = login.credentials.refresh().await {
                login.replace(token);
            }
            drop(refreshing);
        });
    }
}

/// `VertexBase._token_fetch_failed`.
fn token_fetch_failed(failure: &GoogleLoginFailure) -> bool {
    matches!(
        failure.exception,
        PythonException::Refresh | PythonException::Transport
    )
}

pub(crate) struct Logins {
    cache: Cache<LoginKey, Arc<Login>>,
    /// The keys most recently used last, which the earlier key file is looked up in.
    recent: Mutex<Vec<LoginKey>>,
}

impl Default for Logins {
    fn default() -> Self {
        Self {
            cache: Cache::builder()
                .max_capacity(MAX_CACHED_LOGINS as u64)
                .build(),
            recent: Mutex::default(),
        }
    }
}

impl Logins {
    fn touch(&self, key: &LoginKey) {
        let mut recent = self.recent.lock().unwrap_or_else(PoisonError::into_inner);
        recent.retain(|known| known != key);
        recent.push(key.clone());
        let excess = recent.len().saturating_sub(MAX_CACHED_LOGINS);
        recent.drain(..excess);
    }

    /// `_credentials_from_an_earlier_key_file`.
    async fn earlier_version(
        &self,
        key: &LoginKey,
        failure: &GoogleLoginFailure,
    ) -> Option<Arc<Login>> {
        let earlier = self
            .recent
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .rev()
            .find(|candidate| candidate.is_other_version_of(key))
            .cloned()?;
        let login = self.cache.get(&earlier).await?;
        tracing::warn!(
            "Vertex: the replaced credentials file did not load, so the credentials loaded before it stay in use: {}",
            failure.message
        );
        Some(login)
    }

    /// `_load_auth_or_earlier_key_file` and `_cache_credentials`, one load at a time per key.
    async fn get_or_load<F, Fut>(
        &self,
        key: &LoginKey,
        load: &F,
    ) -> Result<Arc<Login>, GoogleLoginFailure>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<Arc<Login>, GoogleLoginFailure>>,
    {
        let login = self
            .cache
            .try_get_with(key.clone(), async {
                let failure = match load().await {
                    Ok(login) => return Ok(login),
                    Err(failure) => failure,
                };
                if !token_fetch_failed(&failure)
                    && let Some(login) = self.earlier_version(key, &failure).await
                {
                    return Ok(login);
                }
                tracing::error!(
                    "Failed to load vertex credentials. Check to see if credentials containing partial/invalid information. Error: {}",
                    failure.message
                );
                Err(failure)
            })
            .await
            .map_err(|failure| (*failure).clone())?;
        self.touch(key);
        if key.project.is_none() {
            let resolved = LoginKey {
                project: Some(login.project_id.clone()),
                ..key.clone()
            };
            self.cache.insert(resolved.clone(), login.clone()).await;
            self.touch(&resolved);
        }
        Ok(login)
    }

    /// `get_access_token_async`: the token to send and the project the URL names.
    pub(crate) async fn token<F, Fut>(
        &self,
        key: LoginKey,
        load: F,
    ) -> Result<(SecretValue, String), GoogleLoginFailure>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<Arc<Login>, GoogleLoginFailure>>,
    {
        let login = self.get_or_load(&key, &load).await?;
        match self.serve(&login).await {
            Ok(token) => Ok((token, login.project_id.clone())),
            Err(failure) if refresh_rejected(&failure) => {
                self.reload(&key, &login, &load, failure).await
            }
            Err(failure) => Err(failure),
        }
    }

    async fn serve(&self, login: &Arc<Login>) -> Result<SecretValue, GoogleLoginFailure> {
        let token = login.current();
        if token.expiry == Expiry::Managed {
            return login.credentials.refresh().await?.usable();
        }
        match token.state(OffsetDateTime::now_utc()) {
            TokenState::Fresh => return token.usable(),
            TokenState::Stale if token.non_string.is_none() => {
                login.refresh_in_background();
                return Ok(token.value);
            }
            TokenState::Stale | TokenState::Invalid => {}
        }
        let _refreshing = login.refreshing.lock().await;
        let token = login.current();
        match token.state(OffsetDateTime::now_utc()) {
            TokenState::Fresh => return token.usable(),
            TokenState::Stale if token.non_string.is_none() => return Ok(token.value),
            TokenState::Stale | TokenState::Invalid => {}
        }
        let refreshed = login.credentials.refresh().await?;
        login.replace(refreshed.clone());
        refreshed.usable()
    }

    /// `_handle_reauthentication_async`: Google refused the cached credentials for good, so they
    /// are loaded once more, and the refusal stands if that fails too.
    async fn reload<F, Fut>(
        &self,
        key: &LoginKey,
        refused: &Login,
        load: &F,
        failure: GoogleLoginFailure,
    ) -> Result<(SecretValue, String), GoogleLoginFailure>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<Arc<Login>, GoogleLoginFailure>>,
    {
        self.cache.invalidate(key).await;
        self.cache
            .invalidate(&LoginKey {
                project: Some(refused.project_id.clone()),
                ..key.clone()
            })
            .await;
        let reloaded = match self.get_or_load(key, load).await {
            Ok(login) => self
                .serve(&login)
                .await
                .map(|token| (token, login.project_id.clone())),
            Err(retry) => Err(retry),
        };
        reloaded.map_err(|retry| {
            tracing::error!(
                "Async reauthentication retry failed for project_id: {}. Original error: {}. Retry error: {}",
                refused.project_id,
                failure.message,
                retry.message
            );
            failure
        })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    use rstest::rstest;

    use super::*;
    use crate::vertex_messages::credentials::{Credentials, RefreshFuture};

    /// Credentials whose refreshes answer with each scripted result in turn, the last one for
    /// good, and count how often they were asked.
    struct Scripted {
        answers: Vec<Result<Minted, GoogleLoginFailure>>,
        refreshes: AtomicUsize,
        delay: Duration,
    }

    impl Credentials for Scripted {
        fn refresh(&self) -> RefreshFuture<'_> {
            Box::pin(async {
                tokio::time::sleep(self.delay).await;
                let index = self.refreshes.fetch_add(1, Ordering::SeqCst);
                self.answers[index.min(self.answers.len() - 1)].clone()
            })
        }
    }

    fn token(value: &str, expires_in: i64) -> Minted {
        Minted::new(
            SecretValue::new(value),
            Expiry::At(OffsetDateTime::now_utc() + time::Duration::seconds(expires_in)),
        )
    }

    fn managed(value: &str) -> Minted {
        Minted::new(SecretValue::new(value), Expiry::Managed)
    }

    fn scripted(answers: Vec<Result<Minted, GoogleLoginFailure>>) -> Arc<Scripted> {
        Arc::new(Scripted {
            answers,
            refreshes: AtomicUsize::new(0),
            delay: Duration::ZERO,
        })
    }

    const FRESH: i64 = 3600;
    const STALE: i64 = 100;
    const EXPIRED: i64 = -1;

    fn key(project: &str) -> LoginKey {
        LoginKey {
            identity: identity(Some("{\"type\": \"service_account\"}"), None),
            project: Some(project.into()),
        }
    }

    /// A load that builds a login from `credentials` holding `first`, the token minted when the
    /// credentials load, and counts how often it ran.
    fn loader(
        credentials: Arc<Scripted>,
        first: Minted,
        loads: Arc<AtomicUsize>,
    ) -> impl Fn() -> std::future::Ready<Result<Arc<Login>, GoogleLoginFailure>> {
        move || {
            loads.fetch_add(1, Ordering::SeqCst);
            std::future::ready(Ok(Login::new(
                credentials.clone(),
                "project".into(),
                first.clone(),
            )))
        }
    }

    async fn served(
        logins: &Logins,
        load: &impl Fn() -> std::future::Ready<Result<Arc<Login>, GoogleLoginFailure>>,
    ) -> String {
        logins
            .token(key("p"), load)
            .await
            .unwrap()
            .0
            .expose()
            .to_owned()
    }

    fn invalid_grant() -> GoogleLoginFailure {
        GoogleLoginFailure::new(
            PythonException::Refresh,
            "('invalid_grant: Invalid JWT Signature.', {'error': 'invalid_grant'})",
        )
    }

    #[tokio::test]
    async fn a_fresh_token_is_reused_without_a_refresh() {
        let credentials = scripted(vec![Ok(token("refreshed", FRESH))]);
        let loads = Arc::new(AtomicUsize::new(0));
        let load = loader(credentials.clone(), token("loaded", FRESH), loads.clone());
        let logins = Logins::default();

        let tokens = (served(&logins, &load).await, served(&logins, &load).await);

        assert_eq!(tokens, ("loaded".to_owned(), "loaded".to_owned()));
        assert_eq!(
            (
                loads.load(Ordering::SeqCst),
                credentials.refreshes.load(Ordering::SeqCst)
            ),
            (1, 0)
        );
    }

    #[tokio::test]
    async fn a_stale_token_is_served_while_one_background_refresh_replaces_it() {
        let credentials = Arc::new(Scripted {
            answers: vec![Ok(token("refreshed", FRESH))],
            refreshes: AtomicUsize::new(0),
            delay: Duration::from_millis(50),
        });
        let load = loader(credentials.clone(), token("stale", STALE), Arc::default());
        let logins = Logins::default();
        let first = served(&logins, &load).await;

        let concurrent =
            futures_util::future::join_all((0..5).map(|_| served(&logins, &load))).await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        let after = served(&logins, &load).await;

        assert_eq!(first, "stale");
        assert!(concurrent.iter().all(|token| token == "stale"));
        assert_eq!(after, "refreshed");
        assert_eq!(credentials.refreshes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn an_expired_token_is_refreshed_once_for_every_caller_waiting_on_it() {
        let credentials = Arc::new(Scripted {
            answers: vec![Ok(token("refreshed", FRESH))],
            refreshes: AtomicUsize::new(0),
            delay: Duration::from_millis(50),
        });
        let load = loader(
            credentials.clone(),
            token("expired", EXPIRED),
            Arc::default(),
        );
        let logins = Logins::default();

        let concurrent =
            futures_util::future::join_all((0..10).map(|_| served(&logins, &load))).await;

        assert!(concurrent.iter().all(|token| token == "refreshed"));
        assert_eq!(credentials.refreshes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn credentials_load_once_for_concurrent_first_callers() {
        let loads = Arc::new(AtomicUsize::new(0));
        let counted = loads.clone();
        let load = move || {
            let counted = counted.clone();
            async move {
                counted.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(Login::new(
                    scripted(vec![]),
                    "project".into(),
                    token("loaded", FRESH),
                ))
            }
        };
        let logins = Logins::default();

        futures_util::future::join_all((0..10).map(|_| logins.token(key("p"), &load))).await;

        assert_eq!(loads.load(Ordering::SeqCst), 1);
    }

    #[rstest]
    #[case::invalid_grant_reloads(invalid_grant(), Ok("reloaded"), 2)]
    #[case::reauthentication_reloads(
        GoogleLoginFailure::new(PythonException::Refresh, "Reauthentication is needed."),
        Ok("reloaded"),
        2
    )]
    #[case::another_refusal_stands(
        GoogleLoginFailure::new(
            PythonException::Refresh,
            "('invalid_client: no', {'error': 'invalid_client'})"
        ),
        Err(PythonException::Refresh),
        1
    )]
    #[tokio::test]
    async fn a_refresh_refused_for_good_loads_the_credentials_once_more(
        #[case] refusal: GoogleLoginFailure,
        #[case] expected: Result<&str, PythonException>,
        #[case] loads_expected: usize,
    ) {
        let loads = Arc::new(AtomicUsize::new(0));
        let counted = loads.clone();
        let load = move || {
            let first = counted.fetch_add(1, Ordering::SeqCst) == 0;
            let login = match first {
                true => Login::new(
                    scripted(vec![Err(refusal.clone())]),
                    "project".into(),
                    token("expired", EXPIRED),
                ),
                false => Login::new(scripted(vec![]), "project".into(), token("reloaded", FRESH)),
            };
            std::future::ready(Ok(login))
        };
        let logins = Logins::default();

        let served = logins.token(key("p"), &load).await;

        assert_eq!(
            served
                .map(|(token, _)| token.expose().to_owned())
                .map_err(|failure| failure.exception),
            expected.map(str::to_owned)
        );
        assert_eq!(loads.load(Ordering::SeqCst), loads_expected);
    }

    #[tokio::test]
    async fn a_reload_that_fails_too_raises_the_first_refusal() {
        let loads = Arc::new(AtomicUsize::new(0));
        let counted = loads.clone();
        let load = move || {
            let login = match counted.fetch_add(1, Ordering::SeqCst) {
                0 => Ok(Login::new(
                    scripted(vec![Err(invalid_grant())]),
                    "project".into(),
                    token("expired", EXPIRED),
                )),
                _ => Err(GoogleLoginFailure::new(
                    PythonException::Value,
                    "the reload failed",
                )),
            };
            std::future::ready(login)
        };

        let failure = Logins::default().token(key("p"), &load).await.unwrap_err();

        assert_eq!(failure, invalid_grant());
    }

    #[tokio::test]
    async fn sdk_managed_credentials_are_asked_on_every_call() {
        let credentials = scripted(vec![Ok(managed("first")), Ok(managed("second"))]);
        let load = loader(credentials.clone(), managed("loaded"), Arc::default());
        let logins = Logins::default();

        let tokens = (served(&logins, &load).await, served(&logins, &load).await);

        assert_eq!(tokens, ("first".to_owned(), "second".to_owned()));
    }

    fn versioned(project: &str, version: u64) -> LoginKey {
        LoginKey {
            identity: Some(Identity {
                text: Some([1; 32]),
                version: Some(FileVersion {
                    inode: version,
                    modified: 0,
                    size: 0,
                }),
            }),
            project: Some(project.into()),
        }
    }

    #[rstest]
    #[case::the_earlier_version(
        versioned("p", 2),
        GoogleLoginFailure::new(PythonException::Value, "bad file"),
        Ok("earlier")
    )]
    #[case::a_token_fetch_failure(
        versioned("p", 2),
        invalid_grant(),
        Err(PythonException::Refresh)
    )]
    #[case::another_project(LoginKey { project: Some("q".into()), ..versioned("p", 2) }, GoogleLoginFailure::new(PythonException::Value, "bad file"), Err(PythonException::Value))]
    #[case::not_a_key_file(LoginKey { identity: Some(Identity { text: Some([1; 32]), version: None }), project: Some("p".into()) }, GoogleLoginFailure::new(PythonException::Value, "bad file"), Ok("earlier"))]
    #[tokio::test]
    async fn a_replaced_key_file_that_does_not_load_keeps_the_earlier_credentials(
        #[case] replaced: LoginKey,
        #[case] failure: GoogleLoginFailure,
        #[case] expected: Result<&str, PythonException>,
    ) {
        let logins = Logins::default();
        let earlier = loader(scripted(vec![]), token("earlier", FRESH), Arc::default());
        logins.token(versioned("p", 1), &earlier).await.unwrap();

        let served = logins
            .token(replaced, || std::future::ready(Err(failure.clone())))
            .await;

        assert_eq!(
            served
                .map(|(token, _)| token.expose().to_owned())
                .map_err(|failure| failure.exception),
            expected.map(str::to_owned)
        );
    }

    #[test]
    fn a_key_file_s_identity_changes_when_the_file_is_rewritten() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        std::fs::write(path, "{}").unwrap();
        let before = identity(Some(path), None);
        std::fs::write(path, "{\"rotated\": true}").unwrap();
        let after = identity(Some(path), None);

        assert_ne!(before, after);
        assert!(before.is_some_and(|identity| identity.version.is_some()));
    }

    #[rstest]
    #[case::inline_json(Some("{\"type\": \"service_account\"}"), None, true, false)]
    #[case::adc_without_a_file(None, None, false, false)]
    #[case::adc_file_missing(None, Some("/nonexistent/adc.json"), false, false)]
    fn an_identity_is_versioned_only_by_a_file_that_exists(
        #[case] credentials: Option<&str>,
        #[case] application_credentials: Option<&str>,
        #[case] known: bool,
        #[case] versioned: bool,
    ) {
        let identity = identity(credentials, application_credentials);
        assert_eq!(
            (
                identity.is_some(),
                identity.is_some_and(|identity| identity.version.is_some())
            ),
            (known, versioned)
        );
    }

    #[test]
    fn application_default_credentials_are_known_by_their_file_s_version() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let path = file.path().to_str().unwrap();
        let identity = identity(None, Some(path)).unwrap();
        assert_eq!(
            (identity.text, identity.version),
            (None, file_version(path))
        );
    }

    #[derive(Clone, Default)]
    struct Recorded(Arc<Mutex<Vec<(litellm_tracing::Level, String, String)>>>);

    impl litellm_tracing::Sink for Recorded {
        fn enabled(&self, _: &litellm_tracing::Metadata<'_>) -> bool {
            true
        }

        fn emit(&self, record: &litellm_tracing::Record) {
            self.0.lock().unwrap().push((
                *record.metadata.level(),
                record.metadata.target().to_owned(),
                record.message.clone(),
            ));
        }
    }

    /// What `future` logged at warning level and above, as the bridge's sink would deliver it
    /// to the `LiteLLM` logger.
    async fn logged<T>(future: impl Future<Output = T>) -> Vec<(litellm_tracing::Level, String)> {
        let recorded = Recorded::default();
        litellm_tracing::Logger::new(recorded.clone())
            .instrument(future)
            .await;
        let records = recorded.0.lock().unwrap().clone();
        records
            .into_iter()
            .filter(|(level, target, _)| {
                *level <= litellm_tracing::Level::WARN && target.starts_with("litellm_")
            })
            .map(|(level, _, message)| (level, message))
            .collect()
    }

    #[rstest]
    #[case::unloadable(
        GoogleLoginFailure::new(PythonException::Value, "Could not resolve project_id"),
        "Could not resolve project_id"
    )]
    #[case::token_fetch_failed(
        invalid_grant(),
        "('invalid_grant: Invalid JWT Signature.', {'error': 'invalid_grant'})"
    )]
    #[tokio::test]
    async fn credentials_that_do_not_load_are_logged_as_an_error(
        #[case] failure: GoogleLoginFailure,
        #[case] error: &str,
    ) {
        let logins = Logins::default();
        let records =
            logged(logins.token(key("p"), || std::future::ready(Err(failure.clone())))).await;

        assert_eq!(
            records,
            [(
                litellm_tracing::Level::ERROR,
                format!(
                    "Failed to load vertex credentials. Check to see if credentials containing partial/invalid information. Error: {error}"
                )
            )]
        );
    }

    #[tokio::test]
    async fn an_earlier_key_file_in_use_is_a_warning_not_an_error() {
        let logins = Logins::default();
        let earlier = loader(scripted(vec![]), token("earlier", FRESH), Arc::default());
        logins.token(versioned("p", 1), &earlier).await.unwrap();

        let records = logged(logins.token(versioned("p", 2), || {
            std::future::ready(Err(GoogleLoginFailure::new(
                PythonException::Value,
                "bad file",
            )))
        }))
        .await;

        assert_eq!(
            records,
            [(
                litellm_tracing::Level::WARN,
                "Vertex: the replaced credentials file did not load, so the credentials loaded before it stay in use: bad file".to_owned()
            )]
        );
    }

    #[tokio::test]
    async fn a_failed_reload_is_logged_with_both_errors() {
        let loads = Arc::new(AtomicUsize::new(0));
        let counted = loads.clone();
        let load = move || {
            let login = match counted.fetch_add(1, Ordering::SeqCst) {
                0 => Ok(Login::new(
                    scripted(vec![Err(invalid_grant())]),
                    "project".into(),
                    token("expired", EXPIRED),
                )),
                _ => Err(GoogleLoginFailure::new(
                    PythonException::Value,
                    "the reload failed",
                )),
            };
            std::future::ready(login)
        };

        let records = logged(Logins::default().token(key("p"), &load)).await;

        assert_eq!(
            records.last().unwrap(),
            &(
                litellm_tracing::Level::ERROR,
                "Async reauthentication retry failed for project_id: project. Original error: ('invalid_grant: Invalid JWT Signature.', {'error': 'invalid_grant'}). Retry error: the reload failed".to_owned()
            )
        );
    }
}
