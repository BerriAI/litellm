use std::{
    error::Error as StdError,
    future::{Future, poll_fn},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::Poll,
    time::{Duration, SystemTime},
};

use litellm_auth_types::{
    Error, ErrorDetail, ResolvedCredential, SecretValue, TokenProviderHandle,
};
use rstest::rstest;

fn credential(index: usize, access_token: bool) -> ResolvedCredential {
    let token = SecretValue::new(format!("credential-{index}"));
    if access_token {
        return ResolvedCredential::AccessToken {
            token,
            expires_on: Some(SystemTime::UNIX_EPOCH + Duration::from_secs(index as u64)),
        };
    }
    ResolvedCredential::Static(token)
}

#[rstest]
#[case::static_secret(false)]
#[case::access_token(true)]
#[tokio::test]
async fn callbacks_acquire_fresh_credentials_on_demand(#[case] access_token: bool) {
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = calls.clone();
    let provider = TokenProviderHandle::from_callback(move || {
        let index = callback_calls.fetch_add(1, Ordering::SeqCst);
        async move {
            tokio::task::yield_now().await;
            Ok(credential(index, access_token))
        }
    });
    let cloned = provider.clone();

    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        provider.acquire().await.unwrap(),
        credential(0, access_token)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(cloned.acquire().await.unwrap(), credential(1, access_token));
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[rstest]
#[tokio::test]
async fn callback_errors_preserve_the_original_source() {
    let provider = TokenProviderHandle::from_callback(|| async {
        Err(Error::CredentialAcquisition(ErrorDetail::failed(
            "caller credential",
            std::io::Error::from(std::io::ErrorKind::PermissionDenied),
        )))
    });

    let error = provider.acquire().await.unwrap_err();
    assert!(matches!(error, Error::CredentialAcquisition(_)));
    let source = std::iter::successors(Some(&error as &(dyn StdError + 'static)), |error| {
        (*error).source()
    })
    .find_map(|error| error.downcast_ref::<std::io::Error>())
    .unwrap();
    assert_eq!(source.kind(), std::io::ErrorKind::PermissionDenied);
}

struct Release(Arc<AtomicBool>);

impl Drop for Release {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[rstest]
#[tokio::test]
async fn cancelling_acquisition_drops_the_callback_future() {
    let released = Arc::new(AtomicBool::new(false));
    let callback_released = released.clone();
    let provider = TokenProviderHandle::from_callback(move || {
        let released = callback_released.clone();
        async move {
            let _release = Release(released);
            std::future::pending().await
        }
    });

    let mut acquisition = Box::pin(provider.acquire());
    poll_fn(|context| {
        assert!(acquisition.as_mut().poll(context).is_pending());
        assert!(!released.load(Ordering::SeqCst));
        Poll::Ready(())
    })
    .await;
    drop(acquisition);
    assert!(released.load(Ordering::SeqCst));
}
