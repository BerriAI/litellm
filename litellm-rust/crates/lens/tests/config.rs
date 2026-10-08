use litellm_lens::{Error, config::RuntimeThreads};
use rstest::rstest;

#[rstest]
#[case::single_core(None, 1, 2, 4)]
#[case::many_cores(None, 16, 16, 32)]
#[case::empty_override(Some(""), 8, 8, 16)]
#[case::override_below_cores(Some("3"), 16, 3, 6)]
#[case::override_one(Some("1"), 16, 1, 4)]
fn runtime_threads_follow_parallelism_or_override(
    #[case] configured: Option<&str>,
    #[case] parallelism: usize,
    #[case] workers: usize,
    #[case] blocking: usize,
) {
    assert_eq!(
        RuntimeThreads::resolve(configured, parallelism).unwrap(),
        RuntimeThreads { workers, blocking }
    );
}

#[rstest]
#[case::zero("0")]
#[case::negative("-2")]
#[case::text("many")]
fn runtime_threads_reject_invalid_override(#[case] configured: &str) {
    assert!(matches!(
        RuntimeThreads::resolve(Some(configured), 8),
        Err(Error::Configuration("LITELLM_LENS_WORKER_THREADS"))
    ));
}
