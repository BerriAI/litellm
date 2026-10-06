use litellm_cache_response::{
    CacheControls, CacheKeyContext, CacheKeyField, CacheKeyInput, CacheKeyParticipation,
    CacheKeyRequest, CacheKeyTransport, get_cache_key, should_use_cache,
};
use rstest::rstest;
use sha2::{Digest, Sha256};

const ENABLED: CacheControls = CacheControls {
    supported_call_type: true,
    configured: true,
    default_on: true,
    caching: None,
    no_cache: false,
    no_store: false,
    use_cache: false,
};

#[rstest]
#[case::enabled(ENABLED, true, true)]
#[case::default_off(CacheControls { default_on: false, ..ENABLED }, false, false)]
#[case::default_off_with_use_cache(
    CacheControls { default_on: false, use_cache: true, ..ENABLED },
    true,
    true
)]
#[case::no_cache(CacheControls { no_cache: true, ..ENABLED }, false, true)]
#[case::no_store(CacheControls { no_store: true, ..ENABLED }, true, false)]
#[case::no_cache_and_no_store(
    CacheControls { no_cache: true, no_store: true, ..ENABLED },
    false,
    false
)]
#[case::caching_disabled(CacheControls { caching: Some(false), ..ENABLED }, false, false)]
#[case::caching_enabled(CacheControls { caching: Some(true), ..ENABLED }, true, true)]
#[case::unsupported_call_type(
    CacheControls { supported_call_type: false, ..ENABLED },
    false,
    false
)]
#[case::unconfigured(CacheControls { configured: false, ..ENABLED }, false, false)]
fn cache_controls_honor_default_modes_and_directives(
    #[case] controls: CacheControls,
    #[case] reads: bool,
    #[case] writes: bool,
) {
    assert_eq!(controls.reads(), reads);
    assert_eq!(controls.writes(), writes);
    assert_eq!(should_use_cache(controls), reads || writes);
}
