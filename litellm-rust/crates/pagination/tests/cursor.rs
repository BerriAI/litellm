use litellm_pagination::{Binding, Cursor, Error, FailureCode, KeyRing, Page, Traversal};
use rstest::{fixture, rstest};
use serde::{Deserialize, Serialize};
use time::{Duration, OffsetDateTime};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct Position {
    start_ms: i64,
    trace_ref: String,
}

#[fixture]
fn now() -> OffsetDateTime {
    OffsetDateTime::from_unix_timestamp(1_790_000_000).unwrap()
}

#[fixture]
fn keys() -> KeyRing {
    KeyRing::new(["primary-secret"]).unwrap()
}

#[fixture]
fn binding() -> Binding {
    Binding::new("traces:list", &("team-a", "user-1"), &(0_i64, 100_i64)).unwrap()
}

fn cursor(now: OffsetDateTime) -> Cursor<Position> {
    Cursor {
        position: Position {
            start_ms: 42,
            trace_ref: "ref-42".into(),
        },
        revision: "rev-1".into(),
        published_ms: 1_790_000_000_000,
        expires_at: now + Duration::minutes(30),
    }
}

#[rstest]
fn cursors_round_trip_their_typed_position_and_pins(
    keys: KeyRing,
    binding: Binding,
    now: OffsetDateTime,
) {
    let token = keys.encode(&binding, &cursor(now)).unwrap();
    assert!(!token.contains("ref-42") && !token.contains("team-a"));
    let decoded: Cursor<Position> = keys.decode(&binding, &token, now).unwrap();
    assert_eq!(decoded, cursor(now));
    assert!(decoded.require_revision("rev-1").is_ok());
    assert!(matches!(
        decoded.require_revision("rev-2"),
        Err(Error::TraversalChanged)
    ));
}

#[rstest]
#[case::other_resource(Binding::new("traces:detail", &("team-a", "user-1"), &(0_i64, 100_i64)).unwrap())]
#[case::other_scope(Binding::new("traces:list", &("team-b", "user-1"), &(0_i64, 100_i64)).unwrap())]
#[case::other_query(Binding::new("traces:list", &("team-a", "user-1"), &(0_i64, 200_i64)).unwrap())]
fn cursors_do_not_continue_a_different_traversal(
    keys: KeyRing,
    binding: Binding,
    now: OffsetDateTime,
    #[case] other: Binding,
) {
    let token = keys.encode(&binding, &cursor(now)).unwrap();
    assert!(matches!(
        keys.decode::<Position>(&other, &token, now),
        Err(Error::InvalidCursor)
    ));
}

#[rstest]
#[case::garbage("not-a-cursor")]
#[case::empty("")]
#[case::missing_tag("eyJ2IjoxfQ")]
#[case::bad_base64("!!!.!!!")]
fn malformed_tokens_are_invalid_cursors(
    keys: KeyRing,
    binding: Binding,
    now: OffsetDateTime,
    #[case] token: &str,
) {
    assert!(matches!(
        keys.decode::<Position>(&binding, token, now),
        Err(Error::InvalidCursor)
    ));
}

#[rstest]
fn tampered_payloads_fail_verification(keys: KeyRing, binding: Binding, now: OffsetDateTime) {
    let token = keys.encode(&binding, &cursor(now)).unwrap();
    let (payload, tag) = token.split_once('.').unwrap();
    let forged = format!("{}.{tag}", &payload[..payload.len() - 2]);
    assert!(matches!(
        keys.decode::<Position>(&binding, &forged, now),
        Err(Error::InvalidCursor)
    ));
    let other_keys = KeyRing::new(["other-secret"]).unwrap();
    assert!(matches!(
        other_keys.decode::<Position>(&binding, &token, now),
        Err(Error::InvalidCursor)
    ));
}

#[rstest]
fn expired_cursors_report_traversal_expiry(keys: KeyRing, binding: Binding, now: OffsetDateTime) {
    let token = keys.encode(&binding, &cursor(now)).unwrap();
    assert!(
        keys.decode::<Position>(&binding, &token, now + Duration::minutes(29))
            .is_ok()
    );
    assert!(matches!(
        keys.decode::<Position>(&binding, &token, now + Duration::minutes(30)),
        Err(Error::TraversalExpired)
    ));
}

#[rstest]
fn rotated_keys_keep_verifying_earlier_cursors(binding: Binding, now: OffsetDateTime) {
    let old = KeyRing::new(["old-secret"]).unwrap();
    let token = old.encode(&binding, &cursor(now)).unwrap();
    let rotated = KeyRing::new(["new-secret", "old-secret"]).unwrap();
    assert_ne!(rotated.signing_key_id(), old.signing_key_id());
    assert!(rotated.decode::<Position>(&binding, &token, now).is_ok());
    let fresh = rotated.encode(&binding, &cursor(now)).unwrap();
    assert!(matches!(
        old.decode::<Position>(&binding, &fresh, now),
        Err(Error::InvalidCursor)
    ));
}

#[rstest]
#[case::no_keys(Vec::<&str>::new())]
#[case::empty_key(vec![""])]
fn key_rings_require_nonempty_secrets(#[case] secrets: Vec<&str>) {
    assert!(matches!(KeyRing::new(secrets), Err(Error::InvalidKeys)));
}

#[rstest]
fn traversal_identity_follows_binding_and_publication(binding: Binding, now: OffsetDateTime) {
    let first = Traversal::new(&binding, 1_790_000_000_000, now);
    let same = Traversal::new(&binding, 1_790_000_000_000, now + Duration::minutes(1));
    let later = Traversal::new(&binding, 1_790_000_000_001, now);
    let other_scope =
        Binding::new("traces:list", &("team-b", "user-1"), &(0_i64, 100_i64)).unwrap();
    assert_eq!(first.id, same.id);
    assert_ne!(first.id, later.id);
    assert_ne!(
        first.id,
        Traversal::new(&other_scope, 1_790_000_000_000, now).id
    );
    assert_eq!(first.published_at, "2026-09-21T14:13:20Z");
    assert_eq!(first.expires_at, "2026-09-21T14:13:20Z");
    assert_eq!(later.published_at, "2026-09-21T14:13:20.001Z");
}

#[rstest]
fn bounded_pages_keep_a_complete_prefix_and_reissue_the_cursor(
    binding: Binding,
    now: OffsetDateTime,
) {
    let page = Page {
        items: (0..8).map(|index| format!("item-{index:02}")).collect(),
        next_cursor: Some("after-07".into()),
        traversal: Traversal::new(&binding, 0, now),
    };
    let full_bytes = serde_json::to_vec(&page).unwrap().len();
    let bounded = page
        .clone()
        .bounded(full_bytes - 1, |last| Ok(format!("after-{}", &last[5..])))
        .unwrap();
    assert_eq!(bounded.items, page.items[..4]);
    assert_eq!(bounded.next_cursor.as_deref(), Some("after-03"));
    assert_eq!(bounded.traversal, page.traversal);
    assert!(
        page.clone()
            .bounded(full_bytes, |_| Ok(String::new()))
            .is_ok()
    );
    assert!(matches!(
        page.bounded(1, |_| Ok(String::new())),
        Err(Error::ResourceTooLarge)
    ));
}

#[rstest]
#[case::cursor(Error::InvalidCursor, FailureCode::InvalidCursor, false, false)]
#[case::expired(Error::TraversalExpired, FailureCode::TraversalExpired, false, true)]
#[case::changed(Error::TraversalChanged, FailureCode::TraversalChanged, false, true)]
#[case::too_large(Error::ResourceTooLarge, FailureCode::ResourceTooLarge, false, false)]
#[case::keys(Error::InvalidKeys, FailureCode::Unavailable, true, false)]
fn errors_map_to_stable_codes(
    #[case] error: Error,
    #[case] code: FailureCode,
    #[case] transient: bool,
    #[case] restarts: bool,
) {
    assert_eq!(error.code(), code);
    assert_eq!(code.is_transient(), transient);
    assert_eq!(code.restarts_traversal(), restarts);
    assert_eq!(serde_json::to_value(code).unwrap(), code.as_str());
}
