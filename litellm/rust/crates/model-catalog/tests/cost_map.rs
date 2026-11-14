use litellm_model_catalog::{
    BUNDLED_REGISTRY, CostMap, ErrorKind, IntegrityLimits, Provenance, Rejects, git_blob_id,
};
use rstest::rstest;
use serde_json::json;

fn any_size() -> IntegrityLimits {
    IntegrityLimits::new(0, 0.0).unwrap()
}

#[rstest]
#[case::not_json(b"{".as_slice(), false, ErrorKind::Json)]
#[case::empty(b"{}".as_slice(), true, ErrorKind::Empty)]
fn a_refused_document_says_whether_it_failed_the_integrity_limits(
    #[case] body: &[u8],
    #[case] integrity: bool,
    #[case] kind: ErrorKind,
) {
    let error = CostMap::read(body, None, any_size()).unwrap_err();
    assert_eq!((error.is_integrity(), error.kind()), (integrity, kind));
}

#[rstest]
#[case::string(json!("mirrored 2026-11-30"))]
#[case::null(json!(null))]
#[case::list(json!([1, 2]))]
#[case::number(json!(5))]
fn a_document_with_a_top_level_value_that_is_not_a_row_is_refused(
    #[case] value: serde_json::Value,
) {
    let body = json!({"m": {"litellm_provider": "p"}, "_comment": value}).to_string();
    let error = CostMap::read(body.as_bytes(), None, any_size()).unwrap_err();
    assert_eq!(
        (error.kind(), error.key().as_deref(), error.is_integrity()),
        (ErrorKind::UnservableValue, Some("_comment"), false)
    );
    assert_eq!(error.rejects(), Rejects::Document);
}

#[rstest]
#[case::sample_spec("sample_spec")]
#[case::fallback_generalizations("fallback_generalizations")]
fn a_reserved_top_level_key_may_hold_any_value(#[case] key: &str) {
    let body = json!({"m": {"litellm_provider": "p"}, key: "not a row"}).to_string();
    assert!(CostMap::read(body.as_bytes(), None, any_size()).is_ok());
}

#[rstest]
fn provenance_of_a_document_with_a_byte_order_mark_is_its_served_bytes() {
    let body = [
        b"\xEF\xBB\xBF".as_slice(),
        br#"{"m":{"litellm_provider":"p"}}"#,
    ]
    .concat();
    let cost_map = CostMap::read(&body, None, any_size()).unwrap();
    assert_eq!(cost_map.provenance().revision, git_blob_id(&body));
}

#[rstest]
fn provenance_is_the_read_bytes_and_etag() {
    let body = br#"{"m":{"litellm_provider":"p"}}"#;
    let cost_map = CostMap::read(body, Some("\"v1\"".into()), any_size()).unwrap();
    assert_eq!(
        cost_map.provenance(),
        &Provenance {
            revision: "eaa747f210a1527d5ea7138be0f768c47578ca68".to_owned(),
            etag: Some("\"v1\"".into()),
        }
    );
    assert_eq!(
        CostMap::bundled().provenance().revision,
        git_blob_id(BUNDLED_REGISTRY)
    );
}

#[rstest]
fn every_name_of_an_aliased_row_lists_its_owner_first(
    #[values("owner", "short", "owner-latest")] key: &str,
) {
    let body = json!({
        "owner": {"litellm_provider": "p", "aliases": ["short"]},
        "owner-latest": {"alias_of": "owner"},
        "plain": {"litellm_provider": "p"},
    })
    .to_string();
    let cost_map = CostMap::read(body.as_bytes(), None, any_size()).unwrap();
    assert_eq!(cost_map.names(key), ["owner", "short", "owner-latest"]);
    assert!(cost_map.names("plain").is_empty());
}
