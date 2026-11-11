use litellm_model_catalog::{
    BUNDLED_REGISTRY, CostMap, ErrorKind, IntegrityLimits, Provenance, git_blob_id,
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
