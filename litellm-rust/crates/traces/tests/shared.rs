use litellm_traces::Shared;
use rstest::rstest;

#[rstest]
fn clones_preserve_values_and_serialize_transparently() {
    let original = Shared::new(vec!["value".to_owned()]);
    let cloned = original.clone();
    assert_eq!(cloned.as_ref(), original.as_ref());
    assert_eq!(
        serde_json::to_value(&cloned).unwrap(),
        serde_json::json!(["value"])
    );
}

#[rstest]
fn clones_share_storage_without_merging_equal_values() {
    let original = Shared::new("value".to_owned());
    let cloned = original.clone();
    let equal = Shared::new("value".to_owned());
    assert!(original.shares_storage_with(&cloned));
    assert!(!original.shares_storage_with(&equal));
    assert_eq!(*original, *equal);
}
