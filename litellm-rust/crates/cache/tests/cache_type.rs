use litellm_cache::CacheType;
use rstest::rstest;
use strum::VariantArray;

#[rstest]
#[case(CacheType::Local, "local")]
#[case(CacheType::Redis, "redis")]
#[case(CacheType::RedisSemantic, "redis-semantic")]
#[case(CacheType::ValkeySemantic, "valkey-semantic")]
#[case(CacheType::S3, "s3")]
#[case(CacheType::Disk, "disk")]
#[case(CacheType::QdrantSemantic, "qdrant-semantic")]
#[case(CacheType::AzureBlob, "azure-blob")]
#[case(CacheType::Gcs, "gcs")]
fn every_python_cache_type_has_one_round_trip_identity(
    #[case] cache_type: CacheType,
    #[case] name: &str,
) {
    assert_eq!(<&'static str>::from(cache_type), name);
    assert_eq!(name.parse::<CacheType>().ok(), Some(cache_type));
    assert_eq!(
        serde_json::to_value(cache_type).unwrap(),
        serde_json::Value::from(name)
    );
    assert_eq!(
        CacheType::VARIANTS
            .iter()
            .filter(|candidate| <&'static str>::from(**candidate) == name)
            .count(),
        1
    );
}

#[rstest]
fn python_cache_types_are_listed_in_python_order() {
    assert_eq!(
        CacheType::VARIANTS
            .iter()
            .map(|cache_type| <&'static str>::from(*cache_type))
            .collect::<Vec<_>>(),
        [
            "local",
            "redis",
            "redis-semantic",
            "valkey-semantic",
            "s3",
            "disk",
            "qdrant-semantic",
            "azure-blob",
            "gcs",
        ]
    );
}

#[rstest]
#[case::unknown("memcached")]
#[case::case_sensitive("Redis")]
fn unknown_python_names_have_no_cache_type(#[case] name: &str) {
    assert!(name.parse::<CacheType>().is_err());
}
