use std::{path::PathBuf, time::Duration};

use litellm_cache_redis::RedisTopology;

/// What makes a native backend the one a Python facade describes: the configuration a user can
/// observe on the Python object, captured once so facade projection and native construction
/// compare plain data instead of reaching into each backend type.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum BackendIdentity {
    Memory {
        capacity: usize,
        max_entry_bytes: Option<usize>,
        default_ttl: Option<Duration>,
    },
    Redis {
        topology: RedisTopology,
        namespace: Option<String>,
        default_ttl: Option<Duration>,
    },
    S3 {
        bucket: String,
        key_prefix: String,
        region: String,
        endpoint: Option<String>,
    },
    Gcs {
        bucket_name: String,
        key_prefix: String,
        path_service_account: Option<String>,
    },
    Disk {
        directory: PathBuf,
    },
    AzureBlob {
        account_url: String,
        container: String,
    },
    RedisSemantic {
        index_name: String,
        /// The backend stores the threshold as `f32`; a facade's `f64` is compared at that width.
        similarity_threshold: f32,
    },
    ValkeySemantic {
        index_name: String,
        similarity_threshold: f64,
    },
    QdrantSemantic {
        collection_name: String,
        similarity_threshold: f64,
        vector_size: u64,
        embedding_model: String,
    },
}

const TYPES: &str = "facade and native backend types must match";

impl BackendIdentity {
    /// The native backend name the facade guard maps to its Python backend class.
    pub(super) fn kind(&self) -> &'static str {
        match self {
            Self::Memory { .. } => "memory",
            Self::Redis { .. } => "redis",
            Self::S3 { .. } => "s3",
            Self::Gcs { .. } => "gcs",
            Self::ValkeySemantic { .. } => "valkey-semantic",
            Self::RedisSemantic { .. } => "redis_semantic",
            Self::QdrantSemantic { .. } => "qdrant_semantic",
            Self::Disk { .. } => "disk",
            Self::AzureBlob { .. } => "azure-blob",
        }
    }

    /// The `LiteLLMCacheType` value a facade of this backend carries in `Cache.type`.
    pub(super) fn cache_type(&self) -> &'static str {
        match self {
            Self::Memory { .. } => "local",
            Self::Redis { .. } => "redis",
            Self::S3 { .. } => "s3",
            Self::Gcs { .. } => "gcs",
            Self::ValkeySemantic { .. } => "valkey-semantic",
            Self::RedisSemantic { .. } => "redis-semantic",
            Self::QdrantSemantic { .. } => "qdrant-semantic",
            Self::Disk { .. } => "disk",
            Self::AzureBlob { .. } => "azure-blob",
        }
    }

    /// The first difference between the facade's configuration (`self`) and the native
    /// backend (`native`), in the order Python users see the attributes.
    pub(super) fn mismatch(&self, native: &Self) -> Option<&'static str> {
        let mut differences: Vec<(bool, &'static str)> = Vec::new();
        let mut differs = |condition: bool, message: &'static str| {
            differences.push((condition, message));
        };
        match (self, native) {
            (
                Self::Memory {
                    capacity,
                    max_entry_bytes,
                    default_ttl,
                },
                Self::Memory {
                    capacity: native_capacity,
                    max_entry_bytes: native_max_entry_bytes,
                    default_ttl: native_default_ttl,
                },
            ) => {
                differs(
                    default_ttl != native_default_ttl,
                    "facade and native backend default TTLs must match",
                );
                differs(
                    capacity != native_capacity,
                    "facade and native backend capacities must match",
                );
                differs(
                    max_entry_bytes != native_max_entry_bytes,
                    "facade and native backend item limits must match",
                );
            }
            (
                Self::Redis {
                    topology,
                    namespace,
                    default_ttl,
                },
                Self::Redis {
                    topology: native_topology,
                    namespace: native_namespace,
                    default_ttl: native_default_ttl,
                },
            ) => {
                differs(
                    default_ttl != native_default_ttl,
                    "facade and native backend default TTLs must match",
                );
                differs(
                    topology != native_topology,
                    "facade and native backend topologies must match",
                );
                differs(
                    namespace != native_namespace,
                    "facade and native backend namespaces must match",
                );
            }
            (
                Self::S3 {
                    bucket,
                    key_prefix,
                    region,
                    endpoint,
                },
                Self::S3 {
                    bucket: native_bucket,
                    key_prefix: native_key_prefix,
                    region: native_region,
                    endpoint: native_endpoint,
                },
            ) => {
                differs(
                    bucket != native_bucket,
                    "facade and native backend buckets must match",
                );
                differs(
                    key_prefix != native_key_prefix,
                    "facade and native backend key prefixes must match",
                );
                differs(
                    region != native_region,
                    "facade and native backend regions must match",
                );
                differs(
                    endpoint != native_endpoint,
                    "facade and native backend endpoints must match",
                );
            }
            (
                Self::Gcs {
                    bucket_name,
                    key_prefix,
                    path_service_account,
                },
                Self::Gcs {
                    bucket_name: native_bucket_name,
                    key_prefix: native_key_prefix,
                    path_service_account: native_path_service_account,
                },
            ) => {
                differs(
                    bucket_name != native_bucket_name,
                    "facade and native backend buckets must match",
                );
                differs(
                    key_prefix != native_key_prefix,
                    "facade and native backend key prefixes must match",
                );
                differs(
                    path_service_account != native_path_service_account,
                    "facade and native backend credentials must match",
                );
            }
            (
                Self::Disk { directory },
                Self::Disk {
                    directory: native_directory,
                },
            ) => {
                let canonical = |path: &PathBuf| std::fs::canonicalize(path).ok();
                differs(
                    canonical(directory) != canonical(native_directory),
                    "facade and native backend directories must match",
                );
            }
            (
                Self::AzureBlob {
                    account_url,
                    container,
                },
                Self::AzureBlob {
                    account_url: native_account_url,
                    container: native_container,
                },
            ) => {
                differs(
                    account_url != native_account_url || container != native_container,
                    "facade and native backend containers must match",
                );
            }
            (
                Self::RedisSemantic {
                    index_name,
                    similarity_threshold,
                },
                Self::RedisSemantic {
                    index_name: native_index_name,
                    similarity_threshold: native_similarity_threshold,
                },
            ) => {
                differs(
                    index_name != native_index_name,
                    "facade and native backend index names must match",
                );
                differs(
                    similarity_threshold != native_similarity_threshold,
                    "facade and native backend similarity thresholds must match",
                );
            }
            (
                Self::ValkeySemantic {
                    index_name,
                    similarity_threshold,
                },
                Self::ValkeySemantic {
                    index_name: native_index_name,
                    similarity_threshold: native_similarity_threshold,
                },
            ) => {
                differs(
                    index_name != native_index_name
                        || similarity_threshold != native_similarity_threshold,
                    "facade and native semantic settings must match",
                );
            }
            (
                Self::QdrantSemantic {
                    collection_name,
                    similarity_threshold,
                    vector_size,
                    embedding_model,
                },
                Self::QdrantSemantic {
                    collection_name: native_collection_name,
                    similarity_threshold: native_similarity_threshold,
                    vector_size: native_vector_size,
                    embedding_model: native_embedding_model,
                },
            ) => {
                differs(
                    collection_name != native_collection_name,
                    "facade and native backend collections must match",
                );
                differs(
                    similarity_threshold != native_similarity_threshold,
                    "facade and native backend similarity thresholds must match",
                );
                differs(
                    vector_size != native_vector_size,
                    "facade and native backend vector sizes must match",
                );
                differs(
                    embedding_model != native_embedding_model,
                    "facade and native backend embedding models must match",
                );
            }
            _ => return Some(TYPES),
        }
        differences
            .into_iter()
            .find_map(|(condition, message)| condition.then_some(message))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use litellm_cache_redis::{RedisNode, RedisTopology};

    use super::BackendIdentity;

    fn memory() -> BackendIdentity {
        BackendIdentity::Memory {
            capacity: 200,
            max_entry_bytes: Some(1024),
            default_ttl: Some(Duration::from_secs(60)),
        }
    }

    fn redis() -> BackendIdentity {
        BackendIdentity::Redis {
            topology: RedisTopology::Standalone,
            namespace: Some("team".into()),
            default_ttl: Some(Duration::from_secs(60)),
        }
    }

    fn s3() -> BackendIdentity {
        BackendIdentity::S3 {
            bucket: "bucket".into(),
            key_prefix: "cache/".into(),
            region: "us-east-1".into(),
            endpoint: None,
        }
    }

    fn gcs() -> BackendIdentity {
        BackendIdentity::Gcs {
            bucket_name: "bucket".into(),
            key_prefix: "cache/".into(),
            path_service_account: Some("credentials.json".into()),
        }
    }

    fn azure() -> BackendIdentity {
        BackendIdentity::AzureBlob {
            account_url: "https://account.blob.core.windows.net".into(),
            container: "cache".into(),
        }
    }

    fn redis_semantic() -> BackendIdentity {
        BackendIdentity::RedisSemantic {
            index_name: "idx".into(),
            similarity_threshold: 0.8,
        }
    }

    #[test]
    fn redis_semantic_thresholds_compare_at_backend_precision() {
        let facade = BackendIdentity::RedisSemantic {
            index_name: "idx".into(),
            similarity_threshold: 0.8_f64 as f32,
        };
        assert_eq!(facade.mismatch(&redis_semantic()), None);
    }

    fn valkey_semantic() -> BackendIdentity {
        BackendIdentity::ValkeySemantic {
            index_name: "idx".into(),
            similarity_threshold: 0.8,
        }
    }

    fn qdrant() -> BackendIdentity {
        BackendIdentity::QdrantSemantic {
            collection_name: "collection".into(),
            similarity_threshold: 0.8,
            vector_size: 1536,
            embedding_model: "text-embedding-3-small".into(),
        }
    }

    #[test]
    fn identical_identities_have_no_mismatch() {
        for identity in [
            memory(),
            redis(),
            s3(),
            gcs(),
            azure(),
            redis_semantic(),
            valkey_semantic(),
            qdrant(),
            BackendIdentity::Disk {
                directory: std::env::temp_dir(),
            },
        ] {
            assert_eq!(identity.mismatch(&identity), None, "{identity:?}");
        }
    }

    #[test]
    fn different_kinds_report_a_type_mismatch() {
        assert_eq!(
            memory().mismatch(&redis()),
            Some("facade and native backend types must match")
        );
        assert_eq!(
            redis_semantic().mismatch(&valkey_semantic()),
            Some("facade and native backend types must match")
        );
    }

    #[test]
    fn the_first_differing_field_names_the_mismatch() {
        let BackendIdentity::Memory { capacity, .. } = memory() else {
            unreachable!()
        };
        assert_eq!(
            memory().mismatch(&BackendIdentity::Memory {
                capacity: capacity + 1,
                max_entry_bytes: Some(1),
                default_ttl: Some(Duration::from_secs(60)),
            }),
            Some("facade and native backend capacities must match")
        );
        assert_eq!(
            memory().mismatch(&BackendIdentity::Memory {
                capacity,
                max_entry_bytes: Some(1),
                default_ttl: Some(Duration::from_secs(61)),
            }),
            Some("facade and native backend default TTLs must match")
        );
        assert_eq!(
            redis().mismatch(&BackendIdentity::Redis {
                topology: RedisTopology::Cluster {
                    startup_nodes: vec![RedisNode {
                        host: "node".into(),
                        port: 7000,
                    }],
                },
                namespace: None,
                default_ttl: Some(Duration::from_secs(60)),
            }),
            Some("facade and native backend topologies must match")
        );
        assert_eq!(
            s3().mismatch(&BackendIdentity::S3 {
                bucket: "bucket".into(),
                key_prefix: "cache/".into(),
                region: "us-east-1".into(),
                endpoint: Some("http://localhost:9000".into()),
            }),
            Some("facade and native backend endpoints must match")
        );
        assert_eq!(
            gcs().mismatch(&BackendIdentity::Gcs {
                bucket_name: "bucket".into(),
                key_prefix: "cache/".into(),
                path_service_account: None,
            }),
            Some("facade and native backend credentials must match")
        );
        assert_eq!(
            azure().mismatch(&BackendIdentity::AzureBlob {
                account_url: "https://account.blob.core.windows.net".into(),
                container: "other".into(),
            }),
            Some("facade and native backend containers must match")
        );
        assert_eq!(
            valkey_semantic().mismatch(&BackendIdentity::ValkeySemantic {
                index_name: "idx".into(),
                similarity_threshold: 0.9,
            }),
            Some("facade and native semantic settings must match")
        );
        assert_eq!(
            qdrant().mismatch(&BackendIdentity::QdrantSemantic {
                collection_name: "collection".into(),
                similarity_threshold: 0.8,
                vector_size: 1536,
                embedding_model: "text-embedding-3-large".into(),
            }),
            Some("facade and native backend embedding models must match")
        );
    }

    #[test]
    fn disk_directories_compare_canonically() {
        let directory = std::env::temp_dir();
        let mut indirect = directory.clone();
        indirect.push(".");
        assert_eq!(
            BackendIdentity::Disk {
                directory: directory.clone()
            }
            .mismatch(&BackendIdentity::Disk {
                directory: indirect
            }),
            None
        );
        assert_eq!(
            BackendIdentity::Disk { directory }.mismatch(&BackendIdentity::Disk {
                directory: "/definitely/missing".into()
            }),
            Some("facade and native backend directories must match")
        );
    }
}
