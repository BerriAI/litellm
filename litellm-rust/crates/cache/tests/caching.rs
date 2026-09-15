use litellm_cache::{
    CacheControls, CacheKeyContext, CacheKeyField, CacheKeyInput, cache_key, get_cache_key,
};
use sha2::{Digest, Sha256};

#[test]
fn keys_match_python_order_groups_files_presets_and_namespaces() {
    let mut input = CacheKeyInput {
        fields: vec![
            CacheKeyField {
                name: "model".into(),
                value: Some("deployment".into()),
                api_parameter: true,
                internal_parameter: false,
            },
            CacheKeyField {
                name: "file".into(),
                value: None,
                api_parameter: true,
                internal_parameter: false,
            },
        ],
        namespace: Some("team".into()),
        ..Default::default()
    };
    CacheKeyContext {
        model_group: Some("group".into()),
        caching_groups: vec![(vec!["group".into()], "['group']".into())],
        file_checksum: Some("checksum".into()),
        ..Default::default()
    }
    .apply(&mut input);
    assert_eq!(
        cache_key(&input),
        format!(
            "team:{:x}",
            Sha256::digest(b"model: ['group']file: checksum")
        )
    );
    input.preset = Some("preset".into());
    assert_eq!(get_cache_key(&input), "preset");
}

#[test]
fn cache_controls_honor_default_modes_and_directives() {
    let enabled = CacheControls {
        supported_call_type: true,
        configured: true,
        default_on: true,
        ..Default::default()
    };
    assert!(enabled.reads());
    assert!(enabled.writes());
    assert!(
        !CacheControls {
            default_on: false,
            ..enabled
        }
        .reads()
    );
    assert!(
        CacheControls {
            default_on: false,
            use_cache: true,
            ..enabled
        }
        .reads()
    );
    assert!(
        !CacheControls {
            no_cache: true,
            ..enabled
        }
        .reads()
    );
    assert!(
        !CacheControls {
            no_store: true,
            ..enabled
        }
        .writes()
    );
}
