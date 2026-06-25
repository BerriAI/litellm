use std::fs;
use std::path::{Path, PathBuf};

const FORBIDDEN_RAW_JSON_MARKERS: &[&str] = &[
    "serde_json::Value",
    "serde_json::Map",
    "use serde_json::{Map",
    "use serde_json::{Value",
    "json!(",
];

fn core_src_root() -> PathBuf {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"))
        .canonicalize()
        .expect("core src root should resolve")
}

fn rust_files(root: &Path) -> Vec<PathBuf> {
    fs::read_dir(root)
        .expect("directory should be readable")
        .filter_map(Result::ok)
        .flat_map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                rust_files(&path)
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                vec![path]
            } else {
                Vec::new()
            }
        })
        .collect()
}

#[test]
fn core_translation_code_does_not_use_raw_json_values() {
    let violations: Vec<String> = rust_files(&core_src_root())
        .into_iter()
        .filter_map(|path| {
            let source = fs::read_to_string(&path).expect("rust source should be readable");
            let markers: Vec<&str> = FORBIDDEN_RAW_JSON_MARKERS
                .iter()
                .copied()
                .filter(|marker| source.contains(marker))
                .collect();
            if markers.is_empty() {
                None
            } else {
                Some(format!("{}: {}", path.display(), markers.join(", ")))
            }
        })
        .collect();

    assert!(
        violations.is_empty(),
        "litellm-core is the typed translation layer. Do not use raw serde_json::Value/Map or json! in crates/core/src; add route/provider structs instead.\n{}",
        violations.join("\n")
    );
}
