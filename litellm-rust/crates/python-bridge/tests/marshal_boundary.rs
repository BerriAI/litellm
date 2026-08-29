use std::fs;
use std::path::{Path, PathBuf};

const DISALLOWED_OUTSIDE_MARSHAL: &[&str] = &[
    "py.import(\"json\")",
    "pythonize::",
    "serde_json::to_string",
    "serde_json::from_str",
];

fn source_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn rust_sources(directory: &Path) -> Vec<PathBuf> {
    fs::read_dir(directory)
        .expect("bridge source directory should be readable")
        .map(|entry| {
            entry
                .expect("bridge source entry should be readable")
                .path()
        })
        .flat_map(|path| {
            if path.is_dir() {
                rust_sources(&path)
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                vec![path]
            } else {
                Vec::new()
            }
        })
        .collect()
}

#[test]
fn serialization_is_centralized_in_marshal_module() {
    let root = source_root();

    for path in rust_sources(&root) {
        if path == root.join("marshal.rs") {
            continue;
        }
        let source = fs::read_to_string(&path).expect("bridge source should be readable");
        for disallowed in DISALLOWED_OUTSIDE_MARSHAL {
            assert!(
                !source.contains(disallowed),
                "{} bypasses the typed marshal module with `{disallowed}`",
                path.display()
            );
        }
    }
}
