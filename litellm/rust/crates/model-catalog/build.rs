use std::{env, fs, path::Path};

use serde_json::{Map, Value};

const REGISTRY: &str = "../../../../model_prices_and_context_window.json";

fn main() {
    println!("cargo:rerun-if-changed={REGISTRY}");
    let body = fs::read(REGISTRY).expect("the bundled registry is readable");
    let document: Map<String, Value> =
        serde_json::from_slice(&body).expect("the bundled registry is a JSON object");
    let models = document
        .iter()
        .filter(|(key, _)| !matches!(key.as_str(), "sample_spec" | "fallback_generalizations"))
        .filter(|(_, value)| {
            value
                .as_object()
                .is_some_and(|row| row.get("alias_of").is_none_or(Value::is_null))
        })
        .count();
    let out = Path::new(&env::var("OUT_DIR").unwrap()).join("bundled_model_count.rs");
    fs::write(out, format!("{models}")).unwrap();
}
