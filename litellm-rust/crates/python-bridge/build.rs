use std::path::Path;

fn pyo3_version_from_lock() -> Option<String> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let lockfile = Path::new(&manifest_dir).join("../../Cargo.lock");
    let contents = std::fs::read_to_string(lockfile).ok()?;
    let lines: Vec<&str> = contents.lines().collect();
    for pair in lines.windows(2) {
        if pair[0].trim() != "name = \"pyo3\"" {
            continue;
        }
        if let Some(version) = pair[1].trim().strip_prefix("version = ") {
            return Some(version.trim_matches('"').to_owned());
        }
    }
    None
}

fn main() {
    pyo3_build_config::use_pyo3_cfgs();
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
    println!(
        "cargo:rustc-env=LITELLM_NATIVE_PROFILE={}",
        std::env::var("PROFILE").unwrap_or_else(|_| "unknown".to_owned())
    );
    println!(
        "cargo:rustc-env=LITELLM_PYO3_VERSION={}",
        pyo3_version_from_lock().unwrap_or_else(|| "unknown".to_owned())
    );
    println!("cargo:rerun-if-changed=../../Cargo.lock");
}
