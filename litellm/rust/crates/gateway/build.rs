//! Hands the build the LiteLLM version the repository's `pyproject.toml` names, which the gateway
//! tells in `x-litellm-version` as the proxy tells its package version. A build outside the
//! repository names none.

use std::{env, fs, path::Path};

fn main() {
    let Ok(manifest) = env::var("CARGO_MANIFEST_DIR") else {
        return;
    };
    let pyproject = Path::new(&manifest).join("../../../../pyproject.toml");
    println!("cargo:rerun-if-changed={}", pyproject.display());
    let version = fs::read_to_string(&pyproject)
        .ok()
        .and_then(|text| text.parse::<toml::Table>().ok())
        .and_then(|table| Some(table.get("project")?.get("version")?.as_str()?.to_owned()));
    if let Some(version) = version {
        println!("cargo:rustc-env=LITELLM_VERSION={version}");
    }
}
