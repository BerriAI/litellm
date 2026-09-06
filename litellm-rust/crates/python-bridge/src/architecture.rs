//! Enforcement: interpreter and runtime boundaries in the PyO3 crates.
//!
//! Companion to `crates/core/tests/workspace_crate_allowlist.rs`: layering
//! rules that live in the crate `AGENTS.md` files become tests that fail
//! until deliberately updated here.
//!
//! Scanned text is *production* source only: each file's content up to its
//! trailing `#[cfg(test)]` module (the codebase keeps test modules at EOF).
//! This scanner skips its own file, so the tokens below can be written out
//! literally.
//!
//! Rules, each tied to a documented failure mode:
//!
//! 1. GIL acquisition and release (`Python::attach`, `Python::with_gil`,
//!    `.detach(`, `.allow_threads(`) appears in `python-bridge` only in
//!    `execution.rs`. The interpreter boundary belongs to
//!    `litellm-python-interop` primitives and the single execution module;
//!    scattered attach/detach calls are how GIL-ordering deadlocks and
//!    5 ms-per-handoff contention creep in.
//! 2. `block_on` appears in `python-bridge` only in `execution.rs`. Blocking
//!    a thread on a future anywhere else (a route body, a pyclass method)
//!    stalls the calling Python thread and risks nested-runtime panics.
//! 3. Tokio runtime construction (`Runtime::new`, `tokio::runtime::Builder`,
//!    `Builder::new_*`) appears in `python-bridge` only in `execution.rs`
//!    and `lib.rs` (the `#[pymodule]` host-init site). One shared runtime
//!    per process; per-call construction costs threads and an event loop,
//!    and a second runtime silently fragments the worker budget.
//! 4. `SendWrapper` appears nowhere in `python-bridge` or
//!    `litellm-python-interop`. It makes `!Send` Python-bound values `Send`
//!    by panicking when touched from another Tokio worker — a latent
//!    runtime bomb, not a fix.
//! 5. `litellm-python-interop` stays domain-neutral: no `litellm_core`,
//!    `litellm_ai_gateway`, or `litellm_python_bridge` tokens in its
//!    sources, keeping the dependency direction acyclic.
//! 6. `litellm-python-interop` never blocks on futures and never constructs
//!    a Tokio runtime: it provides primitives, hosts drive runtimes.

use std::fs;
use std::path::{Path, PathBuf};

const BRIDGE_SRC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
const INTEROP_SRC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../python-interop/src");

const TEST_MODULE_MARKER: &str = "\n#[cfg(test)]";

const GIL_TOKENS: &[&str] = &[
    "Python::attach",
    "Python::with_gil",
    ".detach(",
    ".allow_threads(",
];

const RUNTIME_CONSTRUCTION_TOKENS: &[&str] = &[
    "Runtime::new",
    "tokio::runtime::Builder",
    "Builder::new_multi_thread",
    "Builder::new_current_thread",
];

const INTEROP_DOMAIN_TOKENS: &[&str] = &[
    "litellm_core",
    "litellm_ai_gateway",
    "litellm_python_bridge",
];

/// Collect `*.rs` files under `root`, depth-first.
fn rust_sources(root: &str) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries = fs::read_dir(dir).unwrap_or_else(|error| {
            panic!("{} should be readable: {error}", dir.display());
        });
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }

    let mut sources = Vec::new();
    walk(Path::new(root), &mut sources);
    sources
}

/// The production half of a file: everything before the trailing
/// `#[cfg(test)]` test module.
fn production_text(content: &str) -> &str {
    match content.find(TEST_MODULE_MARKER) {
        Some(offset) => &content[..offset],
        None => content,
    }
}

/// Describe a violation of `token` in `relative_path`, or `None` when the
/// path is allowlisted or the token only appears in test code.
fn violation(
    content: &str,
    relative_path: &Path,
    token: &str,
    allowed_suffixes: &[&str],
) -> Option<String> {
    if !production_text(content).contains(token) {
        return None;
    }
    if allowed_suffixes
        .iter()
        .any(|suffix| relative_path.ends_with(suffix))
    {
        return None;
    }
    Some(format!(
        "`{token}` in {} (allowed only in {allowed_suffixes:?})",
        relative_path.display()
    ))
}

/// Assert that `token` never appears in the production half of any scanned
/// source under `root`, except in files whose path ends with an allowed
/// suffix. The scanner's own file is skipped so its literals cannot match.
fn assert_confined(root: &str, token: &str, allowed_suffixes: &[&str], rule: &str) {
    let mut violations = Vec::new();
    for path in rust_sources(root) {
        if path
            .file_name()
            .is_some_and(|name| name == "architecture.rs")
        {
            continue;
        }
        let content = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("{} should be readable: {error}", path.display()));
        let relative = path.strip_prefix(root).unwrap_or(&path);
        if let Some(violation) = violation(&content, relative, token, allowed_suffixes) {
            violations.push(format!("{}: {violation}", path.display()));
        }
    }
    assert!(
        violations.is_empty(),
        "{rule}\nviolations:\n  {}",
        violations.join("\n  ")
    );
}

#[test]
fn python_bridge_gil_calls_are_confined_to_execution() {
    for token in GIL_TOKENS {
        assert_confined(
            BRIDGE_SRC,
            token,
            &["execution.rs"],
            "interpreter attach/detach belongs to litellm-python-interop and python-bridge/src/execution.rs (rule 1 in the module docs)",
        );
    }
}

#[test]
fn python_bridge_block_on_is_confined_to_execution() {
    assert_confined(
        BRIDGE_SRC,
        "block_on",
        &["execution.rs"],
        "blocking on futures belongs to python-bridge/src/execution.rs (rule 2 in the module docs)",
    );
}

#[test]
fn python_bridge_runtime_construction_is_confined_to_execution_and_module_init() {
    for token in RUNTIME_CONSTRUCTION_TOKENS {
        assert_confined(
            BRIDGE_SRC,
            token,
            &["execution.rs", "lib.rs"],
            "one shared Tokio runtime, constructed only at the host-init site (rule 3 in the module docs)",
        );
    }
}

#[test]
fn send_wrapper_is_banned_in_the_pyo3_crates() {
    assert_confined(
        BRIDGE_SRC,
        "SendWrapper",
        &[],
        "SendWrapper panics across Tokio workers; convert to owned types instead (rule 4 in the module docs)",
    );
    assert_confined(
        INTEROP_SRC,
        "SendWrapper",
        &[],
        "SendWrapper panics across Tokio workers; convert to owned types instead (rule 4 in the module docs)",
    );
}

#[test]
fn python_interop_stays_domain_neutral() {
    for token in INTEROP_DOMAIN_TOKENS {
        assert_confined(
            INTEROP_SRC,
            token,
            &[],
            "litellm-python-interop must not depend on LiteLLM domain crates (rule 5 in the module docs)",
        );
    }
}

#[test]
fn python_interop_never_blocks_or_builds_runtimes() {
    assert_confined(
        INTEROP_SRC,
        "block_on",
        &[],
        "litellm-python-interop provides primitives; hosts drive runtimes (rule 6 in the module docs)",
    );
    for token in RUNTIME_CONSTRUCTION_TOKENS {
        assert_confined(
            INTEROP_SRC,
            token,
            &[],
            "litellm-python-interop provides primitives; hosts drive runtimes (rule 6 in the module docs)",
        );
    }
}

#[test]
fn scanner_flags_a_token_outside_the_allowlist() {
    let content = "fn route() {\n    Python::attach(|py| drop(py));\n}\n";
    let violation = violation(
        content,
        Path::new("routes/chat.rs"),
        GIL_TOKENS[0],
        &["execution.rs"],
    );
    assert!(violation.is_some(), "token outside the allowlist must flag");
}

#[test]
fn scanner_accepts_a_token_inside_the_allowlist() {
    let content = "fn runner() {\n    Python::attach(|py| drop(py));\n}\n";
    let violation = violation(
        content,
        Path::new("src/execution.rs"),
        GIL_TOKENS[0],
        &["execution.rs"],
    );
    assert!(violation.is_none(), "allowlisted file must not flag");
}

#[test]
fn scanner_ignores_test_modules() {
    let content = "fn route() {}\n\n#[cfg(test)]\nmod tests {\n    fn probe() {\n        Python::attach(|py| drop(py));\n    }\n}\n";
    let violation = violation(
        content,
        Path::new("routes/chat.rs"),
        GIL_TOKENS[0],
        &["execution.rs"],
    );
    assert!(violation.is_none(), "test modules must not flag");
}
