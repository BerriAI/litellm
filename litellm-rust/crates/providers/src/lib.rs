pub mod mistral;
pub mod ocr;

// `litellm_providers::ocr(...)` — the typed async OCR entry point. The function
// lives in the `ocr` module; this re-export lets callers spell it without the
// `ocr::ocr` stutter. (A module and a value can share a name in Rust.)
pub use ocr::ocr;
