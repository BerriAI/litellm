use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::OnceLock;

use tokenizers::Tokenizer;

use super::encoding::count_text;
use crate::{CoreError, CoreResult};

enum TokenizerImpl {
    Tiktoken(&'static tiktoken::CoreBpe),
    HuggingFace(Box<Tokenizer>),
}

pub(crate) struct ResolvedTokenizer(TokenizerImpl);

impl ResolvedTokenizer {
    pub(crate) fn count(&self, text: &str) -> usize {
        match &self.0 {
            TokenizerImpl::Tiktoken(enc) => count_text(enc, text),
            TokenizerImpl::HuggingFace(tok) => tok
                .encode(text, false)
                .map(|e| e.get_ids().len())
                .unwrap_or(0),
        }
    }
}

static ANTHROPIC_TOKENIZER: OnceLock<Option<Box<Tokenizer>>> = OnceLock::new();

pub(crate) fn resolve(model: &str) -> CoreResult<ResolvedTokenizer> {
    if let Some(hf) = try_resolve_hf(model) {
        return Ok(ResolvedTokenizer(TokenizerImpl::HuggingFace(hf)));
    }

    let normalized = fix_model_name(model);

    if is_non_openai_model(&normalized) {
        return Err(CoreError::InvalidRequest(format!(
            "model '{model}' requires a HuggingFace tokenizer; falling back to Python path"
        )));
    }

    let enc = if normalized.contains("gpt-4o") {
        tiktoken::get_encoding("o200k_base")
            .expect("o200k_base is a built-in encoding and always available")
    } else {
        tiktoken::encoding_for_model(normalized.as_ref()).unwrap_or(
            tiktoken::get_encoding("cl100k_base")
                .expect("cl100k_base is a built-in encoding and always available"),
        )
    };

    Ok(ResolvedTokenizer(TokenizerImpl::Tiktoken(enc)))
}

fn is_non_openai_model(model: &str) -> bool {
    let lower = model.to_lowercase();
    lower.contains("cohere")
        || lower.contains("command-r")
        || lower.contains("llama")
        || lower.contains("mistral")
        || lower.contains("mixtral")
        || lower.contains("qwen")
        || lower.contains("yi-")
        || lower.contains("deepseek")
        || lower.contains("gemma")
}

fn try_resolve_hf(model: &str) -> Option<Box<Tokenizer>> {
    let lower = model.to_lowercase();

    if lower.contains("claude") && !lower.contains("claude-3") {
        return load_anthropic_tokenizer();
    }

    None
}

fn load_anthropic_tokenizer() -> Option<Box<Tokenizer>> {
    ANTHROPIC_TOKENIZER
        .get_or_init(|| {
            let path = find_anthropic_tokenizer_path()?;
            let json = std::fs::read_to_string(&path).ok()?;
            Tokenizer::from_bytes(json.as_bytes()).ok().map(Box::new)
        })
        .clone()
}

fn find_anthropic_tokenizer_path() -> Option<PathBuf> {
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        let candidate = cwd.join("litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json");
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

fn fix_model_name(model: &str) -> Cow<'_, str> {
    if model.contains("-35") {
        Cow::Owned(model.replace("-35", "-3.5"))
    } else {
        Cow::Borrowed(model)
    }
}
