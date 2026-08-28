use std::borrow::Cow;

pub(crate) use super::hf_tokenizer::ResolvedTokenizer;
pub(crate) use super::hf_tokenizer::resolve;

pub(crate) fn count_text(encoding: &tiktoken::CoreBpe, text: &str) -> usize {
    encoding.count(text)
}

fn fix_model_name(model: &str) -> Cow<'_, str> {
    if model.contains("-35") {
        Cow::Owned(model.replace("-35", "-3.5"))
    } else {
        Cow::Borrowed(model)
    }
}

pub(crate) fn normalized_model_name(model: &str) -> Cow<'_, str> {
    fix_model_name(model)
}
