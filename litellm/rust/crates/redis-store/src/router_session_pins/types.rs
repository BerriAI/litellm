use litellm_python_compat::Value;

/// What a claim asks for beyond the pin itself: the values that may keep holding the key, and
/// whether a bare deployment id an older writer left counts as one of them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Eligible {
    pub values: Option<Vec<Value>>,
    pub preserve_legacy_model_ids: bool,
}
