#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sources {
    pub setting: &'static str,
    pub litellm_params: &'static [&'static str],
    pub litellm_setting: Option<&'static str>,
    pub env: &'static [&'static str],
}

impl Sources {
    pub fn guidance(&self) -> String {
        let steps: Vec<String> = [
            (!self.litellm_params.is_empty())
                .then(|| format!("pass {}", self.litellm_params.join(" or "))),
            self.litellm_setting
                .map(|setting| format!("set litellm.{setting}")),
            (!self.env.is_empty()).then(|| format!("set {}", self.env.join(" or "))),
        ]
        .into_iter()
        .flatten()
        .collect();
        match steps.as_slice() {
            [first, second, third] => format!("{first}, {second}, or {third}"),
            steps => steps.join(" or "),
        }
    }
}
