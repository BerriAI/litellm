use std::ops::Add;

use semver::Version;

use crate::Settings;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prompt {
    pub text: String,
    pub allow_tools: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl Add for Usage {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            input_tokens: self.input_tokens + other.input_tokens,
            output_tokens: self.output_tokens + other.output_tokens,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    pub text: String,
    pub tool_calls: Vec<String>,
    pub usage: Usage,
    pub errors: Vec<String>,
    pub exit_code: Option<i32>,
}

impl Outcome {
    pub fn succeeded(&self) -> bool {
        self.exit_code == Some(0) && self.errors.is_empty()
    }
}

pub trait Drive {
    fn args(&self, version: &Version, settings: &Settings, prompt: &Prompt) -> Vec<String>;

    fn parse(&self, version: &Version, stdout: &str) -> Outcome;
}

pub(crate) fn json_lines<'a, T: serde::de::DeserializeOwned + 'a>(
    stdout: &'a str,
) -> impl Iterator<Item = T> + 'a {
    stdout
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
}
