use std::{fs, path::PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};
use strum::VariantArray;

use crate::{Error, huggingface::HuggingFaceTokenizer, tiktoken::TiktokenTokenizer};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LocalTokenizerConfig {
    pub id: String,
    pub revision: String,
    pub engine: RawTextEngine,
    pub artifact: RawTextArtifact,
    pub special_text: Option<SpecialTextPolicy>,
    pub add_special_tokens: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum RawTextEngine {
    HuggingFace,
    Tiktoken { encoding: String },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum RawTextArtifact {
    Inline { content: String },
    File { path: PathBuf, sha256: String },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, VariantArray)]
#[serde(rename_all = "snake_case")]
pub enum SpecialTextPolicy {
    Literal,
    Recognize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct EffectiveRawTextPolicy {
    pub special_text: SpecialTextPolicy,
    pub add_special_tokens: bool,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RawTextIdentity {
    pub id: String,
    pub revision: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RawTextCacheKey {
    identity: RawTextIdentity,
    engine: RawTextEngine,
    policy: EffectiveRawTextPolicy,
}

pub struct PreparedRawTextConfig {
    identity: RawTextIdentity,
    engine: RawTextEngine,
    policy: EffectiveRawTextPolicy,
    artifact_content: String,
}

pub struct RawTextEncoding {
    pub count: usize,
    pub token_ids: Option<Vec<u32>>,
    pub tokens: Option<Vec<String>>,
}

enum Codec {
    HuggingFace(HuggingFaceTokenizer),
    Tiktoken(TiktokenTokenizer),
}

pub struct RawTextTokenizer {
    identity: RawTextIdentity,
    policy: EffectiveRawTextPolicy,
    codec: Codec,
}

impl LocalTokenizerConfig {
    pub fn resolved_policy(&self) -> Result<EffectiveRawTextPolicy, Error> {
        let (default_special_text, default_add_special_tokens) = match &self.engine {
            RawTextEngine::HuggingFace => (SpecialTextPolicy::Recognize, true),
            RawTextEngine::Tiktoken { .. } => (SpecialTextPolicy::Literal, false),
        };
        let policy = EffectiveRawTextPolicy {
            special_text: self.special_text.unwrap_or(default_special_text),
            add_special_tokens: self
                .add_special_tokens
                .unwrap_or(default_add_special_tokens),
        };
        if self.id.trim().is_empty() {
            return Err(Error::RawTextConfig("id"));
        }
        if self.revision.trim().is_empty() {
            return Err(Error::RawTextConfig("revision"));
        }
        if matches!(&self.engine, RawTextEngine::Tiktoken { .. }) && policy.add_special_tokens {
            return Err(Error::RawTextConfig("add_special_tokens"));
        }
        if let RawTextArtifact::File { sha256, .. } = &self.artifact
            && (sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            return Err(Error::RawTextConfig("artifact.sha256"));
        }
        if let RawTextEngine::Tiktoken { encoding } = &self.engine
            && encoding.trim().is_empty()
        {
            return Err(Error::RawTextConfig("engine.encoding"));
        }
        Ok(policy)
    }

    pub fn cache_key(&self) -> Result<RawTextCacheKey, Error> {
        let policy = self.resolved_policy()?;
        let sha256 = match &self.artifact {
            RawTextArtifact::Inline { content } => {
                format!("{:x}", Sha256::digest(content.as_bytes()))
            }
            RawTextArtifact::File { sha256, .. } => sha256.to_ascii_lowercase(),
        };
        Ok(RawTextCacheKey {
            identity: RawTextIdentity {
                id: self.id.clone(),
                revision: self.revision.clone(),
                sha256,
            },
            engine: self.engine.clone(),
            policy,
        })
    }

    pub fn prepare(&self) -> Result<PreparedRawTextConfig, Error> {
        let key = self.cache_key()?;
        self.prepare_for_key(key)
    }

    pub fn prepare_for_key(&self, key: RawTextCacheKey) -> Result<PreparedRawTextConfig, Error> {
        let policy = self.resolved_policy()?;
        if key.identity.id != self.id
            || key.identity.revision != self.revision
            || key.engine != self.engine
            || key.policy != policy
        {
            return Err(Error::RawTextConfig("cache key"));
        }
        if let RawTextArtifact::File { sha256, .. } = &self.artifact
            && !sha256.eq_ignore_ascii_case(&key.identity.sha256)
        {
            return Err(Error::RawTextConfig("artifact.sha256"));
        }
        let artifact_content = match &self.artifact {
            RawTextArtifact::Inline { content } => content.clone(),
            RawTextArtifact::File { path, .. } => {
                fs::read_to_string(path).map_err(Error::ArtifactRead)?
            }
        };
        let actual_sha256 = format!("{:x}", Sha256::digest(artifact_content.as_bytes()));
        if key.identity.sha256 != actual_sha256 {
            return Err(Error::ArtifactHashMismatch);
        }
        Ok(PreparedRawTextConfig {
            identity: key.identity,
            engine: key.engine,
            policy: key.policy,
            artifact_content,
        })
    }
}

impl RawTextCacheKey {
    pub fn identity(&self) -> &RawTextIdentity {
        &self.identity
    }

    pub fn engine(&self) -> &RawTextEngine {
        &self.engine
    }

    pub fn policy(&self) -> EffectiveRawTextPolicy {
        self.policy
    }
}

impl PreparedRawTextConfig {
    pub fn identity(&self) -> &RawTextIdentity {
        &self.identity
    }

    pub fn engine(&self) -> &RawTextEngine {
        &self.engine
    }

    pub fn policy(&self) -> EffectiveRawTextPolicy {
        self.policy
    }
}

impl RawTextTokenizer {
    pub fn load(config: LocalTokenizerConfig) -> Result<Self, Error> {
        Self::load_prepared(config.prepare()?)
    }

    pub fn load_prepared(prepared: PreparedRawTextConfig) -> Result<Self, Error> {
        let PreparedRawTextConfig {
            identity,
            engine,
            policy,
            artifact_content,
        } = prepared;
        let codec = match engine {
            RawTextEngine::HuggingFace => {
                let tokenizer = HuggingFaceTokenizer::from_json_with_policy(
                    &artifact_content,
                    policy.special_text == SpecialTextPolicy::Recognize,
                )
                .map_err(Error::from)?;
                if !tokenizer.is_deterministic() {
                    return Err(Error::RawTextConfig("model.dropout"));
                }
                Codec::HuggingFace(tokenizer)
            }
            RawTextEngine::Tiktoken { encoding } => Codec::Tiktoken(
                TiktokenTokenizer::from_ranks(&encoding, &artifact_content).map_err(Error::from)?,
            ),
        };
        Ok(Self {
            identity,
            policy,
            codec,
        })
    }

    pub fn identity(&self) -> &RawTextIdentity {
        &self.identity
    }

    pub fn policy(&self) -> EffectiveRawTextPolicy {
        self.policy
    }

    pub fn supports_token_ids(&self) -> bool {
        true
    }

    pub fn supports_tokens(&self) -> bool {
        matches!(&self.codec, Codec::HuggingFace(_))
    }

    pub fn count(&self, text: &str) -> Result<usize, Error> {
        self.encode(text, false, false)
            .map(|encoding| encoding.count)
    }

    pub fn encode(
        &self,
        text: &str,
        include_ids: bool,
        include_tokens: bool,
    ) -> Result<RawTextEncoding, Error> {
        let encoding = match &self.codec {
            Codec::HuggingFace(tokenizer) => {
                let result = tokenizer
                    .encode_result(text.into(), self.policy.add_special_tokens, !include_tokens)
                    .map_err(Error::from)?;
                RawTextEncoding {
                    count: result.len(),
                    token_ids: include_ids.then(|| result.get_ids().to_vec()),
                    tokens: include_tokens.then(|| result.get_tokens().to_vec()),
                }
            }
            Codec::Tiktoken(tokenizer) => {
                if include_tokens {
                    return Err(Error::RawTextConfig("tokens"));
                }
                let ids = match self.policy.special_text {
                    SpecialTextPolicy::Literal => tokenizer.encode(text),
                    SpecialTextPolicy::Recognize => {
                        let allowed: Vec<String> = tokenizer.special_tokens().into_iter().collect();
                        tokenizer
                            .encode_special(text, &allowed)
                            .map_err(|error| Error::Encode(std::io::Error::other(error).into()))?
                    }
                };
                RawTextEncoding {
                    count: ids.len(),
                    token_ids: include_ids.then_some(ids),
                    tokens: None,
                }
            }
        };
        Ok(encoding)
    }
}

#[cfg(test)]
mod tests {
    use base64::{Engine, engine::general_purpose::STANDARD};
    use rstest::rstest;
    use serde_json::Value;
    use sha2::{Digest, Sha256};

    use super::*;

    const HF_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
    ));

    fn hugging_face_config(
        special_text: Option<SpecialTextPolicy>,
        add_special_tokens: Option<bool>,
    ) -> LocalTokenizerConfig {
        hugging_face_config_for_artifact(HF_JSON.into(), special_text, add_special_tokens)
    }

    fn hugging_face_config_for_artifact(
        content: String,
        special_text: Option<SpecialTextPolicy>,
        add_special_tokens: Option<bool>,
    ) -> LocalTokenizerConfig {
        LocalTokenizerConfig {
            id: "test-hf".into(),
            revision: "test-revision".into(),
            engine: RawTextEngine::HuggingFace,
            artifact: RawTextArtifact::Inline { content },
            special_text,
            add_special_tokens,
        }
    }

    fn hugging_face_artifact_with_dropout(dropout: Option<f32>) -> String {
        let mut artifact: Value = serde_json::from_str(HF_JSON).unwrap();
        let model = artifact
            .get_mut("model")
            .and_then(Value::as_object_mut)
            .unwrap();
        match dropout {
            Some(dropout) => {
                model.insert("dropout".into(), serde_json::json!(dropout));
            }
            None => {
                model.remove("dropout");
            }
        }
        serde_json::to_string(&artifact).unwrap()
    }

    fn byte_rank_content(offset: u8) -> String {
        (0u16..=255)
            .map(|byte| {
                let byte = byte as u8;
                format!("{} {}", STANDARD.encode([byte]), byte.wrapping_add(offset))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn tiktoken_config(
        rank_content: String,
        special_text: Option<SpecialTextPolicy>,
        add_special_tokens: Option<bool>,
    ) -> LocalTokenizerConfig {
        LocalTokenizerConfig {
            id: "test-tiktoken".into(),
            revision: "test-revision".into(),
            engine: RawTextEngine::Tiktoken {
                encoding: "cl100k_base".into(),
            },
            artifact: RawTextArtifact::Inline {
                content: rank_content,
            },
            special_text,
            add_special_tokens,
        }
    }

    #[rstest]
    #[case::empty("")]
    #[case::unicode("café 漢字 ع ३ 🙂 ＡﬁⅣ")]
    #[case::long_merge("hello hello hello hello hello hello hello hello")]
    fn hugging_face_count_and_representations_share_exact_policy(#[case] text: &str) {
        let tokenizer = RawTextTokenizer::load(hugging_face_config(None, None)).unwrap();
        let count = tokenizer.count(text).unwrap();
        let encoded = tokenizer.encode(text, true, true).unwrap();

        assert_eq!(count, encoded.count);
        assert_eq!(encoded.token_ids.as_ref().unwrap().len(), encoded.count);
        assert_eq!(encoded.tokens.as_ref().unwrap().len(), encoded.count);
    }

    #[rstest]
    #[case::fullwidth_nfkc("Ａ", "A")]
    #[case::compatibility_ligature("ﬁ", "fi")]
    fn hugging_face_artifact_normalizer_is_preserved(#[case] left: &str, #[case] right: &str) {
        let tokenizer = RawTextTokenizer::load(hugging_face_config(None, Some(false))).unwrap();

        assert_eq!(
            tokenizer.encode(left, true, false).unwrap().token_ids,
            tokenizer.encode(right, true, false).unwrap().token_ids
        );
    }

    #[rstest]
    fn hugging_face_special_text_policy_controls_added_token_recognition() {
        let recognized = RawTextTokenizer::load(hugging_face_config(
            Some(SpecialTextPolicy::Recognize),
            Some(false),
        ))
        .unwrap();
        let literal = RawTextTokenizer::load(hugging_face_config(
            Some(SpecialTextPolicy::Literal),
            Some(false),
        ))
        .unwrap();
        let text = "<SOS>";

        assert_eq!(recognized.count(text).unwrap(), 1);
        assert!(literal.count(text).unwrap() > recognized.count(text).unwrap());
        assert_eq!(
            recognized
                .encode(text, false, true)
                .unwrap()
                .tokens
                .unwrap(),
            vec!["<SOS>"]
        );
    }

    #[rstest]
    fn engine_defaults_resolve_once_and_remain_explicit() {
        let hugging_face = hugging_face_config(None, None).resolved_policy().unwrap();
        let tiktoken = tiktoken_config(byte_rank_content(0), None, None)
            .resolved_policy()
            .unwrap();

        assert_eq!(
            hugging_face,
            EffectiveRawTextPolicy {
                special_text: SpecialTextPolicy::Recognize,
                add_special_tokens: true,
            }
        );
        assert_eq!(
            tiktoken,
            EffectiveRawTextPolicy {
                special_text: SpecialTextPolicy::Literal,
                add_special_tokens: false,
            }
        );
    }

    #[rstest]
    fn hugging_face_postprocessor_truncation_and_padding_are_preserved() {
        let mut tokenizer: tokenizers::Tokenizer = HF_JSON.parse().unwrap();
        let post_processor = tokenizers::processors::template::TemplateProcessing::builder()
            .try_single("$A <EOT>")
            .unwrap()
            .special_tokens(vec![("<EOT>", 0)])
            .build()
            .unwrap();
        tokenizer.with_post_processor(Some(post_processor));
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: 8,
                ..Default::default()
            }))
            .unwrap();
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(8),
            ..Default::default()
        }));
        let config = LocalTokenizerConfig {
            artifact: RawTextArtifact::Inline {
                content: tokenizer.to_string(false).unwrap(),
            },
            ..hugging_face_config(None, None)
        };
        let loaded = RawTextTokenizer::load(config).unwrap();
        let normalized = loaded.encode("Ａ hello world", true, true).unwrap();
        let equivalent = loaded.encode("A hello world", true, true).unwrap();

        assert_eq!(normalized.token_ids, equivalent.token_ids);
        assert_eq!(normalized.count, 8);
        assert!(normalized.tokens.unwrap().contains(&"<EOT>".into()));
    }

    #[rstest]
    fn tiktoken_special_recognition_is_independent_of_added_tokens() {
        let ranks = byte_rank_content(0);
        let literal = RawTextTokenizer::load(tiktoken_config(
            ranks.clone(),
            Some(SpecialTextPolicy::Literal),
            None,
        ))
        .unwrap();
        let recognized = RawTextTokenizer::load(tiktoken_config(
            ranks,
            Some(SpecialTextPolicy::Recognize),
            None,
        ))
        .unwrap();

        assert_eq!(recognized.count("<|endoftext|>").unwrap(), 1);
        assert!(literal.count("<|endoftext|>").unwrap() > 1);
        assert!(recognized.encode("word", false, true).is_err());
    }

    #[rstest]
    #[case::unicode("café 漢字 🙂")]
    #[case::empty("")]
    #[case::long_merge("hello hello hello hello hello hello hello hello")]
    fn tiktoken_encode_length_matches_count_in_each_policy(#[case] text: &str) {
        let tokenizer = RawTextTokenizer::load(tiktoken_config(
            byte_rank_content(0),
            Some(SpecialTextPolicy::Literal),
            None,
        ))
        .unwrap();
        let encoded = tokenizer.encode(text, true, false).unwrap();

        assert_eq!(tokenizer.count(text).unwrap(), encoded.count);
        assert_eq!(encoded.token_ids.unwrap().len(), encoded.count);
    }

    #[rstest]
    fn tiktoken_artifact_digest_and_policy_are_part_of_identity() {
        let first =
            RawTextTokenizer::load(tiktoken_config(byte_rank_content(0), None, None)).unwrap();
        let second =
            RawTextTokenizer::load(tiktoken_config(byte_rank_content(1), None, None)).unwrap();
        let different_policy = RawTextTokenizer::load(tiktoken_config(
            byte_rank_content(0),
            Some(SpecialTextPolicy::Recognize),
            None,
        ))
        .unwrap();

        assert_ne!(first.identity().sha256, second.identity().sha256);
        assert_ne!(first.policy(), different_policy.policy());
    }

    #[rstest]
    fn file_artifacts_require_a_matching_sha256() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.json");
        std::fs::write(&path, HF_JSON).unwrap();
        let digest = format!("{:x}", Sha256::digest(HF_JSON.as_bytes()));
        let config = LocalTokenizerConfig {
            artifact: RawTextArtifact::File {
                path: path.clone(),
                sha256: digest,
            },
            ..hugging_face_config(None, None)
        };
        let loaded = RawTextTokenizer::load(config).unwrap();

        assert_eq!(
            loaded.identity().sha256,
            format!("{:x}", Sha256::digest(HF_JSON))
        );
        let mismatch = LocalTokenizerConfig {
            artifact: RawTextArtifact::File {
                path,
                sha256: "0".repeat(64),
            },
            ..hugging_face_config(None, None)
        };

        assert!(matches!(
            RawTextTokenizer::load(mismatch),
            Err(Error::ArtifactHashMismatch)
        ));
    }

    #[rstest]
    fn tokenizer_configuration_rejects_unknown_fields_and_tiktoken_additions() {
        assert!(
            serde_json::from_str::<LocalTokenizerConfig>(
                r#"{"id":"x","revision":"r","engine":{"type":"hugging_face"},"artifact":{"type":"inline","content":"{}"},"unexpected":true}"#
            )
            .is_err()
        );
        assert!(
            tiktoken_config(byte_rank_content(0), None, Some(true))
                .resolved_policy()
                .is_err()
        );
    }

    #[rstest]
    #[case::absent(None, true)]
    #[case::zero(Some(0.0), true)]
    #[case::positive(Some(0.1), false)]
    fn hugging_face_dropout_is_rejected_only_for_local_loading(
        #[case] dropout: Option<f32>,
        #[case] local_is_accepted: bool,
    ) {
        let artifact = hugging_face_artifact_with_dropout(dropout);
        let legacy = crate::huggingface::HuggingFaceTokenizer::from_json(&artifact).unwrap();
        assert!(legacy.count_tokens("hello").unwrap() > 0);

        let result = RawTextTokenizer::load(hugging_face_config_for_artifact(artifact, None, None));
        assert_eq!(result.is_ok(), local_is_accepted);
        if !local_is_accepted {
            assert!(matches!(result, Err(Error::RawTextConfig("model.dropout"))));
        }
    }
}
