use super::resolve::{Halt, Resolved, checked};
use super::secrets::{Secrets, Source, Value};
use super::tables::ProviderTables;

use Source::{Env, Given, Literal, Secret, SecretStr};

pub(super) struct Call<'a> {
    pub tables: &'a ProviderTables,
    pub secrets: &'a Secrets<'a>,
    pub api_base: Option<&'a str>,
    pub api_key: Option<&'a str>,
}

impl Call<'_> {
    fn chain(&self, given: Option<&str>, rest: &[Source<'_>]) -> Result<Option<Value>, Halt> {
        let sources = std::iter::once(Given(given))
            .chain(rest.iter().copied())
            .collect::<Vec<_>>();
        Ok(self.secrets.first(&sources)?)
    }

    fn base(&self, rest: &[Source<'_>]) -> Result<Option<Value>, Halt> {
        self.chain(self.api_base, rest)
    }

    fn key(&self, names: &[&'static str]) -> Result<Option<Value>, Halt> {
        let rest = names.iter().map(|name| SecretStr(name)).collect::<Vec<_>>();
        self.chain(self.api_key, &rest)
    }

    fn defaulted(&self, env: Source<'_>, default: &str) -> Result<Option<Value>, Halt> {
        self.base(&[env, Literal(default)])
    }

    fn nadir(&self) -> Result<(Option<Value>, Option<Value>), Halt> {
        let default_base = self
            .secrets
            .secret_str("NADIR_API_BASE")?
            .filter(|base| !base.is_empty())
            .unwrap_or_else(|| self.tables.nadir_default_api_base.clone());
        let api_base = self.base(&[Literal(&default_base)])?;
        let trusted = self.api_base.is_none_or(|caller| {
            caller.trim_end_matches('/') == default_base.trim_end_matches('/')
        });
        let env_key = if trusted {
            self.secrets.secret_str("NADIR_API_KEY")?
        } else {
            None
        };
        let env_key = env_key.map(Value::Text);
        let api_key = match self.api_key {
            Some(key) if !key.is_empty() => Some(Value::Text(key.to_owned())),
            _ => env_key,
        };
        Ok((api_base, api_key))
    }

    pub(super) fn openai_compatible(&self, model: &str) -> Result<Resolved, Halt> {
        let (provider, model) = model.split_once('/').ok_or_else(index_error)?;
        if let Some(json) = self.tables.json_provider(provider) {
            let api_base = match self.api_base.filter(|base| !base.is_empty()) {
                Some(base) => base.to_owned(),
                None => match json.api_base_env.as_deref().filter(|env| !env.is_empty()) {
                    Some(env) => self
                        .secrets
                        .secret_str(env)?
                        .filter(|base| !base.is_empty())
                        .unwrap_or_else(|| json.base_url.clone()),
                    None => json.base_url.clone(),
                },
            };
            let api_key = match self.api_key.filter(|key| !key.is_empty()) {
                Some(key) => Some(key.to_owned()),
                None => self.secrets.secret_str(&json.api_key_env)?,
            };
            return Ok(Resolved::new(
                model,
                Some(provider),
                api_key,
                Some(api_base),
            ));
        }
        let provider = if provider == "ai21" && self.tables.lists.ai21_chat_models.contains(model) {
            "ai21_chat"
        } else {
            provider
        };
        let (api_base, dynamic_api_key) = match provider {
            "perplexity" => (
                self.defaulted(
                    SecretStr("PERPLEXITY_API_BASE"),
                    "https://api.perplexity.ai",
                )?,
                self.key(&["PERPLEXITYAI_API_KEY", "PERPLEXITY_API_KEY"])?,
            ),
            "aiohttp_openai" => {
                return Ok(Resolved::new(
                    model,
                    Some("aiohttp_openai"),
                    self.api_key.map(str::to_owned),
                    self.api_base.map(str::to_owned),
                ));
            }
            "anyscale" => (
                self.defaulted(
                    SecretStr("ANYSCALE_API_BASE"),
                    "https://api.endpoints.anyscale.com/v1",
                )?,
                self.key(&["ANYSCALE_API_KEY"])?,
            ),
            "deepinfra" => (
                self.defaulted(
                    SecretStr("DEEPINFRA_API_BASE"),
                    "https://api.deepinfra.com/v1/openai",
                )?,
                self.key(&["DEEPINFRA_API_KEY"])?,
            ),
            "clf_ai_gateway" => (
                self.defaulted(
                    SecretStr("CLF_AI_GATEWAY_API_BASE"),
                    "https://api.clfaigateway.dev/v1",
                )?,
                self.key(&["CLF_AI_GATEWAY_API_KEY"])?,
            ),
            "empower" => (
                self.defaulted(Secret("EMPOWER_API_BASE"), "https://app.empower.dev/api/v1")?,
                self.key(&["EMPOWER_API_KEY"])?,
            ),
            "groq" => (
                self.defaulted(SecretStr("GROQ_API_BASE"), "https://api.groq.com/openai/v1")?,
                self.key(&["GROQ_API_KEY"])?,
            ),
            "nvidia_nim" => (
                self.defaulted(
                    Secret("NVIDIA_NIM_API_BASE"),
                    "https://integrate.api.nvidia.com/v1",
                )?,
                self.key(&["NVIDIA_NIM_API_KEY"])?,
            ),
            "nvidia_riva" => (
                self.base(&[SecretStr("NVIDIA_RIVA_API_BASE")])?,
                self.key(&["NVIDIA_RIVA_API_KEY", "NVIDIA_NIM_API_KEY"])?,
            ),
            "soniox" => (
                self.defaulted(SecretStr("SONIOX_API_BASE"), "https://api.soniox.com")?,
                self.key(&["SONIOX_API_KEY"])?,
            ),
            "cerebras" => (
                self.defaulted(Secret("CEREBRAS_API_BASE"), "https://api.cerebras.ai/v1")?,
                self.key(&["CEREBRAS_API_KEY"])?,
            ),
            "nadir" => self.nadir()?,
            "sambanova" => (
                self.defaulted(Secret("SAMBANOVA_API_BASE"), "https://api.sambanova.ai/v1")?,
                self.key(&["SAMBANOVA_API_KEY"])?,
            ),
            "meta_llama" => (
                self.defaulted(Secret("LLAMA_API_BASE"), "https://api.llama.com/compat/v1")?,
                self.key(&["LLAMA_API_KEY"])?,
            ),
            "nebius" => (
                self.defaulted(Secret("NEBIUS_API_BASE"), "https://api.studio.nebius.ai/v1")?,
                self.key(&["NEBIUS_API_KEY"])?,
            ),
            "ollama" => (
                self.defaulted(Secret("OLLAMA_API_BASE"), "http://localhost:11434")?,
                self.key(&["OLLAMA_API_KEY"])?,
            ),
            "ai21_chat" => (
                self.defaulted(Secret("AI21_API_BASE"), "https://api.ai21.com/studio/v1")?,
                self.key(&["AI21_API_KEY"])?,
            ),
            "volcengine" => (
                self.defaulted(
                    Secret("VOLCENGINE_API_BASE"),
                    "https://ark.cn-beijing.volces.com/api/v3",
                )?,
                self.key(&["VOLCENGINE_API_KEY"])?,
            ),
            "codestral" => (
                self.defaulted(
                    Secret("CODESTRAL_API_BASE"),
                    "https://codestral.mistral.ai/v1",
                )?,
                self.key(&["CODESTRAL_API_KEY"])?,
            ),
            "hosted_vllm" => (
                self.base(&[SecretStr("HOSTED_VLLM_API_BASE")])?,
                self.chain(
                    self.api_key,
                    &[SecretStr("HOSTED_VLLM_API_KEY"), Literal("fake-api-key")],
                )?,
            ),
            "lm_studio" => (
                self.base(&[SecretStr("LM_STUDIO_API_BASE")])?,
                self.chain(
                    self.api_key,
                    &[SecretStr("LM_STUDIO_API_KEY"), Literal("fake-api-key")],
                )?,
            ),
            "deepseek" => (
                self.defaulted(Secret("DEEPSEEK_API_BASE"), "https://api.deepseek.com/beta")?,
                self.key(&["DEEPSEEK_API_KEY"])?,
            ),
            "tencent" => (
                self.defaulted(
                    Secret("TENCENT_API_BASE"),
                    "https://tokenhub-intl.tencentcloudmaas.com/v1",
                )?,
                self.key(&["TENCENT_API_KEY"])?,
            ),
            "fireworks_ai" => (
                self.defaulted(
                    SecretStr("FIREWORKS_API_BASE"),
                    "https://api.fireworks.ai/inference/v1",
                )?,
                self.key(&[
                    "FIREWORKS_API_KEY",
                    "FIREWORKS_AI_API_KEY",
                    "FIREWORKSAI_API_KEY",
                    "FIREWORKS_AI_TOKEN",
                ])?,
            ),
            "github" => (
                self.defaulted(
                    SecretStr("GITHUB_API_BASE"),
                    "https://models.inference.ai.azure.com",
                )?,
                self.key(&["GITHUB_API_KEY"])?,
            ),
            "litellm_proxy" => (
                self.base(&[SecretStr("LITELLM_PROXY_API_BASE")])?,
                self.key(&["LITELLM_PROXY_API_KEY"])?,
            ),
            "jina_ai" => (
                self.defaulted(SecretStr("JINA_AI_API_BASE"), "https://api.jina.ai/v1")?,
                self.key(&["JINA_AI_API_KEY", "JINA_API_KEY", "JINA_AI_TOKEN"])?,
            ),
            "together_ai" => (
                self.defaulted(
                    SecretStr("TOGETHER_AI_API_BASE"),
                    "https://api.together.ai/v1",
                )?,
                self.key(&[
                    "TOGETHER_API_KEY",
                    "TOGETHER_AI_API_KEY",
                    "TOGETHERAI_API_KEY",
                    "TOGETHER_AI_TOKEN",
                ])?,
            ),
            "friendliai" => (
                self.defaulted(
                    Secret("FRIENDLI_API_BASE"),
                    "https://api.friendli.ai/serverless/v1",
                )?,
                self.key(&["FRIENDLIAI_API_KEY", "FRIENDLI_TOKEN"])?,
            ),
            "galadriel" => (
                self.defaulted(Secret("GALADRIEL_API_BASE"), "https://api.galadriel.com/v1")?,
                self.key(&["GALADRIEL_API_KEY"])?,
            ),
            "novita" => (
                self.defaulted(Secret("NOVITA_API_BASE"), "https://api.novita.ai/v3/openai")?,
                self.key(&["NOVITA_API_KEY"])?,
            ),
            "featherless_ai" => (
                self.base(&[
                    SecretStr("FEATHERLESS_AI_API_BASE"),
                    SecretStr("FEATHERLESS_API_BASE"),
                    Literal("https://api.featherless.ai/v1"),
                ])?,
                self.key(&["FEATHERLESS_AI_API_KEY", "FEATHERLESS_API_KEY"])?,
            ),
            "heroku" => (
                self.base(&[Env("HEROKU_API_BASE")])?,
                self.chain(self.api_key, &[Env("HEROKU_API_KEY")])?,
            ),
            "moonshot" => (
                self.defaulted(SecretStr("MOONSHOT_API_BASE"), "https://api.moonshot.ai/v1")?,
                self.key(&["MOONSHOT_API_KEY"])?,
            ),
            "docker_model_runner" => (
                self.defaulted(
                    SecretStr("DOCKER_MODEL_RUNNER_API_BASE"),
                    "http://localhost:22088/engines/llama.cpp",
                )?,
                self.chain(
                    self.api_key,
                    &[
                        SecretStr("DOCKER_MODEL_RUNNER_API_KEY"),
                        Literal("dummy-key"),
                    ],
                )?,
            ),
            "v0" => (
                self.defaulted(SecretStr("V0_API_BASE"), "https://api.v0.dev/v1")?,
                self.key(&["V0_API_KEY"])?,
            ),
            "morph" => (
                self.defaulted(SecretStr("MORPH_API_BASE"), "https://api.morphllm.com/v1")?,
                self.key(&["MORPH_API_KEY"])?,
            ),
            "lambda_ai" => (
                self.defaulted(SecretStr("LAMBDA_API_BASE"), "https://api.lambda.ai/v1")?,
                self.key(&["LAMBDA_API_KEY"])?,
            ),
            "hyperbolic" => (
                self.defaulted(
                    SecretStr("HYPERBOLIC_API_BASE"),
                    "https://api.hyperbolic.xyz/v1",
                )?,
                self.key(&["HYPERBOLIC_API_KEY"])?,
            ),
            "vercel_ai_gateway" => (
                self.defaulted(
                    SecretStr("VERCEL_AI_GATEWAY_API_BASE"),
                    "https://ai-gateway.vercel.sh/v1",
                )?,
                self.key(&["VERCEL_AI_GATEWAY_API_KEY", "VERCEL_OIDC_TOKEN"])?,
            ),
            "wandb" => (
                self.defaulted(
                    Secret("WANDB_API_BASE"),
                    "https://api.inference.wandb.ai/v1",
                )?,
                self.key(&["WANDB_API_KEY"])?,
            ),
            "langgraph" => (
                self.defaulted(SecretStr("LANGGRAPH_API_BASE"), "http://localhost:2024")?,
                self.key(&["LANGGRAPH_API_KEY"])?,
            ),
            "manus" => (
                self.defaulted(SecretStr("MANUS_API_BASE"), "https://api.manus.im")?,
                self.key(&["MANUS_API_KEY"])?,
            ),
            "gigachat" => (
                self.defaulted(
                    SecretStr("GIGACHAT_API_BASE"),
                    "https://gigachat.devices.sberbank.ru/api/v1",
                )?,
                self.key(&["GIGACHAT_API_KEY"])?,
            ),
            "bedrock_mantle" | "baseten" | "llamafile" | "datarobot" | "azure_ai" | "mistral"
            | "xai" | "bigmodel" | "zai" | "github_copilot" | "chatgpt" | "snowflake"
            | "gradient_ai" | "nscale" | "dashscope" | "qwencloud" | "qwen_ai_platform"
            | "modelscope" | "opencode_go" | "inception" | "edenai" | "fal_ai" | "aiml"
            | "lemonade" | "clarifai" | "ragflow" => return Err(Halt::Deferred),
            _ => (self.api_base.map(|base| Value::Text(base.to_owned())), None),
        };
        let (api_base, dynamic_api_key) = checked(api_base, dynamic_api_key)?;
        let dynamic_api_key = dynamic_api_key.or_else(|| self.api_key.map(str::to_owned));
        Ok(Resolved::new(
            model,
            Some(provider),
            dynamic_api_key,
            api_base,
        ))
    }

    pub(super) fn inferred_from_endpoint(
        &self,
        endpoint: &str,
        api_base: &str,
    ) -> Result<Option<(&str, Option<Value>)>, Halt> {
        let secret = |name: &'static str| -> Result<Option<Value>, Halt> {
            Ok(self.secrets.secret_str(name)?.map(Value::Text))
        };
        let tables = self.tables;
        Ok(Some(match endpoint {
            "api.perplexity.ai" => ("perplexity", secret("PERPLEXITYAI_API_KEY")?),
            "api.endpoints.anyscale.com/v1" => ("anyscale", secret("ANYSCALE_API_KEY")?),
            "api.deepinfra.com/v1/openai" => ("deepinfra", secret("DEEPINFRA_API_KEY")?),
            "api.clfaigateway.dev/v1" => ("clf_ai_gateway", self.key(&["CLF_AI_GATEWAY_API_KEY"])?),
            "api.mistral.ai/v1" => ("mistral", secret("MISTRAL_API_KEY")?),
            "api.groq.com/openai/v1" => ("groq", secret("GROQ_API_KEY")?),
            "https://integrate.api.nvidia.com/v1" => ("nvidia_nim", secret("NVIDIA_NIM_API_KEY")?),
            "https://api.cerebras.ai/v1" => ("cerebras", secret("CEREBRAS_API_KEY")?),
            nadir if nadir == tables.nadir_default_api_base => (
                "nadir",
                if api_base.to_lowercase().starts_with("https://") {
                    secret("NADIR_API_KEY")?
                } else {
                    None
                },
            ),
            "https://inference.baseten.co/v1" => ("baseten", secret("BASETEN_API_KEY")?),
            "https://api.sambanova.ai/v1" => ("sambanova", secret("SAMBANOVA_API_KEY")?),
            "https://api.ai21.com/studio/v1" => ("ai21_chat", secret("AI21_API_KEY")?),
            "codestral.mistral.ai/v1/chat/completions" => {
                ("codestral", secret("CODESTRAL_API_KEY")?)
            }
            "codestral.mistral.ai/v1/fim/completions" => {
                ("text-completion-codestral", secret("CODESTRAL_API_KEY")?)
            }
            "app.empower.dev/api/v1" => ("empower", secret("EMPOWER_API_KEY")?),
            "api.deepseek.com/v1" => ("deepseek", secret("DEEPSEEK_API_KEY")?),
            "api.together.ai/v1" | "api.together.xyz/v1" => (
                "together_ai",
                self.key(&[
                    "TOGETHER_API_KEY",
                    "TOGETHER_AI_API_KEY",
                    "TOGETHERAI_API_KEY",
                    "TOGETHER_AI_TOKEN",
                ])?,
            ),
            "ollama.com" => ("ollama", secret("OLLAMA_API_KEY")?),
            "https://api.friendli.ai/serverless/v1" => (
                "friendliai",
                self.secrets
                    .first(&[SecretStr("FRIENDLIAI_API_KEY"), Secret("FRIENDLI_TOKEN")])?,
            ),
            "api.galadriel.com/v1" => ("galadriel", secret("GALADRIEL_API_KEY")?),
            "https://api.llama.com/compat/v1" => ("meta_llama", self.key(&["LLAMA_API_KEY"])?),
            "https://api.featherless.ai/v1" => {
                ("featherless_ai", secret("FEATHERLESS_AI_API_KEY")?)
            }
            nscale if nscale == tables.nscale_api_base_url => return Err(Halt::Deferred),
            "dashscope-intl.aliyuncs.com/compatible-mode/v1" => {
                ("dashscope", secret("DASHSCOPE_API_KEY")?)
            }
            "https://api-inference.modelscope.cn/v1" => {
                ("modelscope", secret("MODELSCOPE_API_KEY")?)
            }
            "api.moonshot.ai/v1" => ("moonshot", secret("MOONSHOT_API_KEY")?),
            "api.minimax.io/anthropic"
            | "api.minimaxi.com/anthropic"
            | "api.minimax.io/v1"
            | "api.minimaxi.com/v1" => ("minimax", secret("MINIMAX_API_KEY")?),
            "platform.publicai.co/v1" => ("publicai", secret("PUBLICAI_API_KEY")?),
            "https://api.synthetic.new/openai/v1" => ("synthetic", secret("SYNTHETIC_API_KEY")?),
            "https://api.stima.tech/v1" => ("apertis", secret("STIMA_API_KEY")?),
            "https://nano-gpt.com/api/v1" => ("nano-gpt", secret("NANOGPT_API_KEY")?),
            "https://api.poe.com/v1" => ("poe", secret("POE_API_KEY")?),
            "https://llm.chutes.ai/v1/" => ("chutes", secret("CHUTES_API_KEY")?),
            "https://api.v0.dev/v1" => ("v0", secret("V0_API_KEY")?),
            "https://api.lambda.ai/v1" => ("lambda_ai", secret("LAMBDA_API_KEY")?),
            "https://api.inceptionlabs.ai/v1" => ("inception", secret("INCEPTION_API_KEY")?),
            "https://api.hyperbolic.xyz/v1" => ("hyperbolic", secret("HYPERBOLIC_API_KEY")?),
            "https://ai-gateway.vercel.sh/v1" => {
                ("vercel_ai_gateway", secret("VERCEL_AI_GATEWAY_API_KEY")?)
            }
            "https://api.edenai.run/v3" => ("edenai", secret("EDENAI_API_KEY")?),
            "https://api.inference.wandb.ai/v1" => ("wandb", secret("WANDB_API_KEY")?),
            "https://pinstripes.io/v1" => ("pinstripes", secret("PINSTRIPES_API_KEY")?),
            "https://api.meta.ai/v1" => ("meta", secret("META_API_KEY")?),
            "https://gigachat.devices.sberbank.ru/api/v1" => {
                ("gigachat", secret("GIGACHAT_API_KEY")?)
            }
            _ => match tables.json_provider_by_base_url(endpoint) {
                Some(json) => (
                    json.slug.as_str(),
                    match self.api_key {
                        Some(key) => Some(Value::Text(key.to_owned())),
                        None => secret_owned(self.secrets, &json.api_key_env)?,
                    },
                ),
                None => return Ok(None),
            },
        }))
    }
}

fn secret_owned(secrets: &Secrets<'_>, name: &str) -> Result<Option<Value>, Halt> {
    Ok(secrets.secret_str(name)?.map(Value::Text))
}

pub(super) fn index_error() -> Halt {
    Halt::Raised("list index out of range".to_owned())
}
