/**
 * Best-effort provider for a model name, used only to pick a logo. The usage
 * aggregate does not carry a provider per model, so this reads the
 * `provider/model` prefix when there is one and otherwise matches well-known
 * model families. Unknown names return null and render a neutral monogram.
 */

const FAMILY_PROVIDERS: readonly (readonly [RegExp, string])[] = [
  [/claude|anthropic/, "anthropic"],
  [/^(gpt|o\d|chatgpt|text-embedding|dall-e|whisper|tts-|davinci|babbage|omni-moderation)/, "openai"],
  [/gemini|gemma|palm|bison|imagen|veo/, "gemini"],
  [/llama|meta-/, "meta_llama"],
  [/mistral|mixtral|codestral|ministral|pixtral|magistral/, "mistral"],
  [/command|cohere|embed-(english|multilingual)/, "cohere"],
  [/deepseek/, "deepseek"],
  [/grok/, "xai"],
  [/qwen|qwq/, "dashscope"],
  [/kimi|moonshot/, "moonshot"],
  [/sonar|pplx/, "perplexity"],
  [/minimax|abab/, "minimax"],
  [/glm|chatglm/, "zai"],
  [/titan|nova-|amazon\./, "bedrock"],
  [/voyage/, "voyage"],
  [/jina/, "jina_ai"],
];

export const providerForModel = (model: string): string | null => {
  const name = model.trim().toLowerCase();
  const slash = name.indexOf("/");
  if (slash > 0) return name.slice(0, slash);
  return FAMILY_PROVIDERS.find(([pattern]) => pattern.test(name))?.[1] ?? null;
};
