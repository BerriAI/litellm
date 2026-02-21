export interface GuardrailPreset {
  provider: string;
  categoryName?: string;
  guardrailNameSuggestion: string;
  mode: string;
  defaultOn: boolean;
}

export const GUARDRAIL_PRESETS: Record<string, GuardrailPreset> = {
  // ── LiteLLM Content Filter: Content Categories ──
  cf_denied_financial: {
    provider: "LitellmContentFilter",
    categoryName: "denied_financial_advice",
    guardrailNameSuggestion: "Denied Financial Advice",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_denied_legal: {
    provider: "LitellmContentFilter",
    categoryName: "denied_legal_advice",
    guardrailNameSuggestion: "Denied Legal Advice",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_denied_medical: {
    provider: "LitellmContentFilter",
    categoryName: "denied_medical_advice",
    guardrailNameSuggestion: "Denied Medical Advice",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_denied_insults: {
    provider: "LitellmContentFilter",
    categoryName: "denied_insults",
    guardrailNameSuggestion: "Insults & Personal Attacks",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_harmful_violence: {
    provider: "LitellmContentFilter",
    categoryName: "harmful_violence",
    guardrailNameSuggestion: "Harmful Violence",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_harmful_self_harm: {
    provider: "LitellmContentFilter",
    categoryName: "harmful_self_harm",
    guardrailNameSuggestion: "Harmful Self-Harm",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_harmful_child_safety: {
    provider: "LitellmContentFilter",
    categoryName: "harmful_child_safety",
    guardrailNameSuggestion: "Harmful Child Safety",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_harmful_illegal_weapons: {
    provider: "LitellmContentFilter",
    categoryName: "harmful_illegal_weapons",
    guardrailNameSuggestion: "Harmful Illegal Weapons",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_bias_gender: {
    provider: "LitellmContentFilter",
    categoryName: "bias_gender",
    guardrailNameSuggestion: "Bias: Gender",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_bias_racial: {
    provider: "LitellmContentFilter",
    categoryName: "bias_racial",
    guardrailNameSuggestion: "Bias: Racial",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_bias_religious: {
    provider: "LitellmContentFilter",
    categoryName: "bias_religious",
    guardrailNameSuggestion: "Bias: Religious",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_bias_sexual_orientation: {
    provider: "LitellmContentFilter",
    categoryName: "bias_sexual_orientation",
    guardrailNameSuggestion: "Bias: Sexual Orientation",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_prompt_injection_jailbreak: {
    provider: "LitellmContentFilter",
    categoryName: "prompt_injection_jailbreak",
    guardrailNameSuggestion: "Prompt Injection: Jailbreak",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_prompt_injection_data_exfil: {
    provider: "LitellmContentFilter",
    categoryName: "prompt_injection_data_exfiltration",
    guardrailNameSuggestion: "Prompt Injection: Data Exfiltration",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_prompt_injection_sql: {
    provider: "LitellmContentFilter",
    categoryName: "prompt_injection_sql",
    guardrailNameSuggestion: "Prompt Injection: SQL",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_prompt_injection_malicious_code: {
    provider: "LitellmContentFilter",
    categoryName: "prompt_injection_malicious_code",
    guardrailNameSuggestion: "Prompt Injection: Malicious Code",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_prompt_injection_system_prompt: {
    provider: "LitellmContentFilter",
    categoryName: "prompt_injection_system_prompt",
    guardrailNameSuggestion: "Prompt Injection: System Prompt",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_toxic_abuse: {
    provider: "LitellmContentFilter",
    categoryName: "harm_toxic_abuse",
    guardrailNameSuggestion: "Toxic & Abusive Language",
    mode: "pre_call",
    defaultOn: false,
  },

  // ── LiteLLM Content Filter: Patterns & Keywords (no category) ──
  cf_patterns: {
    provider: "LitellmContentFilter",
    guardrailNameSuggestion: "Pattern Matching",
    mode: "pre_call",
    defaultOn: false,
  },
  cf_keywords: {
    provider: "LitellmContentFilter",
    guardrailNameSuggestion: "Keyword Blocking",
    mode: "pre_call",
    defaultOn: false,
  },

  // ── Partner Guardrails ──
  presidio: {
    provider: "PresidioPII",
    guardrailNameSuggestion: "Presidio PII",
    mode: "pre_call",
    defaultOn: false,
  },
  bedrock: {
    provider: "Bedrock",
    guardrailNameSuggestion: "Bedrock Guardrail",
    mode: "pre_call",
    defaultOn: false,
  },
  lakera: {
    provider: "Lakera",
    guardrailNameSuggestion: "Lakera",
    mode: "pre_call",
    defaultOn: false,
  },
  openai_moderation: {
    provider: "OpenaiModeration",
    guardrailNameSuggestion: "OpenAI Moderation",
    mode: "pre_call",
    defaultOn: false,
  },
  google_model_armor: {
    provider: "ModelArmor",
    guardrailNameSuggestion: "Google Cloud Model Armor",
    mode: "pre_call",
    defaultOn: false,
  },
  guardrails_ai: {
    provider: "GuardrailsAi",
    guardrailNameSuggestion: "Guardrails AI",
    mode: "pre_call",
    defaultOn: false,
  },
  zscaler: {
    provider: "ZscalerAiGuard",
    guardrailNameSuggestion: "Zscaler AI Guard",
    mode: "pre_call",
    defaultOn: false,
  },
  panw: {
    provider: "PanwPrismaAirs",
    guardrailNameSuggestion: "PANW Prisma AIRS",
    mode: "pre_call",
    defaultOn: false,
  },
  noma: {
    provider: "Noma",
    guardrailNameSuggestion: "Noma Security",
    mode: "pre_call",
    defaultOn: false,
  },
  aporia: {
    provider: "AporiaAi",
    guardrailNameSuggestion: "Aporia AI",
    mode: "pre_call",
    defaultOn: false,
  },
  aim: {
    provider: "Aim",
    guardrailNameSuggestion: "AIM Guardrail",
    mode: "pre_call",
    defaultOn: false,
  },
  prompt_security: {
    provider: "PromptSecurity",
    guardrailNameSuggestion: "Prompt Security",
    mode: "pre_call",
    defaultOn: false,
  },
  lasso: {
    provider: "Lasso",
    guardrailNameSuggestion: "Lasso Guardrail",
    mode: "pre_call",
    defaultOn: false,
  },
  pangea: {
    provider: "Pangea",
    guardrailNameSuggestion: "Pangea Guardrail",
    mode: "pre_call",
    defaultOn: false,
  },
  enkryptai: {
    provider: "Enkryptai",
    guardrailNameSuggestion: "EnkryptAI",
    mode: "pre_call",
    defaultOn: false,
  },
  javelin: {
    provider: "Javelin",
    guardrailNameSuggestion: "Javelin Guardrails",
    mode: "pre_call",
    defaultOn: false,
  },
  pillar: {
    provider: "Pillar",
    guardrailNameSuggestion: "Pillar Guardrail",
    mode: "pre_call",
    defaultOn: false,
  },
};
