export type TeamGuardrailStatus = "active" | "pending" | "rejected";

export type TeamGuardrail = {
  id: string;
  team: string;
  name: string;
  endpoint: string;
  status: TeamGuardrailStatus;
  model: string;
  forwardKey: boolean;
  description: string;
  method: "POST" | "GET";
  customHeaders: {
    key: string;
    value: string;
  }[];
  extraHeaders: string[];
  submittedAt: string;
  submittedBy: string;
  mode?: string;
  unreachable_fallback?: string;
  additionalProviderParams?: Record<string, unknown>;
  guardrailType?: string;
};

const FAIL_OPEN_BY_DEFAULT_GUARDRAILS: ReadonlySet<string> = new Set(["agent_365", "typesafe"]);

export function defaultUnreachableFallback(guardrailType: string | undefined): "fail_open" | "fail_closed" {
  return guardrailType !== undefined && FAIL_OPEN_BY_DEFAULT_GUARDRAILS.has(guardrailType)
    ? "fail_open"
    : "fail_closed";
}

function unreachableFallbackLine(g: TeamGuardrail): string {
  const hint = "fail_closed blocks, fail_open proceeds when the guardrail endpoint is unreachable";
  if (g.unreachable_fallback !== undefined) {
    return `        unreachable_fallback: ${g.unreachable_fallback}  # ${hint}`;
  }
  return `        unreachable_fallback: ${defaultUnreachableFallback(g.guardrailType)}  # ${hint}. Shown value is this guardrail's default.`;
}

export function buildEquivalentConfigYaml(g: TeamGuardrail): string {
  const lines: string[] = [
    "litellm_settings:",
    "  guardrails:",
    `    - guardrail_name: "${g.name.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`,
    "      litellm_params:",
    `        guardrail: ${g.guardrailType ?? "generic_guardrail_api"}`,
    `        mode: ${g.mode ?? "pre_call"}  # or post_call, during_call`,
    `        api_base: ${g.endpoint || "https://your-guardrail-api.com"}`,
    "        api_key: os.environ/YOUR_GUARDRAIL_API_KEY  # optional",
    unreachableFallbackLine(g),
    `        forward_api_key: ${g.forwardKey}`,
  ];
  if (g.model && g.model !== "—") {
    lines.push(`        model: "${g.model}"  # LLM model name sent to the guardrail for context`);
  }
  if (g.customHeaders.length > 0) {
    lines.push("        headers:  # static headers (sent with every request)");
    for (const h of g.customHeaders) {
      lines.push(`          ${h.key}: "${String(h.value).replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`);
    }
  }
  if (g.extraHeaders.length > 0) {
    lines.push("        extra_headers:  # forward these client request headers to the guardrail");
    for (const name of g.extraHeaders) {
      lines.push(`          - ${name}`);
    }
  }
  if (g.additionalProviderParams && Object.keys(g.additionalProviderParams).length > 0) {
    lines.push("        additional_provider_specific_params:");
    for (const [k, v] of Object.entries(g.additionalProviderParams)) {
      const val = typeof v === "string" ? `"${v}"` : String(v);
      lines.push(`          ${k}: ${val}`);
    }
  }
  return lines.join("\n");
}
