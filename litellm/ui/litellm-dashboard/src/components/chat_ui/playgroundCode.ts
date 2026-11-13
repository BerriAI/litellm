import { EndpointType } from "./mode_endpoint_mapping";
import type { CustomHeaders } from "@/components/llm_calls/request_headers";

export type PlaygroundCodeSdk = "openai" | "azure" | "anthropic" | "curl";

export const supportsCustomerEndpoint = (endpoint: EndpointType): boolean =>
  [EndpointType.CHAT, EndpointType.RESPONSES, EndpointType.ANTHROPIC_MESSAGES].includes(endpoint);

const shellQuote = (value: string): string => `'${value.replace(/'/g, "'\\''")}'`;

const pythonValue = (value: unknown, indentation: number): string => {
  const rendered = JSON.stringify(value, null, 4);
  return rendered
    .split("\n")
    .map((line, index) => (index === 0 ? line : `${" ".repeat(indentation)}${line}`))
    .join("\n");
};

const safeSnippetHeaders = (headers: CustomHeaders): CustomHeaders =>
  Object.fromEntries(
    Object.entries(headers).filter(
      ([name]) =>
        !["authorization", "x-api-key", "api-key", "proxy-authorization", "cookie"].includes(name.trim().toLowerCase()),
    ),
  );

const requestFor = (endpoint: EndpointType, model: string, prompt: string) => {
  const messages = [{ role: "user", content: prompt }];
  switch (endpoint) {
    case EndpointType.RESPONSES:
      return { body: { model, input: prompt }, path: "responses", method: "responses.create" };
    case EndpointType.ANTHROPIC_MESSAGES:
      return { body: { model, max_tokens: 1024, messages }, path: "messages", method: "messages.create" };
    default:
      return { body: { model, messages }, path: "chat/completions", method: "chat.completions.create" };
  }
};

export function generatePlaygroundCode({
  endpoint,
  sdk,
  model,
  prompt,
  headers,
  baseUrl,
  metadata = {},
}: {
  endpoint: EndpointType;
  sdk: PlaygroundCodeSdk;
  model: string;
  prompt: string;
  headers: CustomHeaders;
  baseUrl: string;
  metadata?: Readonly<Record<string, readonly string[]>>;
}): string {
  const base = baseUrl.trim().replace(/\/+$/, "").replace(/\/v1$/, "");
  const customerHeaders = safeSnippetHeaders(headers);
  const { body, path, method } = requestFor(endpoint, model, prompt);
  const metadataBody = Object.keys(metadata).length > 0 ? { metadata } : {};
  if (sdk === "curl") {
    return [
      `curl ${shellQuote(`${base}/v1/${path}`)} \\`,
      '  -H "Authorization: Bearer ${LITELLM_API_KEY}" \\',
      "  -H 'Content-Type: application/json' \\",
      ...(endpoint === EndpointType.ANTHROPIC_MESSAGES ? ["  -H 'anthropic-version: 2023-06-01' \\"] : []),
      ...Object.entries(customerHeaders).map(([name, value]) => `  -H ${shellQuote(`${name}: ${value}`)} \\`),
      `  -d ${shellQuote(JSON.stringify({ ...body, ...metadataBody }, null, 2))}`,
    ].join("\n");
  }
  const sdkBody = { ...body, ...(Object.keys(metadata).length > 0 ? { extra_body: metadataBody } : {}) };
  const headerCode =
    Object.keys(customerHeaders).length > 0 ? `,\n    default_headers=${pythonValue(customerHeaders, 4)}` : "";
  const anthropic = endpoint === EndpointType.ANTHROPIC_MESSAGES;
  const openaiClient = sdk === "azure" ? "openai.AzureOpenAI" : "openai.OpenAI";
  const client = anthropic ? "anthropic.Anthropic" : openaiClient;
  const sdkBase = anthropic ? base : `${base}/v1`;
  const endpointCode =
    !anthropic && sdk === "azure"
      ? `azure_endpoint=${JSON.stringify(base)},\n    api_version="2024-02-01"`
      : `base_url=${JSON.stringify(sdkBase)}`;
  const argumentsCode = Object.entries(sdkBody)
    .map(([key, value]) => `    ${key}=${pythonValue(value, 4)}`)
    .join(",\n");
  return `import os
import ${anthropic ? "anthropic" : "openai"}

client = ${client}(
    api_key=os.environ["LITELLM_API_KEY"],
    ${endpointCode}${headerCode}
)

response = client.${method}(
${argumentsCode}
)
print(response)
`;
}
