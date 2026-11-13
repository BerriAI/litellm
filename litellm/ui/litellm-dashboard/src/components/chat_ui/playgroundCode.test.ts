import { describe, expect, it } from "vitest";
import { EndpointType } from "./mode_endpoint_mapping";
import { generatePlaygroundCode } from "./playgroundCode";

describe("Playground customer snippets", () => {
  it.each([EndpointType.CHAT, EndpointType.RESPONSES, EndpointType.ANTHROPIC_MESSAGES])(
    "includes customer headers for %s in Python and curl",
    (endpoint) => {
      for (const sdk of ["openai", "curl"] as const) {
        const options = {
          endpoint,
          sdk,
          model: "model",
          prompt: "Say 'hello'",
          headers: {
            "x-litellm-customer-org-alias": "customer-alias",
            "x-litellm-customer-member-id": "customer-member",
            Authorization: "Bearer do-not-copy",
            "x-api-key": "do-not-copy",
          },
          baseUrl: "https://proxy.example/gateway/v1/",
        };
        const code = generatePlaygroundCode(options);
        expect(code).toContain("x-litellm-customer-org-alias");
        expect(code).toContain("x-litellm-customer-member-id");
        expect(code).toContain("LITELLM_API_KEY");
        expect(code).not.toContain("do-not-copy");
        expect(code).toContain("https://proxy.example/gateway");
        expect(code).not.toContain("/v1/v1");
        if (endpoint === EndpointType.ANTHROPIC_MESSAGES && sdk !== "curl")
          expect(code).toContain("anthropic.Anthropic");
      }
    },
  );

  it("formats Python SDK calls as keyword arguments", () => {
    const options = {
      endpoint: EndpointType.ANTHROPIC_MESSAGES,
      sdk: "anthropic",
      model: "qa-claude",
      prompt: "Say hello",
      headers: { "x-litellm-customer-org-alias": "customer-alias" },
      baseUrl: "http://localhost:3000",
    } as const;
    const code = generatePlaygroundCode(options);

    expect(code).toContain('default_headers={\n        "x-litellm-customer-org-alias": "customer-alias"\n    }');
    expect(code).toContain(
      'client.messages.create(\n    model="qa-claude",\n    max_tokens=1024,\n    messages=[\n        {\n            "role": "user",\n            "content": "Say hello"\n        }\n    ]\n)',
    );
    expect(code).not.toContain("**{");
  });

  it("formats curl headers on separate continuation lines", () => {
    const options = {
      endpoint: EndpointType.CHAT,
      sdk: "curl",
      model: "model",
      prompt: "Say hello",
      headers: {
        "x-litellm-customer-org-alias": "customer-alias",
        "x-litellm-customer-member-id": "customer-member",
      },
      baseUrl: "http://localhost:3000",
    } as const;
    const code = generatePlaygroundCode(options);

    expect(code).toContain("  -H 'x-litellm-customer-org-alias: customer-alias' \\\n");
    expect(code).toContain("  -H 'x-litellm-customer-member-id: customer-member' \\\n");
  });
});
