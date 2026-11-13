import { describe, expect, it } from "vitest";
import { EndpointType } from "./mode_endpoint_mapping";
import { generatePlaygroundCode } from "./playgroundCode";

describe("Playground customer snippets", () => {
  it.each([EndpointType.CHAT, EndpointType.RESPONSES, EndpointType.ANTHROPIC_MESSAGES])(
    "includes customer headers for %s in Python and curl",
    (endpoint) => {
      for (const sdk of ["openai", "curl"] as const) {
        const code = generatePlaygroundCode({
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
        });
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
});
