import { describe, expect, it } from "vitest";
import { buildCurlCommand, curlPayload } from "./curlCommand";

describe("buildCurlCommand", () => {
  it("targets the proxy endpoint with a key placeholder and the pretty payload", () => {
    const command = buildCurlCommand({
      baseUrl: "http://localhost:4000/",
      endpoint: "/v1/systemone",
      payload: JSON.stringify({ model: "jev-latest", state: "it's broken" }),
    });

    expect(command).toContain("curl -X POST 'http://localhost:4000/v1/systemone'");
    expect(command).toContain('-H "Authorization: Bearer $LITELLM_API_KEY"');
    expect(command).not.toContain("sk-");
    expect(command).toContain('  "state": "it\'\\\'\'s broken"');
    expect(JSON.parse(curlPayload(command) ?? "")).toEqual({ model: "jev-latest", state: "it's broken" });
  });

  it("quotes a proxy URL that carries shell metacharacters", () => {
    const command = buildCurlCommand({
      baseUrl: "http://proxy.example/$(touch pwned)'; echo x",
      endpoint: "/v1/decisions",
      payload: "{}",
    });
    expect(command).toContain(`curl -X POST 'http://proxy.example/$(touch pwned)'\\''; echo x/v1/decisions'`);
  });

  it("passes a payload that is not JSON through untouched", () => {
    const command = buildCurlCommand({ baseUrl: "https://proxy.example", endpoint: "/v1/decisions", payload: "{" });
    expect(command).toContain("'https://proxy.example/v1/decisions'");
    expect(curlPayload(command)).toBe("{");
  });
});
