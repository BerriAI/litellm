import { execFileSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { buildCurlCommand, curlPayload } from "./curlCommand";

describe("buildCurlCommand", () => {
  it("targets the proxy endpoint with a key placeholder and the pretty payload", () => {
    const input = {
      baseUrl: "http://localhost:4000/",
      endpoint: "/v1/systemone",
      payload: JSON.stringify({ model: "jev-latest", state: "it's broken" }),
      authHeaderName: "Authorization",
    };
    const command = buildCurlCommand(input);

    expect(command).toContain("curl -X POST 'http://localhost:4000/v1/systemone'");
    expect(command).toContain('-H "Authorization: Bearer $LITELLM_API_KEY"');
    expect(command).not.toContain("sk-");
    expect(command).toContain('  "state": "it\'\\\'\'s broken"');
    expect(JSON.parse(curlPayload(command) ?? "")).toEqual({ model: "jev-latest", state: "it's broken" });
  });

  it("quotes a proxy URL that carries shell metacharacters", () => {
    const input = {
      baseUrl: "http://proxy.example/$(touch pwned)'; echo x",
      endpoint: "/v1/decisions",
      payload: "{}",
      authHeaderName: "Authorization",
    };
    const command = buildCurlCommand(input);
    expect(command).toContain(`curl -X POST 'http://proxy.example/$(touch pwned)'\\''; echo x/v1/decisions'`);
  });

  it("sends the key under the proxy's configured auth header name", () => {
    const input = {
      baseUrl: "https://proxy.example",
      endpoint: "/v1/systemone",
      payload: "{}",
      authHeaderName: "X-Litellm-Key",
    };
    const command = buildCurlCommand(input);
    expect(command).toContain('-H "X-Litellm-Key: Bearer $LITELLM_API_KEY"');
    expect(command).not.toContain("Authorization");
  });

  it("lets the shell expand the key while a hostile header name stays literal", () => {
    const input = {
      baseUrl: "https://proxy.example",
      endpoint: "/v1/systemone" as const,
      payload: "{}",
      authHeaderName: 'X-"$(touch pwned)`id`\\Key',
    };
    const command = buildCurlCommand(input);
    const args = execFileSync("bash", ["-c", command.replace(/^curl /, "printf '%s\\n' ")], {
      encoding: "utf8",
      env: { ...process.env, LITELLM_API_KEY: "sk-test-key" },
    });
    expect(args).toContain('X-"$(touch pwned)`id`\\Key: Bearer sk-test-key\n');
    expect(args).not.toContain("$LITELLM_API_KEY");
  });

  it("passes a payload that is not JSON through untouched", () => {
    const input = {
      baseUrl: "https://proxy.example",
      endpoint: "/v1/decisions",
      payload: "{",
      authHeaderName: "Authorization",
    };
    const command = buildCurlCommand(input);
    expect(command).toContain("'https://proxy.example/v1/decisions'");
    expect(curlPayload(command)).toBe("{");
  });
});
