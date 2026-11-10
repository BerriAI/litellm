import { describe, it, expect } from "vitest";
import { transformKeyInfo } from "./transform_key_info";

describe("transformKeyInfo", () => {
  it("should combine key and info fields into a single object", () => {
    const apiResponse = {
      key: "sk-abc123",
      info: {
        token_id: "tok_1",
        key_name: "my-key",
        spend: 10.5,
      },
    };
    const result = transformKeyInfo(apiResponse);
    expect(result).toEqual({
      token: "sk-abc123",
      token_id: "tok_1",
      key_name: "my-key",
      spend: 10.5,
    });
  });

  it("should set the token field from the key property", () => {
    const apiResponse = {
      key: "sk-xyz789",
      info: { key_name: "test" },
    };
    const result = transformKeyInfo(apiResponse);
    expect(result.token).toBe("sk-xyz789");
  });

  it("should preserve all info fields in the result", () => {
    const apiResponse = {
      key: "sk-abc",
      info: {
        token_id: "tok_2",
        key_name: "prod-key",
        spend: 42,
        models: ["gpt-4"],
        team_id: "team-1",
        metadata: { env: "production" },
      },
    };
    const result = transformKeyInfo(apiResponse);
    expect(result.token_id).toBe("tok_2");
    expect(result.key_name).toBe("prod-key");
    expect(result.spend).toBe(42);
    expect(result.models).toEqual(["gpt-4"]);
    expect(result.team_id).toBe("team-1");
    expect(result.metadata).toEqual({ env: "production" });
  });

  it("should handle empty info object", () => {
    const apiResponse = {
      key: "sk-empty",
      info: {},
    };
    const result = transformKeyInfo(apiResponse);
    expect(result.token).toBe("sk-empty");
    expect(Object.keys(result)).toContain("token");
  });
});
