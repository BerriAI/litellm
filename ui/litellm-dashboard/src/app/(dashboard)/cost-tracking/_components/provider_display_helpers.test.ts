import { describe, it, expect, vi } from "vitest";
import { getProviderBackendValue } from "./provider_display_helpers";

vi.mock("@/components/provider_info_helpers", () => ({
  provider_map: {
    OpenAI: "openai",
    Anthropic: "anthropic",
    Azure: "azure",
  },
}));

describe("getProviderBackendValue", () => {
  it("should return the backend value for a known provider enum key", () => {
    expect(getProviderBackendValue("OpenAI")).toBe("openai");
  });

  it("should return the backend value for another known provider", () => {
    expect(getProviderBackendValue("Anthropic")).toBe("anthropic");
  });

  it("should return null for an unknown enum key", () => {
    expect(getProviderBackendValue("UnknownProvider")).toBeNull();
  });
});
