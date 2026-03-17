import { describe, it, expect, vi, beforeEach } from "vitest";
import { getProviderDisplayInfo, getProviderBackendValue, handleImageError } from "./provider_display_helpers";

vi.mock("../provider_info_helpers", () => ({
  Providers: {
    OpenAI: "OpenAI",
    Anthropic: "Anthropic",
    Azure: "Azure",
  },
  provider_map: {
    OpenAI: "openai",
    Anthropic: "anthropic",
    Azure: "azure",
  },
  providerLogoMap: {
    OpenAI: "https://example.com/openai.png",
    Anthropic: "https://example.com/anthropic.png",
    Azure: "https://example.com/azure.png",
  },
}));

describe("getProviderDisplayInfo", () => {
  it("should return display name and logo for a known backend provider value", () => {
    const info = getProviderDisplayInfo("openai");
    expect(info.displayName).toBe("OpenAI");
    expect(info.logo).toBe("https://example.com/openai.png");
    expect(info.enumKey).toBe("OpenAI");
  });

  it("should return the raw value as display name for an unknown provider", () => {
    const info = getProviderDisplayInfo("my-custom-provider");
    expect(info.displayName).toBe("my-custom-provider");
    expect(info.logo).toBe("");
    expect(info.enumKey).toBeNull();
  });

  it("should match a provider by its backend value regardless of casing", () => {
    const info = getProviderDisplayInfo("anthropic");
    expect(info.displayName).toBe("Anthropic");
    expect(info.enumKey).toBe("Anthropic");
  });
});

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

describe("handleImageError", () => {
  it("should replace the img element with a fallback div showing the first letter", () => {
    const img = document.createElement("img");
    const parent = document.createElement("div");
    parent.appendChild(img);

    const event = { target: img } as any;
    handleImageError(event, "OpenAI");

    expect(parent.querySelector("img")).toBeNull();
    const fallback = parent.firstChild as HTMLElement;
    expect(fallback.tagName).toBe("DIV");
    expect(fallback.textContent).toBe("O");
  });

  it("should use the first character of the fallback text as the label", () => {
    const img = document.createElement("img");
    const parent = document.createElement("div");
    parent.appendChild(img);

    const event = { target: img } as any;
    handleImageError(event, "Anthropic");

    const fallback = parent.firstChild as HTMLElement;
    expect(fallback.textContent).toBe("A");
  });

  it("should do nothing if the image has no parent element", () => {
    const img = document.createElement("img");
    const event = { target: img } as any;
    // Should not throw
    expect(() => handleImageError(event, "OpenAI")).not.toThrow();
  });
});
