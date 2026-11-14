import { describe, expect, it, vi } from "vitest";

import type { AgentCard, ModelGroupInfo } from "@/components/PublicModelHubTableColumns";

import {
  agentCapabilities,
  capabilityLabel,
  featureFacetLabel,
  firstPlus,
  formatPerMillion,
  formatTokens,
  healthOf,
  mcpAuthLabel,
  modeLabel,
  modelCapabilities,
  relativeTime,
  usefulLinkList,
  wildcardExample,
} from "./hubFormat";

vi.mock("@/components/networking", () => ({ getProxyBaseUrl: () => "https://gateway.example.com" }));

const model = (overrides: Partial<ModelGroupInfo>): ModelGroupInfo => ({
  model_group: "gpt-5",
  providers: ["openai"],
  supports_parallel_function_calling: false,
  supports_vision: false,
  supports_function_calling: false,
  ...overrides,
});

describe("modeLabel", () => {
  it("names known modes in sentence case and falls back for unknown ones", () => {
    expect(modeLabel("image_generation")).toBe("Image generation");
    expect(modeLabel("audio_speech")).toBe("Text to speech");
    expect(modeLabel("custom_thing")).toBe("Custom thing");
    expect(modeLabel("evaluation")).toBe("Decisions");
    expect(modeLabel(undefined)).toBe("-");
  });
});

describe("formatTokens", () => {
  it("abbreviates thousands and millions", () => {
    expect(formatTokens(512)).toBe("512");
    expect(formatTokens(200_000)).toBe("200K");
    expect(formatTokens(1_000_000)).toBe("1M");
    expect(formatTokens(1_048_576)).toBe("1M");
    expect(formatTokens(2_500_000)).toBe("2.5M");
    expect(formatTokens(undefined)).toBe("-");
  });
});

describe("formatPerMillion", () => {
  it("prices per million tokens with at least two decimals and keeps sub-cent prices", () => {
    expect(formatPerMillion(0.000003)).toBe("$3.00");
    expect(formatPerMillion(0.0000165)).toBe("$16.50");
    expect(formatPerMillion(3.75e-8)).toBe("$0.0375");
    expect(formatPerMillion(2e-9)).toBe("$0.002");
    expect(formatPerMillion(0)).toBe("Free");
    expect(formatPerMillion(undefined)).toBe("-");
  });
});

describe("capabilities", () => {
  it("labels known flags and derives a label for new ones", () => {
    expect(capabilityLabel("supports_function_calling")).toBe("Function calling");
    expect(capabilityLabel("supports_pdf_input")).toBe("Pdf input");
    expect(featureFacetLabel("web_search")).toBe("Web search");
  });

  it("lists only enabled flags, known ones first in a fixed order", () => {
    const flags = model({ supports_pdf_input: true, supports_web_search: true, supports_vision: true });
    expect(modelCapabilities(flags)).toEqual(["Vision", "Web search", "Pdf input"]);
  });

  it("lists an agent's enabled capabilities", () => {
    const agent = {
      capabilities: { streaming: true, pushNotifications: true, stateTransitionHistory: true, x: false },
    };
    expect(agentCapabilities(agent as AgentCard)).toEqual([
      "Streaming",
      "Push notifications",
      "State transition history",
    ]);
  });
});

describe("healthOf", () => {
  it("maps the health check to a tone and label", () => {
    expect(healthOf(model({ health_status: "healthy" }))).toEqual({ tone: "success", label: "Healthy" });
    expect(healthOf(model({ health_status: "unhealthy" }))).toEqual({ tone: "error", label: "Unhealthy" });
    expect(healthOf(model({}))).toEqual({ tone: "neutral", label: "Not checked" });
  });
});

describe("relativeTime", () => {
  const now = Date.parse("2026-10-31T12:00:00Z");

  it("rounds to minutes, hours or days", () => {
    expect(relativeTime("2026-10-31T11:59:45Z", now)).toBe("just now");
    expect(relativeTime("2026-10-31T11:45:00Z", now)).toBe("15 min ago");
    expect(relativeTime("2026-10-31T09:00:00Z", now)).toBe("3 h ago");
    expect(relativeTime("2026-10-30T12:00:00Z", now)).toBe("1 day ago");
    expect(relativeTime("2026-10-29T12:00:00Z", now)).toBe("2 days ago");
    expect(relativeTime(undefined, now)).toBe("-");
    expect(relativeTime("not-a-date", now)).toBe("-");
  });
});

describe("small formatters", () => {
  it("shows a wildcard model as a placeholder name", () => {
    expect(wildcardExample("anthropic/*")).toBe("anthropic/<model-name>");
  });

  it("collapses a list to its first item and a count", () => {
    expect(firstPlus(["Anthropic"])).toBe("Anthropic");
    expect(firstPlus(["Anthropic", "Amazon Bedrock", "Vertex AI"])).toBe("Anthropic +2");
    expect(firstPlus([])).toBeUndefined();
  });

  it("names MCP sign-in types", () => {
    expect(mcpAuthLabel("oauth2")).toBe("OAuth");
    expect(mcpAuthLabel("custom_header")).toBe("custom_header");
    expect(mcpAuthLabel(null)).toBe("-");
  });
});

describe("usefulLinkList", () => {
  it("orders the admin's links by index and accepts the legacy string form", () => {
    const links = {
      "Status page": { url: "https://status.example.com", index: 2 },
      "Getting started": { url: "https://docs.example.com", index: 0 },
      Legacy: "https://legacy.example.com",
    };
    expect(usefulLinkList(links).map((link) => link.title)).toEqual(["Getting started", "Legacy", "Status page"]);
    expect(usefulLinkList(undefined)).toEqual([]);
  });
});
