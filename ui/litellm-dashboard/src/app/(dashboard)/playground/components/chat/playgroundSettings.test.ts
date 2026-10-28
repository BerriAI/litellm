import { describe, expect, it } from "vitest";
import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import { OPEN_AI_VOICES } from "../chat_ui/chatConstants";
import {
  ALL_MCP_SERVERS,
  allowedTools,
  defaultSettings,
  loadSettings,
  storedValues,
  toggleMcpItem,
  toggleToolLimit,
  withEndpoint,
  withReachableMcp,
  type PlaygroundSettings,
  type SettingsStorage,
} from "./playgroundSettings";

const storageOf = (values: Readonly<Record<string, string>>): SettingsStorage => ({
  read: (key) => values[key] ?? null,
  readSecure: (key) => values[key] ?? null,
});

const settingsWith = (change: Partial<PlaygroundSettings>): PlaygroundSettings => ({
  ...defaultSettings(false),
  ...change,
});

const TOOLS = ["search", "fetch", "summarize"] as const;
const SCOPE = { server: "s", tools: TOOLS };

describe("loadSettings", () => {
  it("restores every persisted field from what storedValues wrote", () => {
    const persisted: Partial<PlaygroundSettings> = {
      keySource: "custom",
      virtualKey: "sk-test",
      proxyBaseUrl: "http://gateway.test",
      endpoint: EndpointType.RESPONSES,
      model: "gpt-test",
      voice: OPEN_AI_VOICES.NOVA,
      tags: ["team-a"],
      headerPairs: [["x-trace", "abc"]],
      vectorStores: ["vs_1"],
      guardrails: ["pii-mask"],
      policies: ["strict"],
      mcpSelection: ["server-1"],
      toolLimits: { "server-1": ["search"] },
      stream: false,
    };
    const original = settingsWith(persisted);
    const written = Object.fromEntries(
      storedValues(original)
        .filter((entry): entry is { key: string; value: string; secure: boolean } => entry.value !== null)
        .map(({ key, value }) => [key, value]),
    );

    expect(loadSettings(storageOf(written), false)).toEqual(original);
  });

  it("falls back to defaults when stored values are corrupt or unknown", () => {
    const corrupt = {
      endpointType: "/v1/not-an-endpoint",
      selectedVoice: "robot",
      selectedTags: "{not json",
      selectedGuardrails: JSON.stringify([1, 2]),
      mcpServerToolRestrictions: JSON.stringify({ good: ["a"], bad: "a" }),
    };
    const loaded = loadSettings(storageOf(corrupt), false);

    expect(loaded.endpoint).toBe(EndpointType.CHAT);
    expect(loaded.voice).toBe(OPEN_AI_VOICES.ALLOY);
    expect(loaded.tags).toEqual([]);
    expect(loaded.guardrails).toEqual([]);
    expect(loaded.toolLimits).toEqual({ good: ["a"] });
  });

  it("forces the virtual key source when personal keys are disabled", () => {
    expect(loadSettings(storageOf({ apiKeySource: JSON.stringify("session") }), true).keySource).toBe("custom");
    expect(loadSettings(storageOf({ apiKeySource: JSON.stringify("session") }), false).keySource).toBe("session");
  });
});

describe("withEndpoint", () => {
  it("clears the model and keeps only a single MCP server when switching to direct tool calls", () => {
    const next = withEndpoint(settingsWith({ model: "gpt-test", mcpSelection: ["a", "b"] }), EndpointType.MCP);

    expect(next.endpoint).toBe(EndpointType.MCP);
    expect(next.model).toBeNull();
    expect(next.mcpSelection).toEqual([]);
    expect(withEndpoint(settingsWith({ mcpSelection: ["a"] }), EndpointType.MCP).mcpSelection).toEqual(["a"]);
    expect(withEndpoint(settingsWith({ mcpSelection: [ALL_MCP_SERVERS] }), EndpointType.MCP).mcpSelection).toEqual([]);
  });

  it("keeps the MCP selection for model endpoints", () => {
    expect(withEndpoint(settingsWith({ mcpSelection: ["a", "b"] }), EndpointType.RESPONSES).mcpSelection).toEqual([
      "a",
      "b",
    ]);
  });
});

describe("toggleMcpItem", () => {
  it("treats all servers as exclusive with individual servers", () => {
    expect(toggleMcpItem(["a", "b"], ALL_MCP_SERVERS, true)).toEqual([ALL_MCP_SERVERS]);
    expect(toggleMcpItem([ALL_MCP_SERVERS], "a", true)).toEqual(["a"]);
    expect(toggleMcpItem(["a", "b"], "a", false)).toEqual(["b"]);
  });
});

describe("withReachableMcp", () => {
  it("drops servers the current key cannot reach along with their tool limits", () => {
    const settings = settingsWith({
      mcpSelection: [ALL_MCP_SERVERS, "kept", "gone"],
      toolLimits: { kept: ["search"], gone: ["fetch"] },
    });

    const next = withReachableMcp(settings, new Set(["kept"]));

    expect(next.mcpSelection).toEqual([ALL_MCP_SERVERS, "kept"]);
    expect(next.toolLimits).toEqual({ kept: ["search"] });
  });

  it("leaves settings untouched while the catalog is loading or when everything is reachable", () => {
    const settings = settingsWith({ mcpSelection: ["kept"] });

    expect(withReachableMcp(settings, null)).toBe(settings);
    expect(withReachableMcp(settings, new Set(["kept", "other"]))).toBe(settings);
  });
});

describe("toggleToolLimit", () => {
  it("narrows an unrestricted server to the remaining tools when one is unchecked", () => {
    const limits = toggleToolLimit({}, SCOPE, "fetch", false);

    expect(allowedTools(limits, "s", TOOLS)).toEqual(["search", "summarize"]);
  });

  it("refuses to uncheck the last allowed tool", () => {
    const limits = { s: ["search"] };

    expect(toggleToolLimit(limits, SCOPE, "search", false)).toBe(limits);
  });

  it("lifts the restriction once every tool is checked again", () => {
    const limits = toggleToolLimit({ s: ["search", "fetch"] }, SCOPE, "summarize", true);

    expect(limits.s).toEqual([]);
    expect(allowedTools(limits, "s", TOOLS)).toEqual(TOOLS);
  });
});
