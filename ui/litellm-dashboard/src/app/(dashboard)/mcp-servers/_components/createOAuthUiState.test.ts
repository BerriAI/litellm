// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from "vitest";
import { setSecureItem } from "@/utils/secureStorage";
import { CreateUiSnapshot, readCreateUiSnapshot, writeCreateUiSnapshot } from "./createOAuthUiState";

const STORAGE_KEY = "litellm-mcp-oauth-create-state";

const fullSnapshot: CreateUiSnapshot = {
  modalVisible: true,
  formValues: { url: "https://example.com/mcp", auth_type: "oauth2", credentials: { client_id: "app-id" } },
  transportType: "http",
  costConfig: { default_cost_per_query: 0.02 },
  allowedTools: ["search"],
  hasToolAllowlistInteraction: true,
  aliasManuallyEdited: true,
  logoUrl: "https://cdn/logo.png",
  authorizedIdentity: "identity-abc",
};

const seedRaw = (value: unknown) => setSecureItem(STORAGE_KEY, JSON.stringify(value));

describe("createOAuthUiState", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    vi.restoreAllMocks();
  });

  it("returns null and leaves storage untouched when nothing was persisted", () => {
    expect(readCreateUiSnapshot()).toBeNull();
  });

  it("round-trips a full snapshot through the redirect", () => {
    writeCreateUiSnapshot(fullSnapshot);
    expect(readCreateUiSnapshot()).toEqual(fullSnapshot);
  });

  it("does not store the snapshot in plaintext", () => {
    writeCreateUiSnapshot(fullSnapshot);
    // secureStorage base64-encodes; a readable url in the raw value would mean the encoding was lost.
    expect(window.sessionStorage.getItem(STORAGE_KEY)).not.toContain("https://example.com/mcp");
  });

  it("consumes the snapshot so a second mount cannot replay it", () => {
    writeCreateUiSnapshot(fullSnapshot);
    expect(readCreateUiSnapshot()).not.toBeNull();
    expect(readCreateUiSnapshot()).toBeNull();
    expect(window.sessionStorage.getItem(STORAGE_KEY)).toBeNull();
  });

  it("strips minted token material so a stale token never rehydrates", () => {
    writeCreateUiSnapshot({
      ...fullSnapshot,
      formValues: {
        url: "https://example.com/mcp",
        credentials: {
          client_id: "app-id",
          client_secret: "app-secret",
          access_token: "stale-tok",
          refresh_token: "stale-refresh",
          expires_in: 3600,
          scope: "read",
        },
      },
    });

    const restored = readCreateUiSnapshot();
    expect(restored?.formValues?.credentials).toEqual({ client_id: "app-id", client_secret: "app-secret" });
    expect(JSON.stringify(restored)).not.toContain("stale-tok");
    expect(JSON.stringify(restored)).not.toContain("stale-refresh");
  });

  it("re-arms invalidation by restoring the authorized identity", () => {
    writeCreateUiSnapshot(fullSnapshot);
    expect(readCreateUiSnapshot()?.authorizedIdentity).toBe("identity-abc");
  });

  it("prefers the persisted form transport over the standalone transportType", () => {
    seedRaw({ formValues: { transport: "sse" }, transportType: "http" });
    expect(readCreateUiSnapshot()?.transportType).toBe("sse");
  });

  it("omits falsy scalars so a restore never blanks freshly mounted state", () => {
    seedRaw({ logoUrl: "", transportType: "", modalVisible: false });
    const restored = readCreateUiSnapshot();
    expect(restored).not.toHaveProperty("logoUrl");
    expect(restored).not.toHaveProperty("transportType");
    expect(restored).not.toHaveProperty("modalVisible");
  });

  it("restores an explicitly empty tool allowlist, which is a real admin choice", () => {
    seedRaw({ allowedTools: [], hasToolAllowlistInteraction: true });
    const restored = readCreateUiSnapshot();
    expect(restored?.allowedTools).toEqual([]);
    expect(restored?.hasToolAllowlistInteraction).toBe(true);
  });

  it.each([
    ["hasToolAllowlistInteraction", false],
    ["aliasManuallyEdited", false],
  ])("restores %s when it was persisted as false", (key, value) => {
    seedRaw({ [key]: value });
    expect(readCreateUiSnapshot()).toHaveProperty(key, value);
  });

  it.each([["hasToolAllowlistInteraction"], ["aliasManuallyEdited"]])(
    "ignores a non-boolean %s rather than coercing it",
    (key) => {
      seedRaw({ [key]: "yes" });
      expect(readCreateUiSnapshot()).not.toHaveProperty(key);
    },
  );

  it("ignores a non-string authorizedIdentity", () => {
    seedRaw({ authorizedIdentity: 42 });
    expect(readCreateUiSnapshot()).not.toHaveProperty("authorizedIdentity");
  });

  it("returns null on a corrupted payload but still clears it", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    setSecureItem(STORAGE_KEY, "{not json");

    expect(readCreateUiSnapshot()).toBeNull();
    expect(window.sessionStorage.getItem(STORAGE_KEY)).toBeNull();
  });
});
