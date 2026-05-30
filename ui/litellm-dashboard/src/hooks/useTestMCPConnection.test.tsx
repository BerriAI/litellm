import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { useTestMCPConnection } from "./useTestMCPConnection";
import * as networking from "../components/networking";

vi.mock("../components/networking", () => ({
  testMCPToolsListRequest: vi.fn().mockResolvedValue({ tools: [], error: null }),
}));

describe("useTestMCPConnection (Bug #15)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const formValues = {
    url: "https://example.com/mcp",
    transport: "http",
    auth_type: "none",
  };

  it("does not call the admin-only preview endpoint when disabled", async () => {
    renderHook(() =>
      useTestMCPConnection({ accessToken: "sk-test", formValues, enabled: false }),
    );

    // Give the auto-fetch effect a chance to run; it must stay gated.
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(networking.testMCPToolsListRequest).not.toHaveBeenCalled();
  });

  it("calls the preview endpoint when enabled", async () => {
    renderHook(() =>
      useTestMCPConnection({ accessToken: "sk-test", formValues, enabled: true }),
    );

    await waitFor(() => {
      expect(networking.testMCPToolsListRequest).toHaveBeenCalled();
    });
  });
});
