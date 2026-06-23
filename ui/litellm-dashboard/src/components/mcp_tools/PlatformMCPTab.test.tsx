import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import PlatformMCPTab from "./PlatformMCPTab";
import * as networking from "../networking";

vi.mock("../networking", () => ({
  getConfigFieldSetting: vi.fn(),
  updateConfigFieldSetting: vi.fn().mockResolvedValue(undefined),
}));

describe("PlatformMCPTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.getConfigFieldSetting).mockImplementation(async (_accessToken, fieldName) => {
      if (fieldName === "platform_mcp_enabled") {
        return { field_value: true };
      }
      return { field_value: null };
    });
  });

  it("shows the pre-v0 warning and platform MCP tools", async () => {
    render(<PlatformMCPTab accessToken="token" />);

    expect(await screen.findByText(/This can change unexpectedly/i)).toBeInTheDocument();
    expect(screen.getByText(/product@berri.ai/i)).toBeInTheDocument();
    expect(screen.getByText(/platform-managed MCP discovery and tool calling/i)).toBeInTheDocument();
    expect(screen.getByText(/list_servers, get_server_tools, and/i)).toBeInTheDocument();
    expect(screen.getByText("list_servers")).toBeInTheDocument();
    expect(screen.getByText("get_server_tools")).toBeInTheDocument();
    expect(screen.getByText("call_tool")).toBeInTheDocument();
    expect(screen.queryByText("search_tools")).not.toBeInTheDocument();
    expect(screen.queryByText("sandbox_execute")).not.toBeInTheDocument();
    expect(networking.getConfigFieldSetting).toHaveBeenCalledWith("token", "platform_mcp_enabled");
  });

  it("updates the enabled setting through the existing config field endpoint", async () => {
    render(<PlatformMCPTab accessToken="token" />);

    const toggle = await screen.findByRole("switch", { name: /enable platform mcp/i });
    fireEvent.click(toggle);

    await waitFor(() => {
      expect(networking.updateConfigFieldSetting).toHaveBeenCalledWith("token", "platform_mcp_enabled", false);
    });
  });
});
