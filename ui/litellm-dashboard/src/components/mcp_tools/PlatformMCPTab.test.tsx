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
      if (fieldName === "platform_mcp_tool_threshold") {
        return { field_value: 10 };
      }
      return { field_value: null };
    });
  });

  it("shows the pre-v0 warning and v0 meta-tools", async () => {
    render(<PlatformMCPTab accessToken="token" />);

    expect(await screen.findByText(/This can change unexpectedly/i)).toBeInTheDocument();
    expect(screen.getByText(/product@berri.ai/i)).toBeInTheDocument();
    expect(screen.getByText(/tools\/list responses/i)).toBeInTheDocument();
    expect(screen.getByText("list_servers")).toBeInTheDocument();
    expect(screen.getByText("enable_server")).toBeInTheDocument();
    expect(screen.queryByText("search_tools")).not.toBeInTheDocument();
    expect(screen.queryByText("sandbox_execute")).not.toBeInTheDocument();
  });

  it("updates the enabled setting through the existing config field endpoint", async () => {
    render(<PlatformMCPTab accessToken="token" />);

    const toggle = await screen.findByRole("switch");
    fireEvent.click(toggle);

    await waitFor(() => {
      expect(networking.updateConfigFieldSetting).toHaveBeenCalledWith(
        "token",
        "platform_mcp_enabled",
        false,
      );
    });
  });

  it("updates the threshold through the existing config field endpoint", async () => {
    render(<PlatformMCPTab accessToken="token" />);

    const thresholdInput = await screen.findByRole("spinbutton");
    fireEvent.change(thresholdInput, { target: { value: "12" } });
    fireEvent.click(screen.getByRole("button", { name: /save/i }));

    await waitFor(() => {
      expect(networking.updateConfigFieldSetting).toHaveBeenCalledWith(
        "token",
        "platform_mcp_tool_threshold",
        12,
      );
    });
  });
});
