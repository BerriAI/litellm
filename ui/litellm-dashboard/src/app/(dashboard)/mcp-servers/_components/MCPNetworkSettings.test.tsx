import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import MCPNetworkSettings from "./MCPNetworkSettings";
import {
  getGeneralSettingsCall,
  updateConfigFieldSetting,
  deleteConfigFieldSetting,
  fetchMCPClientIp,
} from "@/components/networking";

vi.mock("@/components/networking", () => ({
  getGeneralSettingsCall: vi.fn(),
  updateConfigFieldSetting: vi.fn(),
  deleteConfigFieldSetting: vi.fn(),
  fetchMCPClientIp: vi.fn(),
}));

const renderSettings = () => render(<MCPNetworkSettings accessToken="tok" />);

describe("MCPNetworkSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([]);
    vi.mocked(fetchMCPClientIp).mockResolvedValue(null);
    vi.mocked(updateConfigFieldSetting).mockResolvedValue(undefined);
    vi.mocked(deleteConfigFieldSetting).mockResolvedValue(undefined);
  });

  it("renders the stored private ranges once settings load", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_internal_ip_ranges", field_value: ["10.0.0.0/8", "192.168.0.0/16"] },
    ]);

    renderSettings();

    expect(await screen.findByText("10.0.0.0/8")).toBeInTheDocument();
    expect(screen.getByText("192.168.0.0/16")).toBeInTheDocument();
  });

  it("ignores unrelated config fields", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "some_other_setting", field_value: ["should-not-show"] },
    ]);

    renderSettings();

    await screen.findByText("Private IP Ranges");
    expect(screen.queryByText("should-not-show")).not.toBeInTheDocument();
  });

  it("suggests the caller's /24 range from the detected client IP", async () => {
    vi.mocked(fetchMCPClientIp).mockResolvedValue("203.0.113.45");

    renderSettings();

    expect(await screen.findByText("203.0.113.45")).toBeInTheDocument();
    expect(screen.getByText("203.0.113.0/24")).toBeInTheDocument();
  });

  it("adds the suggested range to the list when clicked, and stops suggesting it", async () => {
    vi.mocked(fetchMCPClientIp).mockResolvedValue("203.0.113.45");

    renderSettings();
    await userEvent.click(await screen.findByText("203.0.113.0/24"));

    await waitFor(() => expect(screen.queryByText("Suggested range:")).not.toBeInTheDocument());
    expect(screen.getByText("203.0.113.0/24")).toBeInTheDocument();
  });

  it("saves the configured ranges", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_internal_ip_ranges", field_value: ["10.0.0.0/8"] },
    ]);

    renderSettings();
    await userEvent.click(await screen.findByRole("button", { name: /Save/ }));

    await waitFor(() =>
      expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_internal_ip_ranges", ["10.0.0.0/8"]),
    );
    expect(deleteConfigFieldSetting).not.toHaveBeenCalled();
  });

  it("clears the setting instead of saving an empty list", async () => {
    renderSettings();
    await userEvent.click(await screen.findByRole("button", { name: /Save/ }));

    await waitFor(() => expect(deleteConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_internal_ip_ranges"));
    expect(updateConfigFieldSetting).not.toHaveBeenCalled();
  });
});
