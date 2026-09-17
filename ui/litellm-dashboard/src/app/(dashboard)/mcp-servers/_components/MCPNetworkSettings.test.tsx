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

  it("exposes the suggested range as a control a keyboard user can reach and activate", async () => {
    vi.mocked(fetchMCPClientIp).mockResolvedValue("203.0.113.45");

    renderSettings();
    const suggested = await screen.findByRole("button", { name: /203\.0\.113\.0\/24/ });

    suggested.focus();
    expect(suggested).toHaveFocus();

    await userEvent.keyboard("{Enter}");

    await waitFor(() => expect(screen.queryByText("Suggested range:")).not.toBeInTheDocument());
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
    expect(deleteConfigFieldSetting).not.toHaveBeenCalledWith("tok", "mcp_internal_ip_ranges");
  });

  it("clears the setting instead of saving an empty list", async () => {
    renderSettings();
    await userEvent.click(await screen.findByRole("button", { name: /Save/ }));

    await waitFor(() => expect(deleteConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_internal_ip_ranges"));
    expect(updateConfigFieldSetting).not.toHaveBeenCalled();
  });

  it("renders the stored allowed client names once settings load", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_allowed_clients", field_value: ["antigravity-cli", "codex-mcp-client"] },
    ]);

    renderSettings();

    expect(await screen.findByText("antigravity-cli")).toBeInTheDocument();
    expect(screen.getByText("codex-mcp-client")).toBeInTheDocument();
  });

  it("adds typed client names on Enter and saves them under mcp_allowed_clients", async () => {
    renderSettings();
    const input = await screen.findByRole("textbox", { name: "Allowed client names" });

    await userEvent.type(input, "antigravity-cli, codex-mcp-client{Enter}");

    expect(screen.getByText("antigravity-cli")).toBeInTheDocument();
    expect(screen.getByText("codex-mcp-client")).toBeInTheDocument();
    expect(input).toHaveValue("");

    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() =>
      expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_allowed_clients", [
        "antigravity-cli",
        "codex-mcp-client",
      ]),
    );
    expect(deleteConfigFieldSetting).not.toHaveBeenCalledWith("tok", "mcp_allowed_clients");
  });

  it("removes a client name and clears the setting when the list becomes empty", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_allowed_clients", field_value: ["claude-code"] },
    ]);

    renderSettings();
    await userEvent.click(await screen.findByRole("button", { name: "Remove claude-code" }));

    expect(screen.queryByText("claude-code")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() => expect(deleteConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_allowed_clients"));
    expect(updateConfigFieldSetting).not.toHaveBeenCalledWith("tok", "mcp_allowed_clients", expect.anything());
  });

  it("keeps the private ranges and the allowed clients as independent settings on save", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_internal_ip_ranges", field_value: ["10.0.0.0/8"] },
      { field_name: "mcp_allowed_clients", field_value: ["antigravity-cli"] },
    ]);

    renderSettings();
    await userEvent.click(await screen.findByRole("button", { name: /Save/ }));

    await waitFor(() =>
      expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_allowed_clients", ["antigravity-cli"]),
    );
    expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_internal_ip_ranges", ["10.0.0.0/8"]);
    expect(deleteConfigFieldSetting).not.toHaveBeenCalled();
  });
});
