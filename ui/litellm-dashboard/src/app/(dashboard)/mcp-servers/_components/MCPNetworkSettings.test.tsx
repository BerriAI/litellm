import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import MCPNetworkSettings from "./MCPNetworkSettings";
import { toast } from "@/lib/toast";
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

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn() },
}));

const renderSettings = () => render(<MCPNetworkSettings accessToken="tok" />);

const ANTIGRAVITY = { alias: "Antigravity CLI", value: "antigravity-cli" };
const CODEX = { alias: "Codex", value: "codex-mcp-client" };

const addClient = async (alias: string, value: string) => {
  await userEvent.click(screen.getByRole("button", { name: "Add client" }));
  const aliases = screen.getAllByRole("textbox", { name: /^Client \d+ alias$/ });
  const values = screen.getAllByRole("textbox", { name: /^Client \d+ value$/ });
  fireEvent.change(aliases[aliases.length - 1], { target: { value: alias } });
  fireEvent.change(values[values.length - 1], { target: { value } });
};

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

  it("saves the configured ranges once they change", async () => {
    vi.mocked(fetchMCPClientIp).mockResolvedValue("203.0.113.45");
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_internal_ip_ranges", field_value: ["10.0.0.0/8"] },
    ]);

    renderSettings();
    await userEvent.click(await screen.findByText("203.0.113.0/24"));
    await userEvent.click(await screen.findByRole("button", { name: /Save/ }));

    await waitFor(() =>
      expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_internal_ip_ranges", [
        "10.0.0.0/8",
        "203.0.113.0/24",
      ]),
    );
    expect(deleteConfigFieldSetting).not.toHaveBeenCalledWith("tok", "mcp_internal_ip_ranges");
  });

  it("clears a stored range setting instead of saving an empty list", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_internal_ip_ranges", field_value: ["10.0.0.0/8"] },
    ]);

    renderSettings();
    await userEvent.click(await screen.findByRole("button", { name: "Remove 10.0.0.0/8" }));
    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() => expect(deleteConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_internal_ip_ranges"));
    expect(updateConfigFieldSetting).not.toHaveBeenCalled();
  });

  it("does not write settings that were never stored and are still empty", async () => {
    renderSettings();
    await userEvent.click(await screen.findByRole("button", { name: /Save/ }));

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("MCP network settings saved"));
    expect(deleteConfigFieldSetting).not.toHaveBeenCalled();
    expect(updateConfigFieldSetting).not.toHaveBeenCalled();
  });

  it("labels the section Allowed Clients and renders each stored client as an alias and value row", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_allowed_clients", field_value: [ANTIGRAVITY, CODEX] },
    ]);

    renderSettings();

    expect(await screen.findByText("Allowed Clients")).toBeVisible();
    expect(screen.queryByText(/Allowed Client IDs/)).not.toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Client 1 alias" })).toHaveValue("Antigravity CLI");
    expect(screen.getByRole("textbox", { name: "Client 1 value" })).toHaveValue("antigravity-cli");
    expect(screen.getByRole("textbox", { name: "Client 2 alias" })).toHaveValue("Codex");
    expect(screen.getByRole("textbox", { name: "Client 2 value" })).toHaveValue("codex-mcp-client");
  });

  it("warns that a stored allowlist in the old plain-string shape denies every client and lets Save remove it", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_allowed_clients", field_value: ["antigravity-cli"] },
    ]);

    renderSettings();

    expect(await screen.findByText(/stored allowlist is not a list of alias and value pairs/)).toBeVisible();
    expect(screen.queryByRole("textbox", { name: "Client 1 value" })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() => expect(deleteConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_allowed_clients"));
    expect(updateConfigFieldSetting).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.queryByText(/stored allowlist is not a list/)).not.toBeInTheDocument());
  });

  it("replaces a stored allowlist in the old plain-string shape with the clients the admin adds", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_allowed_clients", field_value: ["antigravity-cli"] },
    ]);

    renderSettings();

    await screen.findByText(/stored allowlist is not a list of alias and value pairs/);
    await addClient(ANTIGRAVITY.alias, ANTIGRAVITY.value);
    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() =>
      expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_allowed_clients", [ANTIGRAVITY]),
    );
    expect(deleteConfigFieldSetting).not.toHaveBeenCalledWith("tok", "mcp_allowed_clients");
    await waitFor(() => expect(screen.queryByText(/stored allowlist is not a list/)).not.toBeInTheDocument());
  });

  it("adds clients as alias and value pairs and saves them under mcp_allowed_clients", async () => {
    renderSettings();
    await screen.findByText("Allowed Clients");

    await addClient(" Antigravity CLI ", " antigravity-cli ");
    await addClient("Codex", "codex-mcp-client");
    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() =>
      expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_allowed_clients", [ANTIGRAVITY, CODEX]),
    );
    expect(deleteConfigFieldSetting).not.toHaveBeenCalledWith("tok", "mcp_allowed_clients");
  });

  it("edits a stored client's value in place and saves the new value", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_allowed_clients", field_value: [ANTIGRAVITY] },
    ]);

    renderSettings();
    fireEvent.change(await screen.findByRole("textbox", { name: "Client 1 value" }), {
      target: { value: "0oa1b2c3d4e5f6g7h8i9" },
    });
    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() =>
      expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_allowed_clients", [
        { alias: "Antigravity CLI", value: "0oa1b2c3d4e5f6g7h8i9" },
      ]),
    );
  });

  it("refuses to save a client that has an alias but no value, and reports why", async () => {
    renderSettings();
    await screen.findByText("Allowed Clients");

    await addClient("Antigravity CLI", "");
    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() =>
      expect(toast.fromError).toHaveBeenCalledWith(new Error("Every allowed client needs both an alias and a value")),
    );
    expect(updateConfigFieldSetting).not.toHaveBeenCalled();
    expect(deleteConfigFieldSetting).not.toHaveBeenCalled();
    expect(toast.success).not.toHaveBeenCalled();
  });

  it("drops rows left completely blank instead of saving or failing on them", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_allowed_clients", field_value: [ANTIGRAVITY] },
    ]);

    renderSettings();
    await screen.findByText("Allowed Clients");
    await userEvent.click(screen.getByRole("button", { name: "Add client" }));
    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("MCP network settings saved"));
    expect(updateConfigFieldSetting).not.toHaveBeenCalled();
    expect(deleteConfigFieldSetting).not.toHaveBeenCalled();
  });

  it("removes the right client from the middle of the list", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      {
        field_name: "mcp_allowed_clients",
        field_value: [ANTIGRAVITY, { alias: "Claude Code", value: "claude-code" }, CODEX],
      },
    ]);

    renderSettings();
    await userEvent.click(await screen.findByRole("button", { name: "Remove client Claude Code" }));
    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() =>
      expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_allowed_clients", [ANTIGRAVITY, CODEX]),
    );
    expect(screen.queryByDisplayValue("claude-code")).not.toBeInTheDocument();
  });

  it("removes a client and clears the setting when the list becomes empty", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_allowed_clients", field_value: [{ alias: "Claude Code", value: "claude-code" }] },
    ]);

    renderSettings();
    await userEvent.click(await screen.findByRole("button", { name: "Remove client Claude Code" }));

    expect(screen.queryByDisplayValue("claude-code")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() => expect(deleteConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_allowed_clients"));
    expect(updateConfigFieldSetting).not.toHaveBeenCalledWith("tok", "mcp_allowed_clients", expect.anything());
  });

  it("warns that a stored empty allowlist denies every client and lets Save remove it", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([{ field_name: "mcp_allowed_clients", field_value: [] }]);

    renderSettings();

    expect(await screen.findByText(/An empty allowlist is currently stored, so every client is denied/)).toBeVisible();

    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() => expect(deleteConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_allowed_clients"));
    expect(updateConfigFieldSetting).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.queryByText(/An empty allowlist is currently stored/)).not.toBeInTheDocument());
  });

  it("does not show the deny-all warning when no allowlist is stored", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([{ field_name: "mcp_allowed_clients", field_value: null }]);

    renderSettings();

    await screen.findByText("Allowed Client Applications");
    expect(screen.queryByText(/every client is denied/)).not.toBeInTheDocument();
  });

  it("explains that JWT callers are identified by the configured claim and others by the opt-in header", async () => {
    renderSettings();

    expect(await screen.findByText(/litellm_jwtauth\.mcp_client_id_jwt_field/)).toBeVisible();
    expect(screen.getByText(/Clients pick this value themselves, so it is a policy control/)).toBeVisible();
    expect(screen.queryByText(/clientInfo/)).not.toBeInTheDocument();
  });

  it("renders the stored client identity header once settings load", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_client_id_header", field_value: "x-mcp-client" },
    ]);

    renderSettings();

    expect(await screen.findByRole("textbox", { name: "Client identity header" })).toHaveValue("x-mcp-client");
  });

  it("saves a newly typed client identity header under mcp_client_id_header", async () => {
    renderSettings();
    fireEvent.change(await screen.findByRole("textbox", { name: "Client identity header" }), {
      target: { value: " x-mcp-client " },
    });

    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() =>
      expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_client_id_header", "x-mcp-client"),
    );
    expect(updateConfigFieldSetting).not.toHaveBeenCalledWith("tok", "mcp_allowed_clients", expect.anything());
    expect(deleteConfigFieldSetting).not.toHaveBeenCalled();
  });

  it("clears a stored client identity header when the field is emptied, so only JWT identity is trusted", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_client_id_header", field_value: "x-mcp-client" },
    ]);

    renderSettings();
    fireEvent.change(await screen.findByRole("textbox", { name: "Client identity header" }), {
      target: { value: "" },
    });

    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() => expect(deleteConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_client_id_header"));
    expect(updateConfigFieldSetting).not.toHaveBeenCalled();
  });

  it("does not rewrite an unchanged client identity header on save", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_client_id_header", field_value: "x-mcp-client" },
    ]);

    renderSettings();
    await screen.findByRole("textbox", { name: "Client identity header" });
    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() => expect(toast.success).toHaveBeenCalled());
    expect(updateConfigFieldSetting).not.toHaveBeenCalled();
    expect(deleteConfigFieldSetting).not.toHaveBeenCalled();
  });

  it("keeps the private ranges and the allowed clients as independent settings on save", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_internal_ip_ranges", field_value: ["10.0.0.0/8"] },
      { field_name: "mcp_allowed_clients", field_value: [ANTIGRAVITY] },
    ]);

    renderSettings();
    await screen.findByText("Allowed Clients");
    await addClient("Codex", "codex-mcp-client");
    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() =>
      expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_allowed_clients", [ANTIGRAVITY, CODEX]),
    );
    expect(updateConfigFieldSetting).not.toHaveBeenCalledWith("tok", "mcp_internal_ip_ranges", expect.anything());
    expect(deleteConfigFieldSetting).not.toHaveBeenCalled();
  });

  it("still saves the allowed clients when the private range write fails, and reports the failure", async () => {
    vi.mocked(getGeneralSettingsCall).mockResolvedValue([
      { field_name: "mcp_internal_ip_ranges", field_value: ["10.0.0.0/8"] },
    ]);
    const rangeFailure = new Error("Field name=mcp_internal_ip_ranges not in config");
    vi.mocked(deleteConfigFieldSetting).mockRejectedValue(rangeFailure);

    renderSettings();
    await userEvent.click(await screen.findByRole("button", { name: "Remove 10.0.0.0/8" }));
    await addClient("Codex", "codex-mcp-client");
    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() => expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_allowed_clients", [CODEX]));
    await waitFor(() => expect(toast.fromError).toHaveBeenCalledWith(rangeFailure));
    expect(toast.success).not.toHaveBeenCalled();
  });

  it("writes the private ranges and the allowed clients one after the other, never concurrently", async () => {
    vi.mocked(fetchMCPClientIp).mockResolvedValue("203.0.113.45");
    let finishRangeWrite: (() => void) | undefined;
    vi.mocked(updateConfigFieldSetting).mockImplementation(
      (_token, fieldName) =>
        new Promise<void>((resolve) => {
          if (fieldName === "mcp_internal_ip_ranges") {
            finishRangeWrite = resolve;
          } else {
            resolve();
          }
        }),
    );

    renderSettings();
    await userEvent.click(await screen.findByText("203.0.113.0/24"));
    await addClient("Codex", "codex-mcp-client");
    await userEvent.click(screen.getByRole("button", { name: /Save/ }));

    await waitFor(() =>
      expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_internal_ip_ranges", ["203.0.113.0/24"]),
    );
    expect(updateConfigFieldSetting).not.toHaveBeenCalledWith("tok", "mcp_allowed_clients", expect.anything());

    finishRangeWrite?.();
    await waitFor(() => expect(updateConfigFieldSetting).toHaveBeenCalledWith("tok", "mcp_allowed_clients", [CODEX]));
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("MCP network settings saved"));
  });
});
