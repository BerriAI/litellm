import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import MCPServerSelector from "./MCPServerSelector";
import { ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL } from "../mcp_tools/constants";

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: vi.fn(),
}));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups", () => ({
  useMCPAccessGroups: vi.fn(),
}));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPToolsets", () => ({
  useMCPToolsets: vi.fn(),
}));

import { useMCPAccessGroups } from "@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";

const mockUseMCPServers = vi.mocked(useMCPServers);
const mockUseMCPAccessGroups = vi.mocked(useMCPAccessGroups);
const mockUseMCPToolsets = vi.mocked(useMCPToolsets);

// The dropdown mounts its list asynchronously, so opening means waiting for the options too.
const openSelector = async (user: ReturnType<typeof userEvent.setup>): Promise<void> => {
  await user.click(screen.getByRole("combobox"));
  await screen.findAllByRole("option");
};

const optionByLabel = (label: string): HTMLElement | undefined =>
  screen.queryAllByRole("option").find((option) => option.textContent?.startsWith(label));

const setupMcpMocks = () => {
  mockUseMCPServers.mockReturnValue({
    data: [{ server_id: "srv-1", server_name: "Server One" }],
    isLoading: false,
  } as any);
  mockUseMCPAccessGroups.mockReturnValue({ data: [], isLoading: false } as any);
  mockUseMCPToolsets.mockReturnValue({ data: [], isLoading: false } as any);
};

describe("MCPServerSelector no-mcp-servers option", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupMcpMocks();
  });

  it("hides the No MCP Servers option by default", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <MCPServerSelector accessToken="tok" onChange={vi.fn()} value={{ servers: [], accessGroups: [] }} />,
    );

    await openSelector(user);

    expect(optionByLabel("Server One")).toBeDefined();
    expect(optionByLabel("No MCP Servers")).toBeUndefined();
  });

  it("emits an exclusive sentinel when No MCP Servers is selected", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowNoMcpServers
        onChange={onChange}
        value={{ servers: ["srv-1"], accessGroups: [] }}
      />,
    );

    await openSelector(user);
    await user.click(optionByLabel("No MCP Servers")!);

    expect(onChange).toHaveBeenCalledWith({ servers: [NO_MCP_SERVERS_SENTINEL], accessGroups: [], toolsets: [] });
  });

  it("disables real server options while the sentinel is selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowNoMcpServers
        onChange={vi.fn()}
        value={{ servers: [NO_MCP_SERVERS_SENTINEL], accessGroups: [] }}
      />,
    );

    await openSelector(user);

    expect(optionByLabel("Server One")).toHaveAttribute("aria-disabled", "true");
    expect(optionByLabel("No MCP Servers")).not.toHaveAttribute("aria-disabled", "true");
  });
});

describe("MCPServerSelector all-proxy-mcpservers option", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupMcpMocks();
  });

  it("hides the All Proxy MCP Servers option by default", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <MCPServerSelector accessToken="tok" onChange={vi.fn()} value={{ servers: [], accessGroups: [] }} />,
    );

    await openSelector(user);

    expect(optionByLabel("Server One")).toBeDefined();
    expect(optionByLabel("All Proxy MCP Servers")).toBeUndefined();
  });

  it("emits an exclusive sentinel when All Proxy MCP Servers is selected", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowAllProxyMcpServers
        onChange={onChange}
        value={{ servers: ["srv-1"], accessGroups: [] }}
      />,
    );

    await openSelector(user);
    await user.click(optionByLabel("All Proxy MCP Servers")!);

    expect(onChange).toHaveBeenCalledWith({
      servers: [ALL_PROXY_MCP_SERVERS_SENTINEL],
      accessGroups: [],
      toolsets: [],
    });
  });

  it("disables real server options while the sentinel is selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowAllProxyMcpServers
        onChange={vi.fn()}
        value={{ servers: [ALL_PROXY_MCP_SERVERS_SENTINEL], accessGroups: [] }}
      />,
    );

    await openSelector(user);

    expect(optionByLabel("Server One")).toHaveAttribute("aria-disabled", "true");
    expect(optionByLabel("All Proxy MCP Servers")).not.toHaveAttribute("aria-disabled", "true");
  });

  it("renders the friendly label, not the raw literal, when the sentinel is already stored but the flag is off", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        onChange={vi.fn()}
        value={{ servers: [ALL_PROXY_MCP_SERVERS_SENTINEL], accessGroups: [] }}
      />,
    );

    expect(screen.getByLabelText("All Proxy MCP Servers")).toBeInTheDocument();
    expect(screen.queryByText(ALL_PROXY_MCP_SERVERS_SENTINEL)).not.toBeInTheDocument();

    await openSelector(user);

    expect(optionByLabel("Server One")).toHaveAttribute("aria-disabled", "true");
  });
});
