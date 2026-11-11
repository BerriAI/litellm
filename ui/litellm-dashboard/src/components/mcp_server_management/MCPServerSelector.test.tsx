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

describe("MCPServerSelector scoping", () => {
  const catalog = [
    { server_id: "srv-a", server_name: "Wiki A", mcp_access_groups: ["g1"] },
    { server_id: "srv-b", server_name: "Wiki B", mcp_access_groups: ["g1", "g2"] },
    { server_id: "srv-c", server_name: "Tools C", mcp_access_groups: ["g2"] },
    { server_id: "srv-d", server_name: "Solo D" },
    { server_id: "srv-e", server_name: "Public E", allow_all_keys: true },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseMCPServers.mockReturnValue({ data: catalog, isLoading: false } as any);
    mockUseMCPAccessGroups.mockReturnValue({ data: ["g1", "g2"], isLoading: false } as any);
    mockUseMCPToolsets.mockReturnValue({ data: [], isLoading: false } as any);
  });

  it("asks the server list for the team and user scope it was given", () => {
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        onChange={vi.fn()}
        value={{ servers: [], accessGroups: [] }}
        teamId="team-1"
        userId="user-1"
      />,
    );

    expect(mockUseMCPServers).toHaveBeenCalledWith("team-1", "user-1");
  });

  it("lists every server while no access group is selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <MCPServerSelector accessToken="tok" onChange={vi.fn()} value={{ servers: [], accessGroups: [] }} />,
    );

    await openSelector(user);

    expect(["Wiki A", "Wiki B", "Tools C", "Solo D", "Public E"].map(optionByLabel).every(Boolean)).toBe(true);
  });

  it("narrows the servers to members of the selected access groups, keeping allow-all and granted ones", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <MCPServerSelector accessToken="tok" onChange={vi.fn()} value={{ servers: ["srv-d"], accessGroups: ["g1"] }} />,
    );

    await openSelector(user);

    expect(optionByLabel("Wiki A")).toBeDefined();
    expect(optionByLabel("Wiki B")).toBeDefined();
    expect(optionByLabel("Public E")).toBeDefined();
    expect(optionByLabel("Solo D")).toBeDefined();
    expect(optionByLabel("Tools C")).toBeUndefined();
  });

  it("widens again once the access group is deselected", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { rerender } = renderWithProviders(
      <MCPServerSelector accessToken="tok" onChange={onChange} value={{ servers: [], accessGroups: ["g2"] }} />,
    );

    await openSelector(user);
    expect(optionByLabel("Wiki A")).toBeUndefined();
    await user.keyboard("{Escape}");

    rerender(<MCPServerSelector accessToken="tok" onChange={onChange} value={{ servers: [], accessGroups: [] }} />);
    await openSelector(user);

    expect(optionByLabel("Wiki A")).toBeDefined();
  });
});
