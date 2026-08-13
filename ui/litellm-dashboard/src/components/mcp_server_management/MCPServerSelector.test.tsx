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

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();
  const Select = ({ value, onChange, children, mode }: any) => (
    <select
      data-testid="mcp-select"
      multiple={mode === "multiple"}
      value={value}
      onChange={(e) => onChange(Array.from(e.target.selectedOptions, (o) => (o as HTMLOptionElement).value))}
    >
      {children}
    </select>
  );
  Select.displayName = "MockSelect";
  Select.Option = ({ value, disabled, label }: any) => (
    <option value={value} disabled={disabled}>
      {label}
    </option>
  );
  Select.Option.displayName = "MockSelectOption";
  return { ...actual, Select };
});

import { useMCPAccessGroups } from "@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";

const mockUseMCPServers = vi.mocked(useMCPServers);
const mockUseMCPAccessGroups = vi.mocked(useMCPAccessGroups);
const mockUseMCPToolsets = vi.mocked(useMCPToolsets);

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

  const optionByValue = (value: string) =>
    Array.from(screen.getByTestId("mcp-select").querySelectorAll("option")).find(
      (o) => (o as HTMLOptionElement).value === value,
    ) as HTMLOptionElement | undefined;

  it("hides the No MCP Servers option by default", () => {
    renderWithProviders(
      <MCPServerSelector accessToken="tok" onChange={vi.fn()} value={{ servers: [], accessGroups: [] }} />,
    );
    expect(optionByValue(NO_MCP_SERVERS_SENTINEL)).toBeUndefined();
  });

  it("emits an exclusive sentinel when No MCP Servers is selected", async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowNoMcpServers
        onChange={onChange}
        value={{ servers: ["srv-1"], accessGroups: [] }}
      />,
    );
    expect(optionByValue(NO_MCP_SERVERS_SENTINEL)).toBeDefined();

    await userEvent.selectOptions(screen.getByTestId("mcp-select"), [NO_MCP_SERVERS_SENTINEL]);

    expect(onChange).toHaveBeenCalledWith({ servers: [NO_MCP_SERVERS_SENTINEL], accessGroups: [], toolsets: [] });
  });

  it("disables real server options while the sentinel is selected", () => {
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowNoMcpServers
        onChange={vi.fn()}
        value={{ servers: [NO_MCP_SERVERS_SENTINEL], accessGroups: [] }}
      />,
    );
    expect(optionByValue("srv-1")?.disabled).toBe(true);
    expect(optionByValue(NO_MCP_SERVERS_SENTINEL)?.disabled).toBe(false);
  });
});

describe("MCPServerSelector all-proxy-mcpservers option", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupMcpMocks();
  });

  const optionByValue = (value: string) =>
    Array.from(screen.getByTestId("mcp-select").querySelectorAll("option")).find(
      (o) => (o as HTMLOptionElement).value === value,
    ) as HTMLOptionElement | undefined;

  it("hides the All Proxy MCP Servers option by default", () => {
    renderWithProviders(
      <MCPServerSelector accessToken="tok" onChange={vi.fn()} value={{ servers: [], accessGroups: [] }} />,
    );
    expect(optionByValue(ALL_PROXY_MCP_SERVERS_SENTINEL)).toBeUndefined();
  });

  it("emits an exclusive sentinel when All Proxy MCP Servers is selected", async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowAllProxyMcpServers
        onChange={onChange}
        value={{ servers: ["srv-1"], accessGroups: [] }}
      />,
    );
    expect(optionByValue(ALL_PROXY_MCP_SERVERS_SENTINEL)).toBeDefined();

    await userEvent.selectOptions(screen.getByTestId("mcp-select"), [ALL_PROXY_MCP_SERVERS_SENTINEL]);

    expect(onChange).toHaveBeenCalledWith({
      servers: [ALL_PROXY_MCP_SERVERS_SENTINEL],
      accessGroups: [],
      toolsets: [],
    });
  });

  it("disables real server options while the sentinel is selected", () => {
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowAllProxyMcpServers
        onChange={vi.fn()}
        value={{ servers: [ALL_PROXY_MCP_SERVERS_SENTINEL], accessGroups: [] }}
      />,
    );
    expect(optionByValue("srv-1")?.disabled).toBe(true);
    expect(optionByValue(ALL_PROXY_MCP_SERVERS_SENTINEL)?.disabled).toBe(false);
  });

  it("renders the friendly option, not the raw literal, when the sentinel is already stored but the flag is off", () => {
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        onChange={vi.fn()}
        value={{ servers: [ALL_PROXY_MCP_SERVERS_SENTINEL], accessGroups: [] }}
      />,
    );
    const option = optionByValue(ALL_PROXY_MCP_SERVERS_SENTINEL);
    expect(option).toBeDefined();
    expect(option?.textContent).toContain("All Proxy MCP Servers");
    expect(optionByValue("srv-1")?.disabled).toBe(true);
  });
});
