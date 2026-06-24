import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import MCPServerSelector from "./MCPServerSelector";
import { NO_MCP_SERVERS_SENTINEL } from "../mcp_tools/constants";

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

describe("MCPServerSelector no-mcp-servers option", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseMCPServers.mockReturnValue({
      data: [{ server_id: "srv-1", server_name: "Server One" }],
      isLoading: false,
    } as any);
    mockUseMCPAccessGroups.mockReturnValue({ data: [], isLoading: false } as any);
    mockUseMCPToolsets.mockReturnValue({ data: [], isLoading: false } as any);
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
