import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import MCPServerSelector from "./MCPServerSelector";
import {
  ALL_PROXY_MCPS_SENTINEL,
  ALL_TEAM_MCPS_SENTINEL,
  NO_MCP_SERVERS_SENTINEL,
} from "../mcp_tools/constants";

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

const setupMcpHooks = (servers: { server_id: string; server_name: string }[]) => {
  vi.clearAllMocks();
  mockUseMCPServers.mockReturnValue({ data: servers, isLoading: false } as unknown as ReturnType<typeof useMCPServers>);
  mockUseMCPAccessGroups.mockReturnValue({
    data: [],
    isLoading: false,
  } as unknown as ReturnType<typeof useMCPAccessGroups>);
  mockUseMCPToolsets.mockReturnValue({
    data: [],
    isLoading: false,
  } as unknown as ReturnType<typeof useMCPToolsets>);
};

describe("MCPServerSelector no-mcp-servers option", () => {
  beforeEach(() => {
    setupMcpHooks([{ server_id: "srv-1", server_name: "Server One" }]);
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

describe("MCPServerSelector all-proxy-mcps option", () => {
  beforeEach(() => {
    setupMcpHooks([{ server_id: "srv-1", server_name: "Server One" }]);
  });

  const optionByValue = (value: string) =>
    Array.from(screen.getByTestId("mcp-select").querySelectorAll("option")).find(
      (o) => (o as HTMLOptionElement).value === value,
    ) as HTMLOptionElement | undefined;

  it("hides the All Proxy MCPs option by default", () => {
    renderWithProviders(
      <MCPServerSelector accessToken="tok" onChange={vi.fn()} value={{ servers: [], accessGroups: [] }} />,
    );
    expect(optionByValue(ALL_PROXY_MCPS_SENTINEL)).toBeUndefined();
  });

  it("emits an exclusive sentinel when All Proxy MCPs is selected", async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowAllProxyMcps
        onChange={onChange}
        value={{ servers: ["srv-1"], accessGroups: [] }}
      />,
    );
    expect(optionByValue(ALL_PROXY_MCPS_SENTINEL)).toBeDefined();

    await userEvent.selectOptions(screen.getByTestId("mcp-select"), [ALL_PROXY_MCPS_SENTINEL]);

    expect(onChange).toHaveBeenCalledWith({ servers: [ALL_PROXY_MCPS_SENTINEL], accessGroups: [], toolsets: [] });
  });

  it("disables real server options while the sentinel is selected", () => {
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowAllProxyMcps
        onChange={vi.fn()}
        value={{ servers: [ALL_PROXY_MCPS_SENTINEL], accessGroups: [] }}
      />,
    );
    expect(optionByValue("srv-1")?.disabled).toBe(true);
    expect(optionByValue(ALL_PROXY_MCPS_SENTINEL)?.disabled).toBe(false);
  });
});

describe("MCPServerSelector all-team-mcps option", () => {
  beforeEach(() => {
    setupMcpHooks([{ server_id: "srv-1", server_name: "Server One" }]);
  });

  const optionByValue = (value: string) =>
    Array.from(screen.getByTestId("mcp-select").querySelectorAll("option")).find(
      (o) => (o as HTMLOptionElement).value === value,
    ) as HTMLOptionElement | undefined;

  it("hides All Team MCPs when no team is selected, even with allowAllTeamMcps", () => {
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowAllTeamMcps
        onChange={vi.fn()}
        value={{ servers: [], accessGroups: [] }}
      />,
    );
    expect(optionByValue(ALL_TEAM_MCPS_SENTINEL)).toBeUndefined();
  });

  it("shows All Team MCPs once a team is selected", () => {
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowAllTeamMcps
        teamId="team-1"
        onChange={vi.fn()}
        value={{ servers: [], accessGroups: [] }}
      />,
    );
    expect(optionByValue(ALL_TEAM_MCPS_SENTINEL)).toBeDefined();
  });

  it("emits an exclusive sentinel when All Team MCPs is selected", async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowAllTeamMcps
        teamId="team-1"
        onChange={onChange}
        value={{ servers: ["srv-1"], accessGroups: [] }}
      />,
    );
    await userEvent.selectOptions(screen.getByTestId("mcp-select"), [ALL_TEAM_MCPS_SENTINEL]);

    expect(onChange).toHaveBeenCalledWith({ servers: [ALL_TEAM_MCPS_SENTINEL], accessGroups: [], toolsets: [] });
  });

  it("makes the three exclusive sentinels mutually exclusive", () => {
    renderWithProviders(
      <MCPServerSelector
        accessToken="tok"
        allowNoMcpServers
        allowAllTeamMcps
        teamId="team-1"
        onChange={vi.fn()}
        value={{ servers: [ALL_TEAM_MCPS_SENTINEL], accessGroups: [] }}
      />,
    );
    expect(optionByValue(ALL_TEAM_MCPS_SENTINEL)?.disabled).toBe(false);
    expect(optionByValue(NO_MCP_SERVERS_SENTINEL)?.disabled).toBe(true);
    expect(optionByValue("srv-1")?.disabled).toBe(true);
  });
});
