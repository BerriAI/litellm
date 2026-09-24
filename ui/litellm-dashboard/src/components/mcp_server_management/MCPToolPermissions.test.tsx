import { useState } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import MCPToolPermissions, { McpToolPermissionWrite } from "./MCPToolPermissions";
import * as networking from "../networking";
import { NO_MCP_SERVERS_SENTINEL } from "../mcp_tools/constants";
import type { MCPToolset } from "../mcp_tools/types";

vi.mock("../networking");

describe("MCPToolPermissions", () => {
  const mockAccessToken = "test-token";
  const mockServerId = "server-123";
  const mockServerName = "Test MCP Server";

  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
    vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue([]);
  });

  it("should update tool permissions when user selects a tool", async () => {
    /**
     * Tests that clicking a tool checkbox calls onChange with updated permissions.
     * Pre-populates toolPermissions so the auto-populate logic on fetch is skipped,
     * and switches to flat view for predictable checkbox ordering.
     */
    const mockOnChange = vi.fn();
    const mockTools = [
      { name: "read_wiki_structure", description: "Get documentation topics" },
      { name: "read_wiki_contents", description: "View documentation" },
      { name: "ask_question", description: "Ask questions" },
    ];

    // Mock fetchMCPServers to return server details
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([
      {
        server_id: mockServerId,
        server_name: mockServerName,
        alias: mockServerName,
      },
    ]);

    // Mock listMCPTools to return tools for the server
    vi.mocked(networking.listMCPTools).mockResolvedValue({
      tools: mockTools,
      error: false,
    });

    // Pre-populate with all tools selected so auto-populate doesn't fire
    renderWithProviders(
      <MCPToolPermissions
        accessToken={mockAccessToken}
        selectedServers={[mockServerId]}
        toolPermissions={{ [mockServerId]: ["read_wiki_structure", "read_wiki_contents", "ask_question"] }}
        onChange={mockOnChange}
      />,
    );

    // Wait for server and tools to load
    await waitFor(() => {
      expect(screen.getByText(mockServerName)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText("read_wiki_structure")).toBeInTheDocument();
    });

    await userEvent.click(screen.getByText("Flat List"));
    expect(screen.getByRole("radio", { name: "Flat List" })).toBeChecked();
    expect(await screen.findByText("- Get documentation topics")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("checkbox", { name: "read_wiki_structure" }));

    // Unchecking clears the legacy allowlist key and writes the tool as denied instead,
    // so a tool the upstream adds later stays allowed
    expect(mockOnChange).toHaveBeenCalledWith({
      toolPermissions: {},
      deniedTools: { [mockServerId]: ["read_wiki_structure"] },
    });

    // Verify API calls
    // Note: useMCPServers uses useAuthorized() internally, which returns "123" from global mock
    expect(networking.fetchMCPServers).toHaveBeenCalledWith("123", undefined);
    // listMCPTools uses the accessToken prop directly
    expect(networking.listMCPTools).toHaveBeenCalledWith(mockAccessToken, mockServerId);
  });

  it("should select all tools when Select All button is clicked", async () => {
    const mockOnChange = vi.fn();
    const mockTools = [
      { name: "read_wiki_structure", description: "Get documentation topics" },
      { name: "read_wiki_contents", description: "View documentation" },
      { name: "ask_question", description: "Ask questions" },
    ];

    // Mock fetchMCPServers to return server details
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([
      {
        server_id: mockServerId,
        server_name: mockServerName,
        alias: mockServerName,
      },
    ]);

    // Mock listMCPTools to return tools for the server
    vi.mocked(networking.listMCPTools).mockResolvedValue({
      tools: mockTools,
      error: false,
    });

    renderWithProviders(
      <MCPToolPermissions
        accessToken={mockAccessToken}
        selectedServers={[mockServerId]}
        toolPermissions={{}}
        onChange={mockOnChange}
      />,
    );

    // Wait for server and tools to load
    await waitFor(() => {
      expect(screen.getByText(mockServerName)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText("read_wiki_structure")).toBeInTheDocument();
    });

    // Click the Select All button
    const selectAllButton = screen.getByRole("button", { name: "Select All" });
    await userEvent.click(selectAllButton);

    // Everything checked means nothing denied: the maps stay empty so a tool the
    // upstream adds later is still allowed
    expect(mockOnChange).toHaveBeenCalledWith({
      toolPermissions: {},
      deniedTools: {},
    });
  });

  it("should deselect all tools when Deselect All button is clicked", async () => {
    const mockOnChange = vi.fn();
    const mockTools = [
      { name: "read_wiki_structure", description: "Get documentation topics" },
      { name: "read_wiki_contents", description: "View documentation" },
      { name: "ask_question", description: "Ask questions" },
    ];

    // Mock fetchMCPServers to return server details
    vi.mocked(networking.fetchMCPServers).mockResolvedValue([
      {
        server_id: mockServerId,
        server_name: mockServerName,
        alias: mockServerName,
      },
    ]);

    // Mock listMCPTools to return tools for the server
    vi.mocked(networking.listMCPTools).mockResolvedValue({
      tools: mockTools,
      error: false,
    });

    renderWithProviders(
      <MCPToolPermissions
        accessToken={mockAccessToken}
        selectedServers={[mockServerId]}
        toolPermissions={{ [mockServerId]: ["read_wiki_structure", "read_wiki_contents"] }}
        onChange={mockOnChange}
      />,
    );

    // Wait for server and tools to load
    await waitFor(() => {
      expect(screen.getByText(mockServerName)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText("read_wiki_structure")).toBeInTheDocument();
    });

    // Click the Deselect All button
    const deselectAllButton = screen.getByRole("button", { name: "Deselect All" });
    await userEvent.click(deselectAllButton);

    // Nothing checked writes every fetched tool as denied; a later upstream tool
    // would then be allowed, which is the requested semantics
    expect(mockOnChange).toHaveBeenCalledWith({
      toolPermissions: {},
      deniedTools: { [mockServerId]: ["read_wiki_structure", "read_wiki_contents", "ask_question"] },
    });
  });

  describe("servers reached indirectly", () => {
    const groupServer = {
      server_id: "srv-group-1",
      server_name: "Group Server",
      alias: "Group Server",
      mcp_access_groups: ["production-group"],
    };
    const groupTools = [
      { name: "list_issues", description: "List issues" },
      { name: "delete_issue", description: "Delete an issue" },
    ];

    it("renders the tool matrix for a server granted only through an access group", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([groupServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedAccessGroups={["production-group"]}
          toolPermissions={{}}
          onChange={mockOnChange}
        />,
      );

      expect(await screen.findByText("Group Server")).toBeInTheDocument();
      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      expect(screen.getByText("delete_issue")).toBeInTheDocument();
      expect(networking.listMCPTools).toHaveBeenCalledWith(mockAccessToken, groupServer.server_id);
    });

    it("shows every tool selected in flat view for an unrestricted access-group server", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([groupServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedAccessGroups={["production-group"]}
          toolPermissions={{}}
          onChange={mockOnChange}
        />,
      );

      await screen.findByText("Group Server");
      await userEvent.click(screen.getByText("Flat List"));

      const [listIssues, deleteIssue] = screen.getAllByRole("checkbox");
      expect(listIssues).toBeChecked();
      expect(deleteIssue).toBeChecked();

      await userEvent.click(listIssues);
      expect(mockOnChange).toHaveBeenCalledWith({
        toolPermissions: {},
        deniedTools: { [groupServer.server_id]: ["list_issues"] },
      });
    });

    it("marks an access-group server as inherited and leaves a directly selected one unmarked", async () => {
      const directServer = { server_id: "srv-direct-1", server_name: "Direct Server", alias: "Direct Server" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([directServer, groupServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[directServer.server_id]}
          selectedAccessGroups={["production-group"]}
          toolPermissions={{ [directServer.server_id]: ["list_issues"], [groupServer.server_id]: ["list_issues"] }}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("Direct Server")).toBeInTheDocument();
      expect(await screen.findByText("Group Server")).toBeInTheDocument();
      expect(screen.getByText("Via access group: production-group")).toBeInTheDocument();
      expect(screen.queryAllByText(/^Via /)).toHaveLength(1);
    });

    it("renders a toolset server as inherited from that toolset", async () => {
      const toolsetServer = { server_id: "srv-toolset-1", server_name: "Toolset Server", alias: "Toolset Server" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([toolsetServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([
        {
          toolset_id: "ts-1",
          toolset_name: "Support Toolset",
          tools: [{ server_id: toolsetServer.server_id, tool_name: "list_issues" }],
        },
      ]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedToolsets={["ts-1"]}
          toolPermissions={{}}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("Toolset Server")).toBeInTheDocument();
      expect(screen.getByText("Via toolset: Support Toolset")).toBeInTheDocument();
    });

    // The backend adds a toolset's tools to whatever mcp_tool_permissions holds, so showing the
    // server as unrestricted would invite a deselection that grants every other tool on it.
    it("shows a toolset's own tools as the allowed set and locks them", async () => {
      const toolsetServer = { server_id: "srv-toolset-1", server_name: "Toolset Server", alias: "Toolset Server" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([toolsetServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([
        {
          toolset_id: "ts-1",
          toolset_name: "Support Toolset",
          tools: [{ server_id: toolsetServer.server_id, tool_name: "list_issues" }],
        },
      ]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedToolsets={["ts-1"]}
          toolPermissions={{}}
          onChange={mockOnChange}
        />,
      );

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      expect(
        screen.getByText(
          "list_issues is granted by a selected toolset, so it stays allowed here; edit the toolset to revoke it",
        ),
      ).toBeInTheDocument();

      await userEvent.click(screen.getByText("Flat List"));
      const [listIssues, deleteIssue] = screen.getAllByRole("checkbox");
      expect(listIssues).toBeChecked();
      expect(listIssues).toBeDisabled();
      // A server reached only through a toolset grants just that toolset's tools, matching the
      // backend's union over (keyed, toolset); every other fetched tool stays unchecked.
      expect(deleteIssue).not.toBeChecked();

      await userEvent.click(listIssues);
      expect(mockOnChange).not.toHaveBeenCalled();
    });

    it("ignores a click on a locked tool in the risk-group view", async () => {
      const toolsetServer = { server_id: "srv-toolset-1", server_name: "Toolset Server", alias: "Toolset Server" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([toolsetServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([
        {
          toolset_id: "ts-1",
          toolset_name: "Support Toolset",
          tools: [{ server_id: toolsetServer.server_id, tool_name: "list_issues" }],
        },
      ]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedToolsets={["ts-1"]}
          toolPermissions={{}}
          onChange={mockOnChange}
        />,
      );

      await userEvent.click(await screen.findByText("list_issues"));
      expect(mockOnChange).not.toHaveBeenCalled();
    });

    // Turning a risk group off must not drop a tool the entry grants in its own right, which the
    // toolset happens to grant too: that tool outlives the toolset and the admin did not clear it.
    it("keeps a locked tool the entry also grants when its risk group is turned off", async () => {
      const toolsetServer = { server_id: "srv-toolset-1", server_name: "Toolset Server", alias: "Toolset Server" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([toolsetServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([
        {
          toolset_id: "ts-1",
          toolset_name: "Support Toolset",
          tools: [{ server_id: toolsetServer.server_id, tool_name: "list_issues" }],
        },
      ]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedToolsets={["ts-1"]}
          toolPermissions={{ [toolsetServer.server_id]: ["list_issues"] }}
          onChange={mockOnChange}
        />,
      );

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      // First checkbox is the header toggle of the group holding list_issues.
      await userEvent.click(screen.getAllByRole("checkbox")[0]);

      // A toolset feeds this server, so the write stays an allowlist grant: list_issues is kept
      // under the server's key and delete_issue is simply absent, which the backend resolves the
      // same way (allowed = list_issues only)
      expect(mockOnChange).toHaveBeenCalledWith({
        toolPermissions: { [toolsetServer.server_id]: ["list_issues"] },
        deniedTools: {},
      });
    });

    // Copying the toolset's tools into the entry would outlive the toolset, so a write keeps only
    // what this level grants on its own.
    it("leaves a toolset's tools out of the entry a Select All writes", async () => {
      const toolsetServer = { server_id: "srv-toolset-1", server_name: "Toolset Server", alias: "Toolset Server" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([toolsetServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([
        {
          toolset_id: "ts-1",
          toolset_name: "Support Toolset",
          tools: [{ server_id: toolsetServer.server_id, tool_name: "list_issues" }],
        },
      ]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedToolsets={["ts-1"]}
          toolPermissions={{}}
          onChange={mockOnChange}
        />,
      );

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      await userEvent.click(screen.getByText("Select All"));

      // A denylist cannot grant, so a toolset-only server keeps the allowlist write: every fetched
      // tool outside the toolset lands under its key while the toolset's own tool is withheld.
      expect(mockOnChange).toHaveBeenCalledWith({
        toolPermissions: { [toolsetServer.server_id]: ["delete_issue"] },
        deniedTools: {},
      });
    });

    it("renders an outside tool checked after checking it on a toolset-only server", async () => {
      const toolsetServer = { server_id: "srv-toolset-1", server_name: "Toolset Server", alias: "Toolset Server" };
      const Harness = () => {
        const [write, setWrite] = useState<McpToolPermissionWrite>({ toolPermissions: {}, deniedTools: {} });
        return (
          <MCPToolPermissions
            accessToken={mockAccessToken}
            selectedServers={[]}
            selectedToolsets={["ts-1"]}
            toolPermissions={write.toolPermissions}
            deniedTools={write.deniedTools}
            onChange={setWrite}
          />
        );
      };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([toolsetServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([
        {
          toolset_id: "ts-1",
          toolset_name: "Support Toolset",
          tools: [{ server_id: toolsetServer.server_id, tool_name: "list_issues" }],
        },
      ]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      renderWithProviders(<Harness />);

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      await userEvent.click(screen.getByText("Flat List"));
      const deleteIssue = screen.getByRole("checkbox", { name: "delete_issue" });
      expect(deleteIssue).not.toBeChecked();

      await userEvent.click(deleteIssue);
      expect(deleteIssue).toBeChecked();
    });

    it("lifts a denied editable tool on a toolset-only server when it is re-checked", async () => {
      const toolsetServer = { server_id: "srv-toolset-1", server_name: "Toolset Server", alias: "Toolset Server" };
      const writes: McpToolPermissionWrite[] = [];
      const Harness = () => {
        const [write, setWrite] = useState<McpToolPermissionWrite>({
          toolPermissions: {},
          deniedTools: { [toolsetServer.server_id]: ["delete_issue"] },
        });
        return (
          <MCPToolPermissions
            accessToken={mockAccessToken}
            selectedServers={[]}
            selectedToolsets={["ts-1"]}
            toolPermissions={write.toolPermissions}
            deniedTools={write.deniedTools}
            onChange={(next) => {
              writes.push(next);
              setWrite(next);
            }}
          />
        );
      };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([toolsetServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([
        {
          toolset_id: "ts-1",
          toolset_name: "Support Toolset",
          tools: [{ server_id: toolsetServer.server_id, tool_name: "list_issues" }],
        },
      ]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      renderWithProviders(<Harness />);

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      await userEvent.click(screen.getByText("Flat List"));
      const deleteIssue = screen.getByRole("checkbox", { name: "delete_issue" });
      expect(deleteIssue).not.toBeChecked();

      await userEvent.click(deleteIssue);

      // The backend resolves deny over allow, so the write must lift the deny or the
      // re-checked tool would stay blocked and render unchecked again
      const last = writes.at(-1)!;
      expect(last.toolPermissions).toEqual({ [toolsetServer.server_id]: ["delete_issue"] });
      expect(last.deniedTools).toEqual({});
      expect(deleteIssue).toBeChecked();
    });

    it("keeps a denied locked toolset tool denied after Select All", async () => {
      const toolsetServer = { server_id: "srv-toolset-1", server_name: "Toolset Server", alias: "Toolset Server" };
      const mockOnChange = vi.fn();
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([toolsetServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([
        {
          toolset_id: "ts-1",
          toolset_name: "Support Toolset",
          tools: [{ server_id: toolsetServer.server_id, tool_name: "list_issues" }],
        },
      ]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedToolsets={["ts-1"]}
          toolPermissions={{}}
          deniedTools={{ [toolsetServer.server_id]: ["list_issues"] }}
          onChange={mockOnChange}
        />,
      );

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      await userEvent.click(screen.getByText("Flat List"));
      await userEvent.click(screen.getByText("Select All"));

      // Select All passes every fetched name including the locked toolset tool; the write must
      // not read that as the admin re-enabling it, or the stored deny silently disappears
      expect(mockOnChange).toHaveBeenCalledWith({
        toolPermissions: { [toolsetServer.server_id]: ["delete_issue"] },
        deniedTools: { [toolsetServer.server_id]: ["list_issues"] },
      });
    });

    it("keeps a denied locked toolset tool denied after its risk group is enabled", async () => {
      const toolsetServer = { server_id: "srv-toolset-1", server_name: "Toolset Server", alias: "Toolset Server" };
      const toolsetCrudTools = [
        { name: "list_documents", description: "List every document" },
        { name: "get_document", description: "Fetch one document" },
        { name: "delete_document", description: "Destroy a document" },
      ];
      const mockOnChange = vi.fn();
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([toolsetServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([
        {
          toolset_id: "ts-1",
          toolset_name: "Support Toolset",
          tools: [{ server_id: toolsetServer.server_id, tool_name: "list_documents" }],
        },
      ]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: toolsetCrudTools, error: false });

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedToolsets={["ts-1"]}
          toolPermissions={{}}
          deniedTools={{ [toolsetServer.server_id]: ["list_documents"] }}
          onChange={mockOnChange}
        />,
      );

      // Enabling a risk group re-checks its locked tools too, so the write must keep their
      // stored denies instead of reading the bulk check as the admin re-enabling them
      const readGroupToggle = await screen.findByRole("checkbox", { name: "Allow all Read tools" });
      await userEvent.click(readGroupToggle);

      expect(mockOnChange).toHaveBeenCalledWith({
        toolPermissions: { [toolsetServer.server_id]: ["get_document"] },
        deniedTools: { [toolsetServer.server_id]: ["list_documents"] },
      });
    });

    it("keeps a tool-permission-only server listed and granting after Select All", async () => {
      const keyedServer = { server_id: "srv-keyed-1", server_name: "Keyed Server", alias: "Keyed Server" };
      const Harness = () => {
        const [write, setWrite] = useState<McpToolPermissionWrite>({
          toolPermissions: { [keyedServer.server_id]: ["list_issues"] },
          deniedTools: {},
        });
        return (
          <MCPToolPermissions
            accessToken={mockAccessToken}
            selectedServers={[]}
            toolPermissions={write.toolPermissions}
            deniedTools={write.deniedTools}
            onChange={setWrite}
          />
        );
      };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([keyedServer]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      renderWithProviders(<Harness />);

      expect(await screen.findByText("Keyed Server")).toBeInTheDocument();
      await userEvent.click(screen.getByText("Select All"));

      // The allowlist is the server's only grant, so Select All widens it rather than clearing it.
      expect(screen.getByText("Keyed Server")).toBeInTheDocument();
      await userEvent.click(screen.getByText("Flat List"));
      const [listIssues, deleteIssue] = screen.getAllByRole("checkbox");
      expect(listIssues).toBeChecked();
      expect(deleteIssue).toBeChecked();
    });

    // The default narrows an unrestricted server; against a toolset-restricted one it would widen
    // the grant to every non-delete tool the server exposes.
    it("does not write the delete-blocked default for a directly selected server a toolset restricts", async () => {
      const directServer = { server_id: "srv-direct-1", server_name: "Direct Server", alias: "Direct Server" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([directServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([
        {
          toolset_id: "ts-1",
          toolset_name: "Support Toolset",
          tools: [{ server_id: directServer.server_id, tool_name: "list_issues" }],
        },
      ]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[directServer.server_id]}
          selectedToolsets={["ts-1"]}
          toolPermissions={{}}
          onChange={mockOnChange}
        />,
      );

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      expect(mockOnChange).not.toHaveBeenCalled();
    });

    it("waits for toolsets before writing the delete-blocked default", async () => {
      const directServer = { server_id: "srv-direct-1", server_name: "Direct Server", alias: "Direct Server" };
      let resolveToolsets: (toolsets: MCPToolset[]) => void = () => {};
      const pendingToolsets = new Promise<MCPToolset[]>((resolve) => {
        resolveToolsets = resolve;
      });
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([directServer]);
      vi.mocked(networking.fetchMCPToolsets).mockReturnValue(pendingToolsets);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[directServer.server_id]}
          selectedToolsets={["ts-1"]}
          toolPermissions={{}}
          onChange={mockOnChange}
        />,
      );

      await screen.findByText("Direct Server");
      expect(mockOnChange).not.toHaveBeenCalled();

      resolveToolsets([
        {
          toolset_id: "ts-1",
          toolset_name: "Support Toolset",
          tools: [{ server_id: directServer.server_id, tool_name: "list_issues" }],
        },
      ]);

      await screen.findByText("list_issues");
      expect(screen.getByRole("checkbox", { name: "list_issues" })).toHaveAttribute("aria-disabled", "true");
      expect(mockOnChange).not.toHaveBeenCalled();
    });

    // The backend resolves a selection that is a registry id to that server alone. Rendering the
    // server merely named after it would fire the default write against a server nobody granted,
    // and a tool-permission entry is itself a grant.
    it.each([
      { label: "id owner first", idOwnerFirst: true },
      { label: "name twin first", idOwnerFirst: false },
    ])("does not offer a server merely named after a selected id ($label)", async ({ idOwnerFirst }) => {
      const idOwner = { server_id: "srv-collide", server_name: "Payments", alias: "Payments" };
      const nameTwin = { server_id: "srv-twin", server_name: "srv-collide", alias: "srv-collide" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue(idOwnerFirst ? [idOwner, nameTwin] : [nameTwin, idOwner]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={["srv-collide"]}
          toolPermissions={{}}
          onChange={mockOnChange}
        />,
      );

      expect(await screen.findByText("Payments")).toBeInTheDocument();
      expect(screen.queryByText("srv-collide")).not.toBeInTheDocument();
      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      expect(mockOnChange).not.toHaveBeenCalled();
      expect(networking.listMCPTools).not.toHaveBeenCalledWith(mockAccessToken, "srv-twin");
    });

    it("does not write a default allowlist for an inherited server", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([groupServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedAccessGroups={["production-group"]}
          toolPermissions={{}}
          onChange={mockOnChange}
        />,
      );

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      expect(mockOnChange).not.toHaveBeenCalled();
    });

    it("writes nothing when a directly selected server's tools load", async () => {
      const directServer = { server_id: "srv-direct-1", server_name: "Direct Server", alias: "Direct Server" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([directServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[directServer.server_id]}
          toolPermissions={{}}
          onChange={mockOnChange}
        />,
      );

      // The old enumerated-allowlist write is gone: an unchecked box is the only signal, so the
      // fetch itself emits nothing and every fetched tool renders checked
      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      expect(mockOnChange).not.toHaveBeenCalled();

      await userEvent.click(screen.getByText("Flat List"));
      const [listIssues, deleteIssue] = screen.getAllByRole("checkbox");
      expect(listIssues).toBeChecked();
      expect(deleteIssue).toBeChecked();
    });

    // No allowlist and no toolset narrows the server, so every fetched tool is checked except the
    // ones the denylist names; a tool the upstream adds later renders checked too.
    it("checks every fetched tool but the denied ones on an otherwise unrestricted server", async () => {
      const directServer = { server_id: "srv-direct-1", server_name: "Direct Server", alias: "Direct Server" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([directServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[directServer.server_id]}
          toolPermissions={{}}
          deniedTools={{ [directServer.server_id]: ["delete_issue"] }}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("list_issues")).toBeInTheDocument();

      await userEvent.click(screen.getByText("Flat List"));
      const [listIssues, deleteIssue] = screen.getAllByRole("checkbox");
      expect(listIssues).toBeChecked();
      expect(deleteIssue).not.toBeChecked();
    });

    it("shows a server that only a stale tool-permission entry still entitles", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([groupServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedAccessGroups={[]}
          toolPermissions={{ [groupServer.server_id]: ["list_issues"] }}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("Group Server")).toBeInTheDocument();
      expect(screen.getByText("Via tool permissions")).toBeInTheDocument();
    });

    it("shows nothing for a principal blocked from every MCP server", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([groupServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      const { container } = renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[NO_MCP_SERVERS_SENTINEL]}
          toolPermissions={{ [groupServer.server_id]: ["list_issues"] }}
          onChange={vi.fn()}
        />,
      );

      expect(container).toBeEmptyDOMElement();
      expect(networking.listMCPTools).not.toHaveBeenCalled();
    });

    it("warns instead of showing no inherited servers when the server list cannot be loaded", async () => {
      vi.mocked(networking.fetchMCPServers).mockRejectedValue(new Error("boom"));
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedAccessGroups={["production-group"]}
          toolPermissions={{}}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("Unable to load MCP servers")).toBeInTheDocument();
      expect(screen.queryByText(/has 0 servers/)).not.toBeInTheDocument();
    });

    it("tells the admin when a loaded access group has no member servers", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([groupServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: groupTools, error: false });

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedAccessGroups={["production-group", "ops_readonly"]}
          toolPermissions={{}}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText('Access group "ops_readonly" has 0 servers')).toBeInTheDocument();
      expect(screen.getByText("Group Server")).toBeInTheDocument();
      expect(screen.queryByText('Access group "production-group" has 0 servers')).not.toBeInTheDocument();
    });

    it("does not call a group empty when its servers are only hidden from the caller's catalog", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);
      vi.mocked(networking.fetchMCPAccessGroups).mockResolvedValue(["production-group"]);

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedAccessGroups={["production-group", "ops_readonly"]}
          toolPermissions={{}}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText('Access group "ops_readonly" has 0 servers')).toBeInTheDocument();
      expect(screen.queryByText('Access group "production-group" has 0 servers')).not.toBeInTheDocument();
    });

    it("warns when the selected toolsets cannot be resolved to servers", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([]);
      vi.mocked(networking.fetchMCPToolsets).mockRejectedValue(new Error("boom"));

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedToolsets={["ts-1"]}
          toolPermissions={{}}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("Unable to load toolsets")).toBeInTheDocument();
    });
  });

  describe("grants keyed by server name", () => {
    const namedServer = {
      server_id: "1f4bd6c1-0000-4000-8000-000000000001",
      server_name: "github_mcp",
      alias: "GitHub",
    };
    const namedTools = [
      { name: "list_issues", description: "List issues" },
      { name: "delete_issue", description: "Delete an issue" },
    ];

    it("renders the tool matrix for a grant that names the server instead of its id", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([namedServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: namedTools, error: false });

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={["github_mcp"]}
          toolPermissions={{ github_mcp: ["list_issues"] }}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("github_mcp")).toBeInTheDocument();
      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      expect(screen.getByText("delete_issue")).toBeInTheDocument();
      expect(networking.listMCPTools).toHaveBeenCalledWith(mockAccessToken, namedServer.server_id);
    });

    it("writes an edit back to the name key instead of adding a second id-keyed entry", async () => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([namedServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: namedTools, error: false });

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={["github_mcp"]}
          toolPermissions={{ github_mcp: ["list_issues"] }}
          onChange={mockOnChange}
        />,
      );

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      await userEvent.click(screen.getByRole("button", { name: "Deselect All" }));

      expect(mockOnChange).toHaveBeenCalledWith({
        toolPermissions: {},
        deniedTools: { github_mcp: ["list_issues", "delete_issue"] },
      });
    });
  });

  describe("a server named by several equivalent keys", () => {
    const namedServer = {
      server_id: "1f4bd6c1-0000-4000-8000-000000000001",
      server_name: "github_mcp",
      alias: "GitHub",
      mcp_access_groups: ["production-group"],
    };
    const namedTools = [
      { name: "list_issues", description: "List issues" },
      { name: "create_issue", description: "Open an issue" },
      { name: "delete_issue", description: "Delete an issue" },
    ];

    const renderWithBothKeys = (onChange: () => void) =>
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[namedServer.server_id]}
          toolPermissions={{ [namedServer.server_id]: ["list_issues"], github_mcp: ["create_issue"] }}
          onChange={onChange}
        />,
      );

    beforeEach(() => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([namedServer]);
      vi.mocked(networking.fetchMCPToolsets).mockResolvedValue([]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: namedTools, error: false });
    });

    it("renders one card showing the union both keys grant", async () => {
      renderWithBothKeys(vi.fn());

      expect(await screen.findByText("github_mcp")).toBeInTheDocument();
      expect(screen.getAllByText("github_mcp")).toHaveLength(1);
      expect(await screen.findByText("list_issues")).toBeInTheDocument();

      // Flat view keeps checkbox order identical to the fetched tool order.
      await userEvent.click(screen.getByText("Flat List"));
      const [listIssues, createIssue, deleteIssue] = screen.getAllByRole("checkbox");
      expect(listIssues).toBeChecked();
      expect(createIssue).toBeChecked();
      expect(deleteIssue).not.toBeChecked();
    });

    it("removes a deselected tool from every equivalent key, leaving one entry for the server", async () => {
      const mockOnChange = vi.fn();
      renderWithBothKeys(mockOnChange);

      expect(await screen.findByText("list_issues")).toBeInTheDocument();
      await userEvent.click(screen.getByText("Flat List"));
      await userEvent.click(screen.getAllByRole("checkbox")[0]);

      const written = mockOnChange.mock.calls.at(-1)?.[0] as {
        toolPermissions: Record<string, string[]>;
        deniedTools: Record<string, string[]>;
      };
      // The unchecked tool lands in the denylist under the server's own key while every
      // equivalent allowlist key for the server is cleared
      expect(Object.keys(written.toolPermissions)).toEqual([]);
      expect(written.deniedTools[namedServer.server_id]).toContain("list_issues");
      expect(written.deniedTools[namedServer.server_id]).toContain("delete_issue");
      expect(written.deniedTools[namedServer.server_id]).not.toContain("create_issue");
    });

    // Both catalog orders, because a name resolves to two servers here and a first-match
    // implementation is only wrong in one of them.
    it.each([
      { label: "edited server first", editedFirst: true },
      { label: "twin first", editedFirst: false },
    ])(
      "says on the card when a key names another server too, since its tools cannot be revoked here ($label)",
      async ({ editedFirst }) => {
        const twin = { server_id: "1f4bd6c1-0000-4000-8000-000000000002", server_name: "github_mcp", alias: "Twin" };
        vi.mocked(networking.fetchMCPServers).mockResolvedValue(
          editedFirst ? [namedServer, twin] : [twin, namedServer],
        );

        renderWithProviders(
          <MCPToolPermissions
            accessToken={mockAccessToken}
            selectedServers={[namedServer.server_id]}
            toolPermissions={{ [namedServer.server_id]: ["list_issues"], github_mcp: ["create_issue"] }}
            onChange={vi.fn()}
          />,
        );

        // Both cards say it: the shared key grants on either server and neither card can revoke it,
        // so an admin looking at either one has to be told the same thing.
        expect(
          await screen.findAllByText(
            'Also granted by "github_mcp", which names another server too. Those tools stay allowed here until the servers no longer share that name',
          ),
        ).toHaveLength(2);
      },
    );

    // The shared key is the twin's only entry, so it would otherwise be the key an edit writes,
    // and writing it would move the allowlist of the server the admin is not looking at.
    it.each([
      { label: "edited server first", editedFirst: true },
      { label: "twin first", editedFirst: false },
    ])("edits the twin through its own id rather than the shared key ($label)", async ({ editedFirst }) => {
      const twin = { server_id: "1f4bd6c1-0000-4000-8000-000000000002", server_name: "github_mcp", alias: "Twin" };
      vi.mocked(networking.fetchMCPServers).mockResolvedValue(editedFirst ? [namedServer, twin] : [twin, namedServer]);

      const mockOnChange = vi.fn();
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[twin.server_id]}
          toolPermissions={{ github_mcp: ["list_issues"] }}
          onChange={mockOnChange}
        />,
      );

      // The directly selected twin is the first card; both share the display name "github_mcp".
      expect(await screen.findAllByText("list_issues")).toHaveLength(2);
      await userEvent.click(screen.getAllByText("Select All")[0]);

      const written = mockOnChange.mock.calls.at(-1)?.[0] as {
        toolPermissions: Record<string, string[]>;
        deniedTools: Record<string, string[]>;
      };
      // The shared key is ambiguous, so it survives; Select All writes no denylist entry at all
      expect(written.toolPermissions["github_mcp"]).toEqual(["list_issues"]);
      expect(written.deniedTools).toEqual({});
    });

    it("says nothing about shared names when every key names one server", async () => {
      renderWithBothKeys(vi.fn());

      expect(await screen.findByText("github_mcp")).toBeInTheDocument();
      expect(screen.queryByText(/names another server too/)).not.toBeInTheDocument();
    });

    it("badges the server once, by its strongest grant, when a key and a group both name it", async () => {
      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[]}
          selectedAccessGroups={["production-group"]}
          toolPermissions={{ github_mcp: ["list_issues"] }}
          onChange={vi.fn()}
        />,
      );

      expect(await screen.findByText("github_mcp")).toBeInTheDocument();
      expect(screen.getByText("Via access group: production-group")).toBeInTheDocument();
      expect(screen.queryByText("Via tool permissions")).not.toBeInTheDocument();
      expect(screen.queryAllByText(/^Via /)).toHaveLength(1);
    });
  });

  describe("risk-group (CRUD) view", () => {
    const crudTools = [
      { name: "list_documents", description: "List every document" },
      { name: "get_document", description: "Fetch one document" },
      { name: "delete_document", description: "Destroy a document" },
    ];
    const allCrudToolNames = crudTools.map((t) => t.name);

    const renderCrudView = (toolPermissions: Record<string, string[]>, onChange: () => void) => {
      vi.mocked(networking.fetchMCPServers).mockResolvedValue([
        { server_id: mockServerId, server_name: mockServerName, alias: mockServerName },
      ]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: crudTools, error: false });

      renderWithProviders(
        <MCPToolPermissions
          accessToken={mockAccessToken}
          selectedServers={[mockServerId]}
          toolPermissions={toolPermissions}
          onChange={onChange}
        />,
      );
    };

    it("removes a whole risk group from the saved payload when its group toggle is cleared", async () => {
      const mockOnChange = vi.fn();
      renderCrudView({ [mockServerId]: allCrudToolNames }, mockOnChange);

      const readGroupToggle = await screen.findByRole("checkbox", { name: "Allow all Read tools" });
      expect(readGroupToggle).toBeChecked();

      await userEvent.click(readGroupToggle);

      expect(mockOnChange).toHaveBeenCalledWith({
        toolPermissions: {},
        deniedTools: { [mockServerId]: ["list_documents", "get_document"] },
      });
    });

    it("adds the rest of a partially-allowed risk group when its mixed toggle is clicked", async () => {
      const mockOnChange = vi.fn();
      renderCrudView({ [mockServerId]: ["list_documents"] }, mockOnChange);

      const readGroupToggle = await screen.findByRole("checkbox", { name: "Allow all Read tools" });
      expect(readGroupToggle).toBePartiallyChecked();

      await userEvent.click(readGroupToggle);

      expect(mockOnChange).toHaveBeenCalledWith({
        toolPermissions: {},
        deniedTools: { [mockServerId]: ["delete_document"] },
      });
    });

    it("toggles a single tool exactly once when its checkbox is clicked inside the clickable row", async () => {
      const mockOnChange = vi.fn();
      renderCrudView({ [mockServerId]: allCrudToolNames }, mockOnChange);

      await userEvent.click(await screen.findByRole("checkbox", { name: "delete_document" }));

      expect(mockOnChange).toHaveBeenCalledTimes(1);
      expect(mockOnChange).toHaveBeenCalledWith({
        toolPermissions: {},
        deniedTools: { [mockServerId]: ["delete_document"] },
      });
    });

    it("toggles a single tool when the row around its checkbox is clicked", async () => {
      const mockOnChange = vi.fn();
      renderCrudView({ [mockServerId]: allCrudToolNames }, mockOnChange);

      await userEvent.click(await screen.findByText("Destroy a document"));

      expect(mockOnChange).toHaveBeenCalledTimes(1);
      expect(mockOnChange).toHaveBeenCalledWith({
        toolPermissions: {},
        deniedTools: { [mockServerId]: ["delete_document"] },
      });
    });

    it("re-renders each checkbox from the denylist it emitted", async () => {
      const Harness = () => {
        const [write, setWrite] = useState<McpToolPermissionWrite>({ toolPermissions: {}, deniedTools: {} });
        return (
          <>
            <MCPToolPermissions
              accessToken={mockAccessToken}
              selectedServers={[mockServerId]}
              toolPermissions={write.toolPermissions}
              deniedTools={write.deniedTools}
              onChange={setWrite}
            />
            <output>{(write.deniedTools[mockServerId] ?? []).join(",")}</output>
          </>
        );
      };

      vi.mocked(networking.fetchMCPServers).mockResolvedValue([
        { server_id: mockServerId, server_name: mockServerName, alias: mockServerName },
      ]);
      vi.mocked(networking.listMCPTools).mockResolvedValue({ tools: crudTools, error: false });
      renderWithProviders(<Harness />);

      const deleteTool = await screen.findByRole("checkbox", { name: "delete_document" });
      const readGroupToggle = screen.getByRole("checkbox", { name: "Allow all Read tools" });
      expect(deleteTool).toBeChecked();
      expect(readGroupToggle).toBeChecked();

      await userEvent.click(deleteTool);
      expect(deleteTool).not.toBeChecked();
      expect(screen.getByRole("status")).toHaveTextContent("delete_document");

      await userEvent.click(screen.getByRole("checkbox", { name: "get_document" }));
      expect(readGroupToggle).toBePartiallyChecked();
      expect(screen.getByRole("status")).toHaveTextContent("get_document,delete_document");

      await userEvent.click(readGroupToggle);
      expect(readGroupToggle).toBeChecked();
      expect(screen.getByRole("status")).toHaveTextContent("delete_document");
      expect(screen.getByRole("checkbox", { name: "delete_document" })).not.toBeChecked();
    });
  });
});
