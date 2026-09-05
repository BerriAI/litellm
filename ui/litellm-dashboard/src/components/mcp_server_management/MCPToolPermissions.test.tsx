import { useState } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import MCPToolPermissions from "./MCPToolPermissions";
import * as networking from "../networking";

vi.mock("../networking");

describe("MCPToolPermissions", () => {
  const mockAccessToken = "test-token";
  const mockServerId = "server-123";
  const mockServerName = "Test MCP Server";

  beforeEach(() => {
    vi.clearAllMocks();
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

    // Verify onChange was called with read_wiki_structure removed
    expect(mockOnChange).toHaveBeenCalledWith({
      [mockServerId]: ["read_wiki_contents", "ask_question"],
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

    // Verify onChange was called with all tools selected
    expect(mockOnChange).toHaveBeenCalledWith({
      [mockServerId]: ["read_wiki_structure", "read_wiki_contents", "ask_question"],
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

    // Verify onChange was called with no tools selected
    expect(mockOnChange).toHaveBeenCalledWith({
      [mockServerId]: [],
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

      expect(mockOnChange).toHaveBeenCalledWith({ [mockServerId]: ["delete_document"] });
    });

    it("adds the rest of a partially-allowed risk group when its mixed toggle is clicked", async () => {
      const mockOnChange = vi.fn();
      renderCrudView({ [mockServerId]: ["list_documents"] }, mockOnChange);

      const readGroupToggle = await screen.findByRole("checkbox", { name: "Allow all Read tools" });
      expect(readGroupToggle).toBePartiallyChecked();

      await userEvent.click(readGroupToggle);

      expect(mockOnChange).toHaveBeenCalledWith({ [mockServerId]: ["list_documents", "get_document"] });
    });

    it("toggles a single tool exactly once when its checkbox is clicked inside the clickable row", async () => {
      const mockOnChange = vi.fn();
      renderCrudView({ [mockServerId]: allCrudToolNames }, mockOnChange);

      await userEvent.click(await screen.findByRole("checkbox", { name: "delete_document" }));

      expect(mockOnChange).toHaveBeenCalledTimes(1);
      expect(mockOnChange).toHaveBeenCalledWith({ [mockServerId]: ["list_documents", "get_document"] });
    });

    it("toggles a single tool when the row around its checkbox is clicked", async () => {
      const mockOnChange = vi.fn();
      renderCrudView({ [mockServerId]: allCrudToolNames }, mockOnChange);

      await userEvent.click(await screen.findByText("Destroy a document"));

      expect(mockOnChange).toHaveBeenCalledTimes(1);
      expect(mockOnChange).toHaveBeenCalledWith({ [mockServerId]: ["list_documents", "get_document"] });
    });

    it("re-renders each checkbox from the permissions it emitted", async () => {
      const Harness = () => {
        const [permissions, setPermissions] = useState<Record<string, string[]>>({
          [mockServerId]: allCrudToolNames,
        });
        return (
          <>
            <MCPToolPermissions
              accessToken={mockAccessToken}
              selectedServers={[mockServerId]}
              toolPermissions={permissions}
              onChange={setPermissions}
            />
            <output>{(permissions[mockServerId] ?? []).join(",")}</output>
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
      expect(screen.getByRole("status")).toHaveTextContent("list_documents,get_document");

      await userEvent.click(screen.getByRole("checkbox", { name: "get_document" }));
      expect(readGroupToggle).toBePartiallyChecked();
      expect(screen.getByRole("status")).toHaveTextContent("list_documents");

      await userEvent.click(readGroupToggle);
      expect(readGroupToggle).toBeChecked();
      expect(screen.getByRole("status")).toHaveTextContent("list_documents,get_document");
      expect(screen.getByRole("checkbox", { name: "delete_document" })).not.toBeChecked();
    });
  });
});
