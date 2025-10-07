import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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
     * This is the core functionality of the component.
     */
    const mockOnChange = vi.fn();
    const mockTools = [
      { name: "read_wiki_structure", description: "Get documentation topics" },
      { name: "read_wiki_contents", description: "View documentation" },
      { name: "ask_question", description: "Ask questions" },
    ];

    // Mock fetchMCPServers to return server details
    vi.mocked(networking.fetchMCPServers).mockResolvedValue({
      data: [
        {
          server_id: mockServerId,
          server_name: mockServerName,
          alias: mockServerName,
        },
      ],
    });

    // Mock listMCPTools to return tools for the server
    vi.mocked(networking.listMCPTools).mockResolvedValue({
      tools: mockTools,
      error: false,
    });

    render(
      <MCPToolPermissions
        accessToken={mockAccessToken}
        selectedServers={[mockServerId]}
        toolPermissions={{}}
        onChange={mockOnChange}
      />
    );

    // Wait for server and tools to load
    await waitFor(() => {
      expect(screen.getByText(mockServerName)).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText("read_wiki_structure")).toBeInTheDocument();
    });

    // Get all checkboxes and click the first one (for read_wiki_structure)
    const checkboxes = screen.getAllByRole("checkbox");
    await userEvent.click(checkboxes[0]);

    // Verify onChange was called with correct permissions
    expect(mockOnChange).toHaveBeenCalledWith({
      [mockServerId]: ["read_wiki_structure"],
    });

    // Verify API calls
    expect(networking.fetchMCPServers).toHaveBeenCalledWith(mockAccessToken);
    expect(networking.listMCPTools).toHaveBeenCalledWith(mockAccessToken, mockServerId);
  });
});

