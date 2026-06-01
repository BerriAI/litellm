import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MCPToolConfiguration from "./mcp_tool_configuration";

const tools = [
  { name: "read_user", description: "Read user" },
  { name: "delete_user", description: "Delete user" },
];

const renderToolConfiguration = (onAllowedToolsChange = vi.fn()) => {
  render(
    <MCPToolConfiguration
      accessToken="token"
      formValues={{ url: "https://example.com/mcp", transport: "http", auth_type: "none" }}
      allowedTools={[]}
      existingAllowedTools={null}
      onAllowedToolsChange={onAllowedToolsChange}
      toolNameToDisplayName={{}}
      toolNameToDescription={{}}
      onToolNameToDisplayNameChange={vi.fn()}
      onToolNameToDescriptionChange={vi.fn()}
      externalTools={tools}
      externalCanFetch
      isEditMode
    />,
  );

  return onAllowedToolsChange;
};

describe("MCPToolConfiguration", () => {
  it("shows legacy unrestricted edit tools enabled in flat view", async () => {
    const onAllowedToolsChange = renderToolConfiguration();

    fireEvent.click(screen.getByText("Flat List"));

    expect(screen.getByText("2 of 2 tools enabled for user access")).toBeInTheDocument();
    expect(screen.getAllByText("Enabled")).toHaveLength(2);

    fireEvent.click(screen.getByText("read_user"));

    await waitFor(() => {
      expect(onAllowedToolsChange).toHaveBeenLastCalledWith(["delete_user"]);
    });
  });
});
