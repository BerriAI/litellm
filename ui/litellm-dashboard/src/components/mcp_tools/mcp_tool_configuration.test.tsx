import { useState } from "react";
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
  it("can start new-server onboarding in the flat checklist view", async () => {
    render(
      <MCPToolConfiguration
        accessToken="token"
        formValues={{ url: "https://example.com/mcp", transport: "http", auth_type: "none" }}
        allowedTools={[]}
        existingAllowedTools={null}
        onAllowedToolsChange={vi.fn()}
        toolNameToDisplayName={{}}
        toolNameToDescription={{}}
        onToolNameToDisplayNameChange={vi.fn()}
        onToolNameToDescriptionChange={vi.fn()}
        externalTools={tools}
        externalCanFetch
        defaultViewMode="flat"
      />,
    );

    expect(screen.getByLabelText("Flat List")).toBeChecked();
  });

  it("toggles a flat-list checkbox once without bubbling to the row", async () => {
    const onAllowedToolsChange = vi.fn();

    const Wrapper = () => {
      const [allowedTools, setAllowedTools] = useState<string[]>([]);

      return (
        <MCPToolConfiguration
          accessToken="token"
          formValues={{ url: "https://example.com/mcp", transport: "http", auth_type: "none" }}
          allowedTools={allowedTools}
          existingAllowedTools={null}
          onAllowedToolsChange={(nextAllowedTools) => {
            onAllowedToolsChange(nextAllowedTools);
            setAllowedTools(nextAllowedTools);
          }}
          toolNameToDisplayName={{}}
          toolNameToDescription={{}}
          onToolNameToDisplayNameChange={vi.fn()}
          onToolNameToDescriptionChange={vi.fn()}
          externalTools={tools}
          externalCanFetch
          defaultViewMode="flat"
        />
      );
    };

    render(<Wrapper />);

    await waitFor(() => {
      expect(screen.getByText("2 of 2 tools enabled for user access")).toBeInTheDocument();
    });
    onAllowedToolsChange.mockClear();

    fireEvent.click(screen.getAllByRole("checkbox")[0]);

    await waitFor(() => {
      expect(onAllowedToolsChange).toHaveBeenCalledTimes(1);
      expect(onAllowedToolsChange).toHaveBeenCalledWith(["delete_user"]);
    });
  });

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

  it("preserves a restored selection on a legacy server instead of clearing it on init", async () => {
    const onAllowedToolsChange = vi.fn();

    render(
      <MCPToolConfiguration
        accessToken="token"
        formValues={{ url: "https://example.com/mcp", transport: "http", auth_type: "none" }}
        allowedTools={["read_user"]}
        existingAllowedTools={null}
        hasToolAllowlistInteraction
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

    await waitFor(() => {
      expect(onAllowedToolsChange).toHaveBeenCalled();
    });
    expect(onAllowedToolsChange).toHaveBeenLastCalledWith(["read_user"]);
    expect(onAllowedToolsChange).not.toHaveBeenCalledWith([]);
  });

  it("keeps legacy unrestricted tools disabled after all are toggled off", async () => {
    const Wrapper = () => {
      const [allowedTools, setAllowedTools] = useState<string[]>([]);
      const [hasToolAllowlistInteraction, setHasToolAllowlistInteraction] = useState(false);

      return (
        <MCPToolConfiguration
          accessToken="token"
          formValues={{ url: "https://example.com/mcp", transport: "http", auth_type: "none" }}
          allowedTools={allowedTools}
          existingAllowedTools={null}
          hasToolAllowlistInteraction={hasToolAllowlistInteraction}
          onToolAllowlistInteraction={() => setHasToolAllowlistInteraction(true)}
          onAllowedToolsChange={setAllowedTools}
          toolNameToDisplayName={{}}
          toolNameToDescription={{}}
          onToolNameToDisplayNameChange={vi.fn()}
          onToolNameToDescriptionChange={vi.fn()}
          externalTools={tools}
          externalCanFetch
          isEditMode
        />
      );
    };

    render(<Wrapper />);

    fireEvent.click(screen.getByText("Flat List"));
    fireEvent.click(screen.getByText("read_user"));
    fireEvent.click(screen.getByText("delete_user"));

    await waitFor(() => {
      expect(screen.getByText("0 of 2 tools enabled for user access")).toBeInTheDocument();
      expect(screen.getAllByText("Disabled")).toHaveLength(2);
    });
  });
});
