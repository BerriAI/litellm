import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AddFallbacks, { Fallbacks } from "./AddFallbacks";
import * as fetchModelsModule from "../../../playground/llm_calls/fetch_models";

vi.mock("../../../playground/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(),
}));

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();
  return {
    ...actual,
    message: {
      error: vi.fn(),
    },
  };
});

vi.mock("./FallbackSelectionForm", () => ({
  FallbackSelectionForm: ({ groups, onGroupsChange }: any) => {
    const handleUpdateGroup = () => {
      if (groups.length > 0) {
        const updatedGroups = groups.map((group: any, index: number) => {
          if (index === 0 && !group.primaryModel) {
            return {
              ...group,
              primaryModel: "gpt-4",
              fallbackModels: ["gpt-3.5-turbo"],
            };
          }
          return group;
        });
        onGroupsChange(updatedGroups);
      }
    };

    return (
      <div data-testid="fallback-selection-form">
        <button onClick={handleUpdateGroup} data-testid="update-group-button">
          Update Group
        </button>
        <div data-testid="groups-count">{groups.length}</div>
        {groups.map((group: any) => (
          <div key={group.id} data-testid={`group-${group.id}`}>
            Primary: {group.primaryModel || "None"}, Fallbacks: {group.fallbackModels.length}
          </div>
        ))}
      </div>
    );
  },
}));

describe("AddFallbacks", () => {
  const mockOnChange = vi.fn();
  const mockAccessToken = "test-token";
  const mockModelGroups = [
    { model_group: "gpt-4", mode: "chat" },
    { model_group: "gpt-3.5-turbo", mode: "chat" },
    { model_group: "claude-3-opus", mode: "chat" },
  ];

  const defaultProps = {
    accessToken: mockAccessToken,
    value: [] as Fallbacks,
    onChange: mockOnChange,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchModelsModule.fetchAvailableModels).mockResolvedValue(mockModelGroups);
  });

  it("should render the component", () => {
    render(<AddFallbacks {...defaultProps} />);
    expect(screen.getByRole("button", { name: /add fallbacks/i })).toBeInTheDocument();
  });

  it("should open modal when Add Fallbacks button is clicked", async () => {
    const user = userEvent.setup();
    render(<AddFallbacks {...defaultProps} />);

    const addButton = screen.getByRole("button", { name: /add fallbacks/i });
    await user.click(addButton);

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });
  });

  it("should fetch available models when modal opens", async () => {
    const user = userEvent.setup();
    render(<AddFallbacks {...defaultProps} />);

    const addButton = screen.getByRole("button", { name: /add fallbacks/i });
    await user.click(addButton);

    await waitFor(() => {
      expect(fetchModelsModule.fetchAvailableModels).toHaveBeenCalledWith(mockAccessToken);
    });
  });

  it("should close modal when Cancel button is clicked", async () => {
    const user = userEvent.setup();
    render(<AddFallbacks {...defaultProps} />);

    const addButton = screen.getByRole("button", { name: /add fallbacks/i });
    await user.click(addButton);

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    const cancelButton = screen.getByRole("button", { name: /cancel/i });
    await user.click(cancelButton);

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
  });

  it("should show error when saving incomplete groups", async () => {
    const user = userEvent.setup();
    const antd = await import("antd");
    render(<AddFallbacks {...defaultProps} />);

    const addButton = screen.getByRole("button", { name: /add fallbacks/i });
    await user.click(addButton);

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    await waitFor(() => {
      const saveButton = screen.getByRole("button", { name: /save all configurations/i });
      expect(saveButton).toBeInTheDocument();
    });

    const saveButton = screen.getByRole("button", { name: /save all configurations/i });
    await user.click(saveButton);

    await waitFor(() => {
      expect(antd.message.error).toHaveBeenCalled();
    });
  });

  it("should show error message when saving incomplete groups", async () => {
    const user = userEvent.setup();
    const antd = await import("antd");
    render(<AddFallbacks {...defaultProps} />);

    const addButton = screen.getByRole("button", { name: /add fallbacks/i });
    await user.click(addButton);

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    const saveButton = screen.getByRole("button", { name: /save all configurations/i });
    await user.click(saveButton);

    await waitFor(() => {
      expect(antd.message.error).toHaveBeenCalled();
    });
  });

  it("should call onChange with new fallbacks when Save is clicked with valid configuration", async () => {
    const user = userEvent.setup();
    mockOnChange.mockResolvedValue(undefined);
    render(<AddFallbacks {...defaultProps} />);

    const addButton = screen.getByRole("button", { name: /add fallbacks/i });
    await user.click(addButton);

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByTestId("fallback-selection-form")).toBeInTheDocument();
    });

    const updateGroupButton = screen.getByTestId("update-group-button");
    await user.click(updateGroupButton);

    await waitFor(() => {
      expect(screen.getByText(/Primary: gpt-4/i)).toBeInTheDocument();
    });

    const saveButton = screen.getByRole("button", { name: /save all configurations/i });
    expect(saveButton).not.toBeDisabled();
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled();
      const callArgs = mockOnChange.mock.calls[0][0];
      expect(callArgs).toHaveLength(1);
      expect(callArgs[0]).toHaveProperty("gpt-4");
      expect(callArgs[0]["gpt-4"]).toContain("gpt-3.5-turbo");
    });
  });

  it("should append new fallbacks to existing value", async () => {
    const user = userEvent.setup();
    const existingFallbacks: Fallbacks = [{ "existing-model": ["fallback-1"] }];
    mockOnChange.mockResolvedValue(undefined);
    render(<AddFallbacks {...defaultProps} value={existingFallbacks} />);

    const addButton = screen.getByRole("button", { name: /add fallbacks/i });
    await user.click(addButton);

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByTestId("fallback-selection-form")).toBeInTheDocument();
    });

    const updateGroupButton = screen.getByTestId("update-group-button");
    await user.click(updateGroupButton);

    await waitFor(() => {
      expect(screen.getByText(/Primary: gpt-4/i)).toBeInTheDocument();
    });

    const saveButton = screen.getByRole("button", { name: /save all configurations/i });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled();
      const callArgs = mockOnChange.mock.calls[0][0];
      expect(callArgs).toHaveLength(2);
      expect(callArgs[0]).toEqual({ "existing-model": ["fallback-1"] });
      expect(callArgs[1]).toHaveProperty("gpt-4");
    });
  });

  it("should reset form state when modal is closed", async () => {
    const user = userEvent.setup();
    render(<AddFallbacks {...defaultProps} />);

    const addButton = screen.getByRole("button", { name: /add fallbacks/i });
    await user.click(addButton);

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    const cancelButton = screen.getByRole("button", { name: /cancel/i });
    await user.click(cancelButton);

    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    await user.click(addButton);

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
      expect(screen.getByTestId("fallback-selection-form")).toBeInTheDocument();
    });
  });

  it("should handle onChange error gracefully", async () => {
    const user = userEvent.setup();
    const error = new Error("Save failed");
    mockOnChange.mockRejectedValue(error);
    render(<AddFallbacks {...defaultProps} />);

    const addButton = screen.getByRole("button", { name: /add fallbacks/i });
    await user.click(addButton);

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByTestId("fallback-selection-form")).toBeInTheDocument();
    });

    const updateGroupButton = screen.getByTestId("update-group-button");
    await user.click(updateGroupButton);

    await waitFor(() => {
      expect(screen.getByText(/Primary: gpt-4/i)).toBeInTheDocument();
    });

    const saveButton = screen.getByRole("button", { name: /save all configurations/i });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalled();
    });
  });

  it("should not call onChange when onChange prop is not provided", async () => {
    const user = userEvent.setup();
    render(<AddFallbacks accessToken={mockAccessToken} value={[]} />);

    const addButton = screen.getByRole("button", { name: /add fallbacks/i });
    await user.click(addButton);

    await waitFor(() => {
      expect(screen.getByRole("dialog")).toBeInTheDocument();
    });
  });
});
