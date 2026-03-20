import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import Fallbacks from "./Fallbacks";
import * as networkingModule from "../../../networking";
import * as fetchModelsModule from "../../../playground/llm_calls/fetch_models";

vi.mock("../../../networking", () => ({
  getCallbacksCall: vi.fn(),
  setCallbacksCall: vi.fn(),
}));

vi.mock("../../../playground/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: vi.fn().mockReturnValue({ data: null }),
}));

vi.mock("openai", () => ({
  default: {
    OpenAI: vi.fn().mockImplementation(() => ({
      chat: {
        completions: {
          create: vi.fn(),
        },
      },
    })),
  },
}));

vi.mock("../../../common_components/DeleteResourceModal", () => ({
  __esModule: true,
  default: ({ isOpen, onOk, onCancel, title, message, resourceInformation, confirmLoading }: any) => {
    if (!isOpen) return null;
    return (
      <div data-testid="delete-modal">
        <div>{title}</div>
        <div>{message}</div>
        {resourceInformation?.map((info: any, idx: number) => (
          <div key={idx}>
            {info.label}: {info.value}
          </div>
        ))}
        <button onClick={onCancel} disabled={confirmLoading}>
          Cancel
        </button>
        <button onClick={onOk} disabled={confirmLoading}>
          Delete
        </button>
      </div>
    );
  },
}));

vi.mock("./AddFallbacks", () => ({
  __esModule: true,
  default: ({ value, onChange }: any) => {
    const handleClick = async () => {
      if (onChange) {
        try {
          const newFallbacks = [...(value || []), { "test-model": ["test-fallback"] }];
          await onChange(newFallbacks);
        } catch (error) {
          // Error is handled by the component
        }
      }
    };
    return (
      <button onClick={handleClick} data-testid="add-fallbacks-button">
        Add Fallbacks
      </button>
    );
  },
}));

describe("Fallbacks", () => {
  const mockAccessToken = "test-token";
  const mockUserRole = "Admin";
  const mockUserID = "user-123";
  const mockModelData = {
    data: [
      { model_name: "gpt-4" },
      { model_name: "gpt-3.5-turbo" },
      { model_name: "claude-3-opus" },
    ],
  };

  const mockRouterSettings = {
    fallbacks: [
      { "gpt-4": ["gpt-3.5-turbo", "claude-3-opus"] },
      { "claude-3-opus": ["gpt-4"] },
    ],
  };

  const defaultProps = {
    accessToken: mockAccessToken,
    userRole: mockUserRole,
    userID: mockUserID,
    modelData: mockModelData,
  };

  const getFirstRowDeleteButton = () => {
    const deleteButtons = screen.getAllByTestId("delete-fallback-button");
    return deleteButtons.length > 0 ? deleteButtons[0] : null;
  };

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networkingModule.getCallbacksCall).mockResolvedValue({
      router_settings: mockRouterSettings,
    });
    vi.mocked(networkingModule.setCallbacksCall).mockResolvedValue(undefined);
    vi.mocked(fetchModelsModule.fetchAvailableModels).mockResolvedValue([
      { model_group: "gpt-4", mode: "chat" },
      { model_group: "gpt-3.5-turbo", mode: "chat" },
      { model_group: "claude-3-opus", mode: "chat" },
    ]);
  });

  it("should render the component", async () => {
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByTestId("add-fallbacks-button")).toBeInTheDocument();
    });
  });

  it("should not render when accessToken is null", () => {
    const { container } = render(<Fallbacks {...defaultProps} accessToken={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("should fetch router settings on mount", async () => {
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(networkingModule.getCallbacksCall).toHaveBeenCalledWith(
        mockAccessToken,
        mockUserID,
        mockUserRole,
      );
    });
  });

  it("should display fallback entries in table", async () => {
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getAllByText("gpt-4").length).toBeGreaterThan(0);
      expect(screen.getAllByText(/gpt-3\.5-turbo/).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/claude-3-opus/).length).toBeGreaterThan(0);
    });
  });

  it("should show delete button for each fallback row when fallbacks exist", async () => {
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getAllByText("gpt-4").length).toBeGreaterThan(0);
    });

    const deleteButtons = screen.getAllByTestId("delete-fallback-button");
    expect(deleteButtons.length).toBe(2);
  });

  it("should open delete modal when delete icon is clicked", async () => {
    const user = userEvent.setup();
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getAllByText("gpt-4").length).toBeGreaterThan(0);
    });

    const deleteButton = getFirstRowDeleteButton();
    expect(deleteButton).not.toBeNull();

    await user.click(deleteButton as HTMLElement);

    await waitFor(() => {
      expect(screen.getByTestId("delete-modal")).toBeInTheDocument();
      expect(screen.getByText("Delete Fallback?")).toBeInTheDocument();
    });
  });

  it("should delete fallback when confirmed", async () => {
    const user = userEvent.setup();
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getAllByText("gpt-4").length).toBeGreaterThan(0);
    });

    const deleteButton = getFirstRowDeleteButton();
    expect(deleteButton).not.toBeNull();

    await user.click(deleteButton as HTMLElement);

    await waitFor(() => {
      expect(screen.getByTestId("delete-modal")).toBeInTheDocument();
    });

    const confirmButton = screen.getByRole("button", { name: /delete/i });
    await user.click(confirmButton);

    await waitFor(() => {
      expect(networkingModule.setCallbacksCall).toHaveBeenCalled();
      const callArgs = (networkingModule.setCallbacksCall as any).mock.calls[0];
      expect(callArgs[0]).toBe(mockAccessToken);
      expect(callArgs[1].router_settings.fallbacks).toHaveLength(1);
    });
  });

  it("should close delete modal when cancel is clicked", async () => {
    const user = userEvent.setup();
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getAllByText("gpt-4").length).toBeGreaterThan(0);
    });

    const deleteButton = getFirstRowDeleteButton();
    expect(deleteButton).not.toBeNull();

    await user.click(deleteButton as HTMLElement);

    await waitFor(() => {
      expect(screen.getByTestId("delete-modal")).toBeInTheDocument();
    });

    const cancelButton = screen.getByRole("button", { name: /cancel/i });
    await user.click(cancelButton);

    await waitFor(() => {
      expect(screen.queryByTestId("delete-modal")).not.toBeInTheDocument();
    });
  });

  it("should show error notification on delete failure", async () => {
    const user = userEvent.setup();
    const error = new Error("Delete failed");
    vi.mocked(networkingModule.setCallbacksCall).mockRejectedValueOnce(error);
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getAllByText("gpt-4").length).toBeGreaterThan(0);
    });

    const deleteButton = getFirstRowDeleteButton();
    expect(deleteButton).not.toBeNull();

    await user.click(deleteButton as HTMLElement);

    await waitFor(() => {
      expect(screen.getByTestId("delete-modal")).toBeInTheDocument();
    });

    const confirmButton = screen.getByRole("button", { name: /delete/i });
    await user.click(confirmButton);

    await waitFor(() => {
      expect(networkingModule.setCallbacksCall).toHaveBeenCalled();
    });
  });

  it("should handle delete error gracefully", async () => {
    const user = userEvent.setup();
    const error = new Error("Delete failed");
    vi.mocked(networkingModule.setCallbacksCall).mockRejectedValueOnce(error);
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getAllByText("gpt-4").length).toBeGreaterThan(0);
    });

    const deleteButton = getFirstRowDeleteButton();
    expect(deleteButton).not.toBeNull();

    await user.click(deleteButton as HTMLElement);

    await waitFor(() => {
      expect(screen.getByTestId("delete-modal")).toBeInTheDocument();
    });

    const confirmButton = screen.getByRole("button", { name: /delete/i });
    await user.click(confirmButton);

    await waitFor(() => {
      expect(networkingModule.setCallbacksCall).toHaveBeenCalled();
      expect(screen.queryByTestId("delete-modal")).not.toBeInTheDocument();
    });
  });

  it("should handle empty fallbacks array", async () => {
    vi.mocked(networkingModule.getCallbacksCall).mockResolvedValueOnce({
      router_settings: { fallbacks: [] },
    });
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByTestId("add-fallbacks-button")).toBeInTheDocument();
      expect(
        screen.getByText(/No fallbacks configured. Add fallbacks to automatically try another model/),
      ).toBeInTheDocument();
    });

    expect(screen.queryByText("gpt-4")).not.toBeInTheDocument();
  });

  it("should handle router settings without fallbacks property", async () => {
    vi.mocked(networkingModule.getCallbacksCall).mockResolvedValueOnce({
      router_settings: {},
    });
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByTestId("add-fallbacks-button")).toBeInTheDocument();
      expect(
        screen.getByText(/No fallbacks configured. Add fallbacks to automatically try another model/),
      ).toBeInTheDocument();
    });
  });

  it("should remove model_group_retry_policy from router settings", async () => {
    vi.mocked(networkingModule.getCallbacksCall).mockResolvedValueOnce({
      router_settings: {
        ...mockRouterSettings,
        model_group_retry_policy: { some: "policy" },
      },
    });
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(networkingModule.getCallbacksCall).toHaveBeenCalled();
    });
  });

  it("should update fallbacks when AddFallbacks onChange is called", async () => {
    const user = userEvent.setup();
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByTestId("add-fallbacks-button")).toBeInTheDocument();
    });

    const addButton = screen.getByTestId("add-fallbacks-button");
    await user.click(addButton);

    await waitFor(() => {
      expect(networkingModule.setCallbacksCall).toHaveBeenCalled();
    });
  });

  it("should handle fallbacks change error and refetch", async () => {
    const user = userEvent.setup();
    const error = new Error("Update failed");
    vi.mocked(networkingModule.setCallbacksCall).mockRejectedValueOnce(error);
    vi.mocked(networkingModule.getCallbacksCall).mockResolvedValue({
      router_settings: mockRouterSettings,
    });
    render(<Fallbacks {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByTestId("add-fallbacks-button")).toBeInTheDocument();
    });

    const addButton = screen.getByTestId("add-fallbacks-button");
    await user.click(addButton);

    await waitFor(() => {
      expect(networkingModule.setCallbacksCall).toHaveBeenCalled();
    });

    await waitFor(
      () => {
        expect(networkingModule.getCallbacksCall).toHaveBeenCalledTimes(2);
      },
      { timeout: 3000 },
    );
  });
});
