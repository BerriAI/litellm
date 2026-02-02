import { useDeleteProxyConfigField, useProxyConfig } from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import { useStoreRequestInSpendLogs } from "@/app/(dashboard)/hooks/storeRequestInSpendLogs/useStoreRequestInSpendLogs";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import SpendLogsSettingsModal from "./SpendLogsSettingsModal";

vi.mock("@/app/(dashboard)/hooks/storeRequestInSpendLogs/useStoreRequestInSpendLogs");
vi.mock("@/app/(dashboard)/hooks/proxyConfig/useProxyConfig");
vi.mock("@/components/molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));
vi.mock("@/components/shared/errorUtils", () => ({
  parseErrorMessage: vi.fn(),
}));

const mockUseStoreRequestInSpendLogs = vi.mocked(useStoreRequestInSpendLogs);
const mockUseProxyConfig = vi.mocked(useProxyConfig);
const mockUseDeleteProxyConfigField = vi.mocked(useDeleteProxyConfigField);
const mockNotificationsManager = vi.mocked(NotificationsManager);
const mockParseErrorMessage = vi.mocked(parseErrorMessage);

describe("SpendLogsSettingsModal", () => {
  const mockOnCancel = vi.fn();
  const mockOnSuccess = vi.fn();
  const mockMutateAsync = vi.fn();
  const mockDeleteField = vi.fn();
  const mockRefetch = vi.fn();

  const defaultProps = {
    isVisible: true,
    onCancel: mockOnCancel,
    onSuccess: mockOnSuccess,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseStoreRequestInSpendLogs.mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: false,
    } as any);
    mockUseDeleteProxyConfigField.mockReturnValue({
      mutateAsync: mockDeleteField,
      isPending: false,
    } as any);
    mockUseProxyConfig.mockReturnValue({
      data: [],
      isLoading: false,
      refetch: mockRefetch,
    } as any);
    mockParseErrorMessage.mockImplementation((error: any) => error?.message || String(error));
  });

  it("should render the modal", () => {
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Spend Logs Settings")).toBeInTheDocument();
  });

  it("should render form fields with initial values", () => {
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    expect(screen.getByText("Store Prompts in Spend Logs")).toBeInTheDocument();
    expect(screen.getByLabelText("Maximum Spend Logs Retention Period (Optional)")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g., 7d, 30d")).toBeInTheDocument();
  });

  it("should render cancel and save buttons", () => {
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save Settings" })).toBeInTheDocument();
  });

  it("should call onCancel when cancel button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    await user.click(cancelButton);

    expect(mockOnCancel).toHaveBeenCalledTimes(1);
  });

  it("should call onCancel when modal close button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const closeButton = screen.getByRole("button", { name: /close/i });
    await user.click(closeButton);

    expect(mockOnCancel).toHaveBeenCalledTimes(1);
  });

  it("should toggle store prompts switch", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const switchElement = screen.getByRole("switch");
    expect(switchElement).not.toBeChecked();

    await user.click(switchElement);

    await waitFor(() => {
      expect(switchElement).toBeChecked();
    });
  });

  it("should update retention period input", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const retentionInput = screen.getByPlaceholderText("e.g., 7d, 30d");
    await user.type(retentionInput, "30d");

    expect(retentionInput).toHaveValue("30d");
  });

  it("should submit form with store prompts enabled and retention period", async () => {
    const user = userEvent.setup();
    mockMutateAsync.mockImplementation(async (params, options) => {
      await Promise.resolve();
      options?.onSuccess?.();
      return { message: "Success" };
    });

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const switchElement = screen.getByRole("switch");
    await user.click(switchElement);

    const retentionInput = screen.getByPlaceholderText("e.g., 7d, 30d");
    await user.type(retentionInput, "30d");

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockDeleteField).not.toHaveBeenCalled();
      expect(mockMutateAsync).toHaveBeenCalledWith(
        {
          store_prompts_in_spend_logs: true,
          maximum_spend_logs_retention_period: "30d",
        },
        expect.any(Object)
      );
    });
  });

  it("should submit form with store prompts disabled and no retention period", async () => {
    const user = userEvent.setup();
    mockDeleteField.mockResolvedValue({ message: "Field deleted successfully" });
    mockMutateAsync.mockImplementation(async (params, options) => {
      await Promise.resolve();
      options?.onSuccess?.();
      return { message: "Success" };
    });

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockDeleteField).toHaveBeenCalled();
      expect(mockMutateAsync).toHaveBeenCalledWith(
        {
          store_prompts_in_spend_logs: false,
        },
        expect.any(Object)
      );
    });
  });

  it("should show success notification and call onSuccess on successful submission", async () => {
    const user = userEvent.setup();
    mockDeleteField.mockResolvedValue({ message: "Field deleted successfully" });
    mockMutateAsync.mockImplementation(async (params, options) => {
      await Promise.resolve();
      options?.onSuccess?.();
      return { message: "Success" };
    });

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockNotificationsManager.success).toHaveBeenCalledWith("Spend logs settings updated successfully");
      expect(mockRefetch).toHaveBeenCalled();
      expect(mockOnSuccess).toHaveBeenCalledTimes(1);
    });
  });

  it("should show error notification when submission fails", async () => {
    const user = userEvent.setup();
    const error = new Error("Network error");
    mockMutateAsync.mockRejectedValue(error);
    mockParseErrorMessage.mockReturnValue("Network error");

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockNotificationsManager.fromBackend).toHaveBeenCalledWith("Failed to save spend logs settings: Network error");
    });
  });

  it("should show error notification from onError callback", async () => {
    const user = userEvent.setup();
    const error = new Error("Backend error");
    mockMutateAsync.mockImplementation((params, options) => {
      options?.onError?.(error);
      return Promise.reject(error);
    });
    mockParseErrorMessage.mockReturnValue("Backend error");

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockNotificationsManager.fromBackend).toHaveBeenCalledWith("Failed to save spend logs settings: Backend error");
    });
  });

  it("should disable cancel button when pending", () => {
    mockUseStoreRequestInSpendLogs.mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: true,
    } as any);

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    expect(cancelButton).toBeDisabled();
  });

  it("should disable cancel button when deleting field", () => {
    mockUseDeleteProxyConfigField.mockReturnValue({
      mutateAsync: mockDeleteField,
      isPending: true,
    } as any);

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    expect(cancelButton).toBeDisabled();
  });

  it("should disable cancel button when loading config", () => {
    mockUseProxyConfig.mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: mockRefetch,
    } as any);

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    expect(cancelButton).toBeDisabled();
  });

  it("should show loading state on save button when pending", () => {
    mockUseStoreRequestInSpendLogs.mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: true,
    } as any);

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const saveButton = screen.getByRole("button", { name: /Saving/i });
    expect(saveButton).toBeInTheDocument();
    expect(saveButton.className).toContain("ant-btn-loading");
  });

  it("should show loading state on save button when deleting field", () => {
    mockUseDeleteProxyConfigField.mockReturnValue({
      mutateAsync: mockDeleteField,
      isPending: true,
    } as any);

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const saveButton = screen.getByRole("button", { name: /Saving/i });
    expect(saveButton).toBeInTheDocument();
    expect(saveButton.className).toContain("ant-btn-loading");
  });

  it("should call onCancel when cancel button is clicked after modifying form", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const switchElement = screen.getByRole("switch");
    await user.click(switchElement);

    const retentionInput = screen.getByPlaceholderText("e.g., 7d, 30d");
    await user.type(retentionInput, "30d");

    expect(switchElement).toBeChecked();
    expect(retentionInput).toHaveValue("30d");

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    await user.click(cancelButton);

    expect(mockOnCancel).toHaveBeenCalledTimes(1);
  });

  it("should call refetch after successful submission", async () => {
    const user = userEvent.setup();
    mockDeleteField.mockResolvedValue({ message: "Field deleted successfully" });
    mockMutateAsync.mockImplementation(async (params, options) => {
      await Promise.resolve();
      options?.onSuccess?.();
      return { message: "Success" };
    });

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const switchElement = screen.getByRole("switch");
    await user.click(switchElement);

    const retentionInput = screen.getByPlaceholderText("e.g., 7d, 30d");
    await user.type(retentionInput, "30d");

    expect(switchElement).toBeChecked();
    expect(retentionInput).toHaveValue("30d");

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockNotificationsManager.success).toHaveBeenCalled();
      expect(mockRefetch).toHaveBeenCalled();
    });
  });

  it("should not call onSuccess when it is not provided", async () => {
    const user = userEvent.setup();
    mockDeleteField.mockResolvedValue({ message: "Field deleted successfully" });
    mockMutateAsync.mockImplementation(async (params, options) => {
      await Promise.resolve();
      options?.onSuccess?.();
      return { message: "Success" };
    });

    renderWithProviders(<SpendLogsSettingsModal isVisible={true} onCancel={mockOnCancel} />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockNotificationsManager.success).toHaveBeenCalled();
    });
  });

  it("should not render modal when isVisible is false", () => {
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} isVisible={false} />);

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("should call refetch when modal opens", () => {
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    expect(mockRefetch).toHaveBeenCalledTimes(1);
  });

  it("should render form with initial values from config data", () => {
    mockUseProxyConfig.mockReturnValue({
      data: [
        {
          field_name: "store_prompts_in_spend_logs",
          field_type: "bool",
          field_description: "Store prompts in spend logs",
          field_value: true,
          stored_in_db: true,
          field_default_value: false,
        },
        {
          field_name: "maximum_spend_logs_retention_period",
          field_type: "string",
          field_description: "Maximum retention period",
          field_value: "30d",
          stored_in_db: true,
          field_default_value: undefined,
        },
      ],
      isLoading: false,
      refetch: mockRefetch,
    } as any);

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const switchElement = screen.getByRole("switch");
    const retentionInput = screen.getByPlaceholderText("e.g., 7d, 30d");

    expect(switchElement).toBeChecked();
    expect(retentionInput).toHaveValue("30d");
  });

  it("should show skeleton loaders when config is loading", () => {
    mockUseProxyConfig.mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: mockRefetch,
    } as any);

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    // Check that switch and input are not present when loading (skeletons are shown instead)
    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("e.g., 7d, 30d")).not.toBeInTheDocument();

    // Check for skeleton elements (Ant Design Skeleton.Input renders with ant-skeleton class)
    const skeletons = document.querySelectorAll(".ant-skeleton");
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it("should continue with update even if deleteField fails", async () => {
    const user = userEvent.setup();
    const deleteError = new Error("Field does not exist");
    mockDeleteField.mockRejectedValue(deleteError);
    mockMutateAsync.mockImplementation(async (params, options) => {
      await Promise.resolve();
      options?.onSuccess?.();
      return { message: "Success" };
    });

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockDeleteField).toHaveBeenCalled();
      expect(mockMutateAsync).toHaveBeenCalledWith(
        {
          store_prompts_in_spend_logs: false,
        },
        expect.any(Object)
      );
      expect(mockNotificationsManager.success).toHaveBeenCalled();
    });
  });

  it("should submit form with only store prompts enabled and no retention period", async () => {
    const user = userEvent.setup();
    mockDeleteField.mockResolvedValue({ message: "Field deleted successfully" });
    mockMutateAsync.mockImplementation(async (params, options) => {
      await Promise.resolve();
      options?.onSuccess?.();
      return { message: "Success" };
    });

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    const switchElement = screen.getByRole("switch");
    await user.click(switchElement);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockDeleteField).toHaveBeenCalled();
      expect(mockMutateAsync).toHaveBeenCalledWith(
        {
          store_prompts_in_spend_logs: true,
        },
        expect.any(Object)
      );
    });
  });
});
