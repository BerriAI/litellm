import { useDeleteProxyConfigField, useProxyConfig } from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import { useStoreRequestInSpendLogs } from "@/app/(dashboard)/hooks/storeRequestInSpendLogs/useStoreRequestInSpendLogs";
import NotificationsManager from "@/components/molecules/notifications_manager";
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
  parseErrorMessage: vi.fn((error: unknown) => (error as Error)?.message || String(error)),
}));

const mockUseStoreRequestInSpendLogs = vi.mocked(useStoreRequestInSpendLogs);
const mockUseProxyConfig = vi.mocked(useProxyConfig);
const mockUseDeleteProxyConfigField = vi.mocked(useDeleteProxyConfigField);
const mockNotificationsManager = vi.mocked(NotificationsManager);

describe("SpendLogsSettingsModal", () => {
  const mockOnCancel = vi.fn();
  const mockOnSuccess = vi.fn();
  const mockMutate = vi.fn();
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
      mutate: mockMutate,
      isPending: false,
    } as any);
    mockUseDeleteProxyConfigField.mockReturnValue({
      mutate: mockDeleteField,
      isPending: false,
    } as any);
    mockUseProxyConfig.mockReturnValue({
      data: [],
      isLoading: false,
      refetch: mockRefetch,
    } as any);
  });

  it("should render the modal with form fields and action buttons", () => {
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Spend Logs Settings")).toBeInTheDocument();
    expect(screen.getByText("Store Prompts in Spend Logs")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g., 7d, 30d")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save Settings" })).toBeInTheDocument();
  });

  it("should not render when isVisible is false", () => {
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} isVisible={false} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("should call onCancel when cancel button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(mockOnCancel).toHaveBeenCalledTimes(1);
  });

  it("should submit with store_prompts_in_spend_logs toggled on and a retention period", async () => {
    const user = userEvent.setup();
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    await user.click(screen.getByRole("switch"));
    await user.type(screen.getByPlaceholderText("e.g., 7d, 30d"), "30d");
    await user.click(screen.getByRole("button", { name: "Save Settings" }));

    await waitFor(() => {
      expect(mockMutate).toHaveBeenCalledWith(
        { store_prompts_in_spend_logs: true, maximum_spend_logs_retention_period: "30d" },
        expect.any(Object),
      );
      expect(mockNotificationsManager.success).toHaveBeenCalledWith("Spend logs settings updated successfully");
      expect(mockOnSuccess).toHaveBeenCalled();
    });
  });

  it("should delete retention field then submit when retention period is empty", async () => {
    const user = userEvent.setup();
    mockDeleteField.mockImplementation((_params, options) => {
      options?.onSettled?.();
    });
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    await user.click(screen.getByRole("button", { name: "Save Settings" }));

    await waitFor(() => {
      expect(mockDeleteField).toHaveBeenCalled();
      expect(mockMutate).toHaveBeenCalledWith({ store_prompts_in_spend_logs: false }, expect.any(Object));
    });
  });

  it("should populate form from existing config data", () => {
    mockUseProxyConfig.mockReturnValue({
      data: [
        {
          field_name: "store_prompts_in_spend_logs",
          field_value: true,
          field_type: "bool",
          field_description: "",
          stored_in_db: true,
          field_default_value: false,
        },
        {
          field_name: "maximum_spend_logs_retention_period",
          field_value: "7d",
          field_type: "string",
          field_description: "",
          stored_in_db: true,
          field_default_value: undefined,
        },
      ],
      isLoading: false,
      refetch: mockRefetch,
    } as any);

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    expect(screen.getByRole("switch")).toBeChecked();
    expect(screen.getByPlaceholderText("e.g., 7d, 30d")).toHaveValue("7d");
  });

  it("should show skeleton loaders when config is loading", () => {
    mockUseProxyConfig.mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: mockRefetch,
    } as any);

    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);

    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("e.g., 7d, 30d")).not.toBeInTheDocument();
  });

  it("should refetch config when modal opens", () => {
    renderWithProviders(<SpendLogsSettingsModal {...defaultProps} />);
    expect(mockRefetch).toHaveBeenCalledTimes(1);
  });
});
