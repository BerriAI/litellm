import { useProxyConfig } from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import { useStoreModelInDB } from "@/app/(dashboard)/hooks/storeModelInDB/useStoreModelInDB";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import ModelSettingsModal from "./ModelSettingsModal";

vi.mock("@/app/(dashboard)/hooks/storeModelInDB/useStoreModelInDB");
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

const mockUseStoreModelInDB = vi.mocked(useStoreModelInDB);
const mockUseProxyConfig = vi.mocked(useProxyConfig);
const mockNotificationsManager = vi.mocked(NotificationsManager);
const mockParseErrorMessage = vi.mocked(parseErrorMessage);

describe("ModelSettingsModal", () => {
  const mockOnCancel = vi.fn();
  const mockOnSuccess = vi.fn();
  const mockMutateAsync = vi.fn();
  const mockRefetch = vi.fn();

  const defaultProps = {
    isVisible: true,
    onCancel: mockOnCancel,
    onSuccess: mockOnSuccess,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseStoreModelInDB.mockReturnValue({
      mutateAsync: mockMutateAsync,
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
    renderWithProviders(<ModelSettingsModal {...defaultProps} />);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Model Settings")).toBeInTheDocument();
  });

  it("should render form field with initial values", () => {
    renderWithProviders(<ModelSettingsModal {...defaultProps} />);
    expect(screen.getByText("Store Model in DB")).toBeInTheDocument();
  });

  it("should render cancel and save buttons", () => {
    renderWithProviders(<ModelSettingsModal {...defaultProps} />);
    expect(screen.getByRole("button", { name: "Cancel" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save Settings" })).toBeInTheDocument();
  });

  it("should call onCancel when cancel button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ModelSettingsModal {...defaultProps} />);

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    await user.click(cancelButton);

    expect(mockOnCancel).toHaveBeenCalledTimes(1);
  });

  it("should call onCancel when modal close button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ModelSettingsModal {...defaultProps} />);

    const closeButton = screen.getByRole("button", { name: /close/i });
    await user.click(closeButton);

    expect(mockOnCancel).toHaveBeenCalledTimes(1);
  });

  it("should toggle store model switch", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ModelSettingsModal {...defaultProps} />);

    const switchElement = screen.getByRole("switch");
    expect(switchElement).not.toBeChecked();

    await user.click(switchElement);

    await waitFor(() => {
      expect(switchElement).toBeChecked();
    });
  });

  it("should submit form with store_model_in_db enabled", async () => {
    const user = userEvent.setup();
    mockMutateAsync.mockImplementation(async (params, options) => {
      await Promise.resolve();
      options?.onSuccess?.();
      return { message: "Success" };
    });

    renderWithProviders(<ModelSettingsModal {...defaultProps} />);

    const switchElement = screen.getByRole("switch");
    await user.click(switchElement);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockMutateAsync).toHaveBeenCalledWith(
        { store_model_in_db: true },
        expect.any(Object)
      );
    });
  });

  it("should submit form with store_model_in_db disabled", async () => {
    const user = userEvent.setup();
    mockMutateAsync.mockImplementation(async (params, options) => {
      await Promise.resolve();
      options?.onSuccess?.();
      return { message: "Success" };
    });

    renderWithProviders(<ModelSettingsModal {...defaultProps} />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockMutateAsync).toHaveBeenCalledWith(
        { store_model_in_db: false },
        expect.any(Object)
      );
    });
  });

  it("should show success notification and call onSuccess on successful submission", async () => {
    const user = userEvent.setup();
    mockMutateAsync.mockImplementation(async (params, options) => {
      await Promise.resolve();
      options?.onSuccess?.();
      return { message: "Success" };
    });

    renderWithProviders(<ModelSettingsModal {...defaultProps} />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockNotificationsManager.success).toHaveBeenCalledWith("Model storage settings updated successfully");
      expect(mockRefetch).toHaveBeenCalled();
      expect(mockOnSuccess).toHaveBeenCalledTimes(1);
    });
  });

  it("should show error notification when submission fails", async () => {
    const user = userEvent.setup();
    const error = new Error("Network error");
    mockMutateAsync.mockRejectedValue(error);
    mockParseErrorMessage.mockReturnValue("Network error");

    renderWithProviders(<ModelSettingsModal {...defaultProps} />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockNotificationsManager.fromBackend).toHaveBeenCalledWith("Failed to save model storage settings: Network error");
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

    renderWithProviders(<ModelSettingsModal {...defaultProps} />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockNotificationsManager.fromBackend).toHaveBeenCalledWith("Failed to save model storage settings: Backend error");
    });
  });

  it("should disable cancel button when pending", () => {
    mockUseStoreModelInDB.mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: true,
    } as any);

    renderWithProviders(<ModelSettingsModal {...defaultProps} />);

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    expect(cancelButton).toBeDisabled();
  });

  it("should disable cancel button when loading config", () => {
    mockUseProxyConfig.mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: mockRefetch,
    } as any);

    renderWithProviders(<ModelSettingsModal {...defaultProps} />);

    const cancelButton = screen.getByRole("button", { name: "Cancel" });
    expect(cancelButton).toBeDisabled();
  });

  it("should show loading state on save button when pending", () => {
    mockUseStoreModelInDB.mockReturnValue({
      mutateAsync: mockMutateAsync,
      isPending: true,
    } as any);

    renderWithProviders(<ModelSettingsModal {...defaultProps} />);

    const saveButton = screen.getByRole("button", { name: /Saving/i });
    expect(saveButton).toBeInTheDocument();
    expect(saveButton.className).toContain("ant-btn-loading");
  });

  it("should not render modal when isVisible is false", () => {
    renderWithProviders(<ModelSettingsModal {...defaultProps} isVisible={false} />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("should call refetch when modal opens", () => {
    renderWithProviders(<ModelSettingsModal {...defaultProps} />);
    expect(mockRefetch).toHaveBeenCalledTimes(1);
  });

  it("should render form with initial values from config data", () => {
    mockUseProxyConfig.mockReturnValue({
      data: [
        {
          field_name: "store_model_in_db",
          field_type: "bool",
          field_description: "Store model in DB",
          field_value: true,
          stored_in_db: true,
          field_default_value: false,
        },
      ],
      isLoading: false,
      refetch: mockRefetch,
    } as any);

    renderWithProviders(<ModelSettingsModal {...defaultProps} />);

    const switchElement = screen.getByRole("switch");
    expect(switchElement).toBeChecked();
  });

  it("should show skeleton loader when config is loading", () => {
    mockUseProxyConfig.mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: mockRefetch,
    } as any);

    renderWithProviders(<ModelSettingsModal {...defaultProps} />);

    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
    const skeletons = document.querySelectorAll(".ant-skeleton");
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it("should not call onSuccess when it is not provided", async () => {
    const user = userEvent.setup();
    mockMutateAsync.mockImplementation(async (params, options) => {
      await Promise.resolve();
      options?.onSuccess?.();
      return { message: "Success" };
    });

    renderWithProviders(<ModelSettingsModal isVisible={true} onCancel={mockOnCancel} />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockNotificationsManager.success).toHaveBeenCalled();
    });
  });
});
