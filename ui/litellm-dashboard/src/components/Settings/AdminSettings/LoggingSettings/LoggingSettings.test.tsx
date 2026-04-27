import { useDeleteProxyConfigField, useProxyConfig } from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import { useStoreRequestInSpendLogs } from "@/app/(dashboard)/hooks/storeRequestInSpendLogs/useStoreRequestInSpendLogs";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../../tests/test-utils";
import LoggingSettings from "./LoggingSettings";

vi.mock("@/app/(dashboard)/hooks/storeRequestInSpendLogs/useStoreRequestInSpendLogs");
vi.mock("@/app/(dashboard)/hooks/proxyConfig/useProxyConfig", async () => {
  const actual = await vi.importActual<typeof import("@/app/(dashboard)/hooks/proxyConfig/useProxyConfig")>(
    "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig",
  );
  return {
    ...actual,
    useProxyConfig: vi.fn(),
    useDeleteProxyConfigField: vi.fn(),
  };
});
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

describe("LoggingSettings", () => {
  const mockMutate = vi.fn();
  const mockDeleteField = vi.fn();
  const mockRefetch = vi.fn();

  beforeEach(() => {
    vi.resetAllMocks();
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
    mockParseErrorMessage.mockImplementation((error: any) => error?.message || String(error));
  });

  it("should render the card with title and form fields", () => {
    renderWithProviders(<LoggingSettings />);

    expect(screen.getByText("Logging Settings")).toBeInTheDocument();
    expect(screen.getByText("Store Prompts in Spend Logs")).toBeInTheDocument();
    expect(screen.getByLabelText("Maximum Spend Logs Retention Period (Optional)")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g., 7d, 30d")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Save Settings" })).toBeInTheDocument();
  });

  it("should toggle store prompts switch", async () => {
    const user = userEvent.setup();
    renderWithProviders(<LoggingSettings />);

    const switchElement = screen.getByRole("switch");
    expect(switchElement).not.toBeChecked();

    await user.click(switchElement);

    await waitFor(() => {
      expect(switchElement).toBeChecked();
    });
  });

  it("should update retention period input", async () => {
    const user = userEvent.setup();
    renderWithProviders(<LoggingSettings />);

    const retentionInput = screen.getByPlaceholderText("e.g., 7d, 30d");
    await user.type(retentionInput, "30d");

    expect(retentionInput).toHaveValue("30d");
  });

  it("should submit form with store prompts enabled and retention period", async () => {
    const user = userEvent.setup();
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<LoggingSettings />);

    const switchElement = screen.getByRole("switch");
    await user.click(switchElement);

    const retentionInput = screen.getByPlaceholderText("e.g., 7d, 30d");
    await user.type(retentionInput, "30d");

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockDeleteField).not.toHaveBeenCalled();
      expect(mockMutate).toHaveBeenCalledWith(
        {
          store_prompts_in_spend_logs: true,
          maximum_spend_logs_retention_period: "30d",
        },
        expect.any(Object),
      );
    });
  });

  it("should delete retention period field when left empty on submit", async () => {
    const user = userEvent.setup();
    mockDeleteField.mockImplementation((_params, options) => {
      options?.onSettled?.();
    });
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<LoggingSettings />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockDeleteField).toHaveBeenCalled();
      expect(mockMutate).toHaveBeenCalledWith(
        {
          store_prompts_in_spend_logs: false,
        },
        expect.any(Object),
      );
    });
  });

  it("should show success notification on successful submission", async () => {
    const user = userEvent.setup();
    mockDeleteField.mockImplementation((_params, options) => {
      options?.onSettled?.();
    });
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<LoggingSettings />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockNotificationsManager.success).toHaveBeenCalledWith("Spend logs settings updated successfully");
    });
  });

  it("should show a single error notification via onError callback", async () => {
    const user = userEvent.setup();
    const error = new Error("Backend error");
    mockDeleteField.mockImplementation((_params, options) => {
      options?.onSettled?.();
    });
    mockMutate.mockImplementation((_params, options) => {
      options?.onError?.(error);
    });
    mockParseErrorMessage.mockReturnValue("Backend error");

    renderWithProviders(<LoggingSettings />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockNotificationsManager.fromBackend).toHaveBeenCalledWith(
        "Failed to save spend logs settings: Backend error",
      );
    });
    expect(mockNotificationsManager.fromBackend).toHaveBeenCalledTimes(1);
  });

  it("should show loading state on save button when update pending", () => {
    mockUseStoreRequestInSpendLogs.mockReturnValue({
      mutate: mockMutate,
      isPending: true,
    } as any);

    renderWithProviders(<LoggingSettings />);

    const saveButton = screen.getByRole("button", { name: /Saving/i });
    expect(saveButton).toBeInTheDocument();
    expect(saveButton.className).toContain("ant-btn-loading");
  });

  it("should show loading state on save button when delete pending", () => {
    mockUseDeleteProxyConfigField.mockReturnValue({
      mutate: mockDeleteField,
      isPending: true,
    } as any);

    renderWithProviders(<LoggingSettings />);

    const saveButton = screen.getByRole("button", { name: /Saving/i });
    expect(saveButton).toBeInTheDocument();
    expect(saveButton.className).toContain("ant-btn-loading");
  });

  it("should disable save button while config is loading", () => {
    mockUseProxyConfig.mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: mockRefetch,
    } as any);

    renderWithProviders(<LoggingSettings />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    expect(saveButton).toBeDisabled();
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

    renderWithProviders(<LoggingSettings />);

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

    renderWithProviders(<LoggingSettings />);

    expect(screen.queryByRole("switch")).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("e.g., 7d, 30d")).not.toBeInTheDocument();

    const skeletons = document.querySelectorAll(".ant-skeleton");
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it("should continue with update even if deleteField fails", async () => {
    const user = userEvent.setup();
    const deleteError = new Error("Field does not exist");
    mockDeleteField.mockImplementation((_params, options) => {
      options?.onError?.(deleteError);
      options?.onSettled?.();
    });
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<LoggingSettings />);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockDeleteField).toHaveBeenCalled();
      expect(mockMutate).toHaveBeenCalledWith(
        {
          store_prompts_in_spend_logs: false,
        },
        expect.any(Object),
      );
      expect(mockNotificationsManager.success).toHaveBeenCalled();
    });
  });

  it("should submit with only store prompts enabled when retention is empty", async () => {
    const user = userEvent.setup();
    mockDeleteField.mockImplementation((_params, options) => {
      options?.onSettled?.();
    });
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<LoggingSettings />);

    const switchElement = screen.getByRole("switch");
    await user.click(switchElement);

    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockDeleteField).toHaveBeenCalled();
      expect(mockMutate).toHaveBeenCalledWith(
        {
          store_prompts_in_spend_logs: true,
        },
        expect.any(Object),
      );
    });
  });
});
