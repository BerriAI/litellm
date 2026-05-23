import { useDeleteProxyConfigField, useProxyConfig } from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import { useStoreRequestInSpendLogs } from "@/app/(dashboard)/hooks/storeRequestInSpendLogs/useStoreRequestInSpendLogs";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { screen, waitFor } from "@testing-library/react";
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

describe("LoggingSettings (async data arrival)", () => {
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
    mockParseErrorMessage.mockImplementation((error: any) => error?.message || String(error));
  });

  it("should reflect store_prompts_in_spend_logs=true on the Switch when data is initially loading and then arrives", async () => {
    // First render: still loading (no data yet). The form mounts with
    // initialValues = { store_prompts_in_spend_logs: false }.
    let returnValue: any = {
      data: undefined,
      isLoading: true,
      refetch: mockRefetch,
    };
    mockUseProxyConfig.mockImplementation(() => returnValue);

    const { rerender } = renderWithProviders(<LoggingSettings />);

    // Now the data arrives with field_value: true. Re-render to flush.
    returnValue = {
      data: [
        {
          field_name: "store_prompts_in_spend_logs",
          field_type: "Boolean",
          field_description: "Store prompts in spend logs",
          field_value: true,
          stored_in_db: true,
          field_default_value: null,
        },
        {
          field_name: "maximum_spend_logs_retention_period",
          field_type: "String",
          field_description: "Retention",
          field_value: null,
          stored_in_db: null,
          field_default_value: null,
        },
      ],
      isLoading: false,
      refetch: mockRefetch,
    };
    rerender(<LoggingSettings />);

    await waitFor(() => {
      const switchElement = screen.getByRole("switch");
      expect(switchElement).toBeChecked();
    });
  });
});
