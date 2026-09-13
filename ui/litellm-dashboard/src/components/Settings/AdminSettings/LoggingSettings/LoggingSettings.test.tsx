import {
  DeleteProxyConfigFieldRequest,
  useDeleteProxyConfigField,
  useProxyConfig,
} from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import { useStoreRequestInSpendLogs } from "@/app/(dashboard)/hooks/storeRequestInSpendLogs/useStoreRequestInSpendLogs";
import { toast } from "@/lib/toast";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { screen, waitFor, within } from "@testing-library/react";
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
vi.mock("@/components/shared/errorUtils", () => ({
  parseErrorMessage: vi.fn(),
}));

const mockUseStoreRequestInSpendLogs = vi.mocked(useStoreRequestInSpendLogs);
const mockUseProxyConfig = vi.mocked(useProxyConfig);
const mockUseDeleteProxyConfigField = vi.mocked(useDeleteProxyConfigField);
const mockToast = vi.mocked(toast);
const mockParseErrorMessage = vi.mocked(parseErrorMessage);

describe("LoggingSettings", () => {
  const mockMutate = vi.fn();
  const mockDeleteField = vi.fn();
  const mockRefetch = vi.fn();

  const clearedFieldNames = (): string[] =>
    mockDeleteField.mock.calls.map((call) => (call[0] as DeleteProxyConfigFieldRequest).field_name);

  // Every optional knob already persisted. Clearing is only ever issued for a
  // field that has a stored value, so any test about the clear path has to say
  // so; the default mock below is an empty config, which is a proxy that has
  // never saved these settings and therefore has nothing to clear.
  const withEveryOptionalFieldStored = () =>
    mockUseProxyConfig.mockReturnValue({
      data: [
        {
          field_name: "maximum_spend_logs_retention_period",
          field_type: "string",
          field_description: "Maximum retention period",
          field_value: "30d",
          stored_in_db: true,
        },
        {
          field_name: "maximum_spend_logs_cleanup_batch_size",
          field_type: "Integer",
          field_description: "Rows per delete",
          field_value: 2000,
          stored_in_db: true,
        },
        {
          field_name: "maximum_spend_logs_cleanup_max_batches",
          field_type: "Integer",
          field_description: "Deletes per table per run",
          field_value: 50,
          stored_in_db: true,
        },
        {
          field_name: "maximum_spend_logs_cleanup_run_budget",
          field_type: "string",
          field_description: "Wall clock budget per run",
          field_value: "90s",
          stored_in_db: true,
        },
        {
          field_name: "maximum_spend_logs_cleanup_batch_timeout",
          field_type: "string",
          field_description: "Statement and lock timeout per batch",
          field_value: "10s",
          stored_in_db: true,
        },
      ],
      isLoading: false,
      refetch: mockRefetch,
    } as unknown as ReturnType<typeof useProxyConfig>);

  // Blank every optional input the form rendered from stored values, which is
  // what an admin does to reset a knob to its default.
  const blankEveryOptionalField = async (user: ReturnType<typeof userEvent.setup>) => {
    for (const placeholder of ["e.g., 7d, 30d", "e.g., 1000", "e.g., 500", "e.g., 5m", "e.g., 30s"]) {
      await user.clear(screen.getByPlaceholderText(placeholder));
    }
  };

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

  it("should render a control for every spend logs cleanup knob", () => {
    renderWithProviders(<LoggingSettings />);

    expect(screen.getByLabelText("Spend Logs Cleanup Batch Size (Optional)")).toBeInTheDocument();
    expect(screen.getByLabelText("Spend Logs Cleanup Max Batches (Optional)")).toBeInTheDocument();
    expect(screen.getByLabelText("Spend Logs Cleanup Run Budget (Optional)")).toBeInTheDocument();
    expect(screen.getByLabelText("Spend Logs Cleanup Batch Timeout (Optional)")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g., 1000")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g., 500")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g., 5m")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("e.g., 30s")).toBeInTheDocument();
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
    mockDeleteField.mockImplementation((_params, options) => {
      options?.onSettled?.();
    });
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
      expect(mockMutate).toHaveBeenCalledWith(
        {
          store_prompts_in_spend_logs: true,
          maximum_spend_logs_retention_period: "30d",
        },
        expect.any(Object),
      );
    });
    expect(clearedFieldNames()).not.toContain("maximum_spend_logs_retention_period");
  });

  it("should submit every spend logs cleanup setting that has a value", async () => {
    const user = userEvent.setup();
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<LoggingSettings />);

    await user.click(screen.getByRole("switch"));
    await user.type(screen.getByPlaceholderText("e.g., 7d, 30d"), "30d");
    await user.type(screen.getByPlaceholderText("e.g., 1000"), "2000");
    await user.type(screen.getByPlaceholderText("e.g., 500"), "50");
    await user.type(screen.getByPlaceholderText("e.g., 5m"), "90s");
    await user.type(screen.getByPlaceholderText("e.g., 30s"), "10s");

    await user.click(screen.getByRole("button", { name: "Save Settings" }));

    const expectedParams = {
      store_prompts_in_spend_logs: true,
      maximum_spend_logs_retention_period: "30d",
      maximum_spend_logs_cleanup_batch_size: 2000,
      maximum_spend_logs_cleanup_max_batches: 50,
      maximum_spend_logs_cleanup_run_budget: "90s",
      maximum_spend_logs_cleanup_batch_timeout: "10s",
    };

    await waitFor(() => {
      expect(mockMutate).toHaveBeenCalledWith(expectedParams, expect.any(Object));
    });
    expect(mockDeleteField).not.toHaveBeenCalled();
  });

  it("should omit blank cleanup settings from the save payload instead of sending empty values", async () => {
    const user = userEvent.setup();
    mockDeleteField.mockImplementation((_params, options) => {
      options?.onSettled?.();
    });
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<LoggingSettings />);

    await user.type(screen.getByPlaceholderText("e.g., 1000"), "2000");
    await user.click(screen.getByRole("button", { name: "Save Settings" }));

    await waitFor(() => {
      expect(mockMutate).toHaveBeenCalled();
    });

    const submittedParams = mockMutate.mock.calls[0][0];
    expect(submittedParams).not.toHaveProperty("maximum_spend_logs_retention_period");
    expect(submittedParams).not.toHaveProperty("maximum_spend_logs_cleanup_max_batches");
    expect(submittedParams).not.toHaveProperty("maximum_spend_logs_cleanup_run_budget");
    expect(submittedParams).not.toHaveProperty("maximum_spend_logs_cleanup_batch_timeout");
    expect(submittedParams).toEqual({
      store_prompts_in_spend_logs: false,
      maximum_spend_logs_cleanup_batch_size: 2000,
    });
  });

  it("should clear the stored value of every cleanup setting left blank", async () => {
    const user = userEvent.setup();
    withEveryOptionalFieldStored();
    mockDeleteField.mockImplementation((_params, options) => {
      options?.onSettled?.();
    });
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<LoggingSettings />);

    await blankEveryOptionalField(user);
    await user.type(screen.getByPlaceholderText("e.g., 5m"), "10m");
    await user.click(screen.getByRole("button", { name: "Save Settings" }));

    await waitFor(() => {
      expect(mockMutate).toHaveBeenCalled();
    });

    expect(clearedFieldNames().sort()).toEqual([
      "maximum_spend_logs_cleanup_batch_size",
      "maximum_spend_logs_cleanup_batch_timeout",
      "maximum_spend_logs_cleanup_max_batches",
      "maximum_spend_logs_retention_period",
    ]);
  });

  it("should delete retention period field when left empty on submit", async () => {
    const user = userEvent.setup();
    withEveryOptionalFieldStored();
    mockDeleteField.mockImplementation((_params, options) => {
      options?.onSettled?.();
    });
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<LoggingSettings />);

    await blankEveryOptionalField(user);
    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(clearedFieldNames()).toContain("maximum_spend_logs_retention_period");
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
      expect(mockToast.success).toHaveBeenCalledWith("Spend logs settings updated successfully");
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
      expect(mockToast.fromError).toHaveBeenCalledWith("Failed to save spend logs settings: Backend error");
    });
    expect(mockToast.fromError).toHaveBeenCalledTimes(1);
  });

  it("should show loading state on save button when update pending", () => {
    mockUseStoreRequestInSpendLogs.mockReturnValue({
      mutate: mockMutate,
      isPending: true,
    } as any);

    renderWithProviders(<LoggingSettings />);

    const saveButton = screen.getByRole("button", { name: /Saving/i });
    expect(saveButton).toBeInTheDocument();
    expect(within(saveButton).getByRole("img", { name: "loading" })).toBeInTheDocument();
  });

  it("should show loading state on save button when delete pending", () => {
    mockUseDeleteProxyConfigField.mockReturnValue({
      mutate: mockDeleteField,
      isPending: true,
    } as any);

    renderWithProviders(<LoggingSettings />);

    const saveButton = screen.getByRole("button", { name: /Saving/i });
    expect(saveButton).toBeInTheDocument();
    expect(within(saveButton).getByRole("img", { name: "loading" })).toBeInTheDocument();
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
        {
          field_name: "maximum_spend_logs_cleanup_batch_size",
          field_type: "Integer",
          field_description: "Rows per delete",
          field_value: 2000,
          stored_in_db: true,
          field_default_value: 1000,
        },
        {
          field_name: "maximum_spend_logs_cleanup_max_batches",
          field_type: "Integer",
          field_description: "Deletes per table per run",
          field_value: 50,
          stored_in_db: true,
          field_default_value: 500,
        },
        {
          field_name: "maximum_spend_logs_cleanup_run_budget",
          field_type: "string",
          field_description: "Wall clock budget per run",
          field_value: "90s",
          stored_in_db: true,
          field_default_value: "5m",
        },
        {
          field_name: "maximum_spend_logs_cleanup_batch_timeout",
          field_type: "string",
          field_description: "Statement and lock timeout per batch",
          field_value: "10s",
          stored_in_db: true,
          field_default_value: "30s",
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
    expect(screen.getByPlaceholderText("e.g., 1000")).toHaveDisplayValue("2000");
    expect(screen.getByPlaceholderText("e.g., 500")).toHaveDisplayValue("50");
    expect(screen.getByPlaceholderText("e.g., 5m")).toHaveValue("90s");
    expect(screen.getByPlaceholderText("e.g., 30s")).toHaveValue("10s");
  });

  it("should reflect persisted values that arrive after the initial loading render", async () => {
    mockUseProxyConfig.mockReturnValue({
      data: undefined,
      isLoading: true,
      refetch: mockRefetch,
    } as unknown as ReturnType<typeof useProxyConfig>);

    const { rerender } = renderWithProviders(<LoggingSettings />);

    expect(screen.queryByRole("switch")).not.toBeInTheDocument();

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
    } as unknown as ReturnType<typeof useProxyConfig>);

    rerender(<LoggingSettings />);

    await waitFor(() => {
      expect(screen.getByRole("switch")).toBeChecked();
    });
    expect(screen.getByPlaceholderText("e.g., 7d, 30d")).toHaveValue("30d");
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
    expect(screen.queryByRole("button", { name: "Save Settings" })).not.toBeInTheDocument();

    expect(document.querySelectorAll('[data-slot="skeleton"]').length).toBeGreaterThan(0);
  });

  it("should report an error and not claim success when clearing a field fails", async () => {
    const user = userEvent.setup();
    withEveryOptionalFieldStored();
    mockDeleteField.mockImplementation((_params, options) => {
      options?.onError?.(new Error("Field does not exist"));
      options?.onSettled?.();
    });
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<LoggingSettings />);

    await blankEveryOptionalField(user);
    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockToast.fromError).toHaveBeenCalled();
    });
    // the old value is still in force server side, so an unqualified success
    // notification would tell the admin the opposite of what happened
    expect(mockToast.success).not.toHaveBeenCalled();
  });

  it("should clear fields one at a time, never concurrently", async () => {
    const user = userEvent.setup();
    withEveryOptionalFieldStored();
    let inFlight = 0;
    let maxInFlight = 0;

    mockDeleteField.mockImplementation((_params, options) => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      // Settle on a microtask rather than synchronously, so a parallel
      // implementation genuinely overlaps: Promise.all would issue every call
      // before any of them settles, driving inFlight to the number of fields.
      void Promise.resolve().then(() => {
        inFlight -= 1;
        options?.onSettled?.();
      });
    });
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<LoggingSettings />);

    await blankEveryOptionalField(user);
    const saveButton = screen.getByRole("button", { name: "Save Settings" });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mockToast.success).toHaveBeenCalled();
    });
    // /config/field/delete rewrites the whole general_settings object, so two of
    // them in flight at once means the later write restores what the earlier cleared
    expect(maxInFlight).toBe(1);
    expect(mockDeleteField.mock.calls.length).toBeGreaterThan(1);
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
      expect(mockMutate).toHaveBeenCalledWith(
        {
          store_prompts_in_spend_logs: true,
        },
        expect.any(Object),
      );
    });
    // nothing is stored for the blank fields, so there is nothing to clear
    expect(mockDeleteField).not.toHaveBeenCalled();
  });

  it("should save on a proxy that has never stored these settings, without clearing anything", async () => {
    // The first save on a new deployment: no general_settings row exists, so
    // /config/field/delete answers 400 for every blank field. Issuing those
    // clears anyway failed the whole save and persisted nothing.
    const user = userEvent.setup();
    mockDeleteField.mockImplementation((_params, options) => {
      options?.onError?.(new Error("Field name=... not in config"));
      options?.onSettled?.();
    });
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<LoggingSettings />);

    const switchElement = screen.getByRole("switch");
    await user.click(switchElement);
    await user.click(screen.getByRole("button", { name: "Save Settings" }));

    await waitFor(() => {
      expect(mockToast.success).toHaveBeenCalled();
    });
    expect(mockDeleteField).not.toHaveBeenCalled();
    expect(mockMutate).toHaveBeenCalledWith({ store_prompts_in_spend_logs: true }, expect.any(Object));
    expect(mockToast.fromError).not.toHaveBeenCalled();
  });

  it("should still clear a field that does have a stored value", async () => {
    // The guard above must not turn into "never clear anything": a field the
    // admin blanks out that IS stored still has to be deleted server side.
    const user = userEvent.setup();
    withEveryOptionalFieldStored();
    mockDeleteField.mockImplementation((_params, options) => {
      options?.onSettled?.();
    });
    mockMutate.mockImplementation((_params, options) => {
      options?.onSuccess?.();
    });

    renderWithProviders(<LoggingSettings />);

    await user.clear(screen.getByPlaceholderText("e.g., 5m"));
    await user.click(screen.getByRole("button", { name: "Save Settings" }));

    await waitFor(() => {
      expect(mockToast.success).toHaveBeenCalled();
    });
    expect(clearedFieldNames()).toContain("maximum_spend_logs_cleanup_run_budget");
  });
  describe("numeric coercion parity", () => {
    it("should round a fractional batch size to a whole number, in the input and in the payload", async () => {
      const user = userEvent.setup();
      mockMutate.mockImplementation((_params, options) => {
        options?.onSuccess?.();
      });

      renderWithProviders(<LoggingSettings />);

      const batchSize = screen.getByPlaceholderText("e.g., 1000");
      await user.type(batchSize, "2000.7");
      await user.click(screen.getByRole("button", { name: "Save Settings" }));

      await waitFor(() => {
        expect(mockMutate).toHaveBeenCalled();
      });
      expect(batchSize).toHaveDisplayValue("2001");
      expect(mockMutate.mock.calls[0][0]).toEqual({
        store_prompts_in_spend_logs: false,
        maximum_spend_logs_cleanup_batch_size: 2001,
      });
      expect(typeof mockMutate.mock.calls[0][0].maximum_spend_logs_cleanup_batch_size).toBe("number");
    });

    it("should raise a below-minimum batch size to one rather than sending it", async () => {
      const user = userEvent.setup();
      mockMutate.mockImplementation((_params, options) => {
        options?.onSuccess?.();
      });

      renderWithProviders(<LoggingSettings />);

      const batchSize = screen.getByPlaceholderText("e.g., 1000");
      await user.type(batchSize, "0");
      await user.click(screen.getByRole("button", { name: "Save Settings" }));

      await waitFor(() => {
        expect(mockMutate).toHaveBeenCalled();
      });
      expect(batchSize).toHaveDisplayValue("1");
      expect(mockMutate.mock.calls[0][0].maximum_spend_logs_cleanup_batch_size).toBe(1);
    });

    it("should send a trimmable duration exactly as typed, without coercion", async () => {
      const user = userEvent.setup();
      mockMutate.mockImplementation((_params, options) => {
        options?.onSuccess?.();
      });

      renderWithProviders(<LoggingSettings />);

      await user.type(screen.getByPlaceholderText("e.g., 7d, 30d"), "30d");
      await user.click(screen.getByRole("button", { name: "Save Settings" }));

      await waitFor(() => {
        expect(mockMutate).toHaveBeenCalled();
      });
      expect(mockMutate.mock.calls[0][0]).toEqual({
        store_prompts_in_spend_logs: false,
        maximum_spend_logs_retention_period: "30d",
      });
    });

    it("should keep a whitespace-only duration out of the payload", async () => {
      const user = userEvent.setup();
      mockMutate.mockImplementation((_params, options) => {
        options?.onSuccess?.();
      });

      renderWithProviders(<LoggingSettings />);

      await user.type(screen.getByPlaceholderText("e.g., 7d, 30d"), "   ");
      await user.click(screen.getByRole("button", { name: "Save Settings" }));

      await waitFor(() => {
        expect(mockMutate).toHaveBeenCalled();
      });
      expect(mockMutate.mock.calls[0][0]).toEqual({ store_prompts_in_spend_logs: false });
    });
  });
});
