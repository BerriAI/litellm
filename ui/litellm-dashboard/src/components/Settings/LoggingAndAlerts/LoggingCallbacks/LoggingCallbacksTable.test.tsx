import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../../tests/test-utils";
import { LoggingCallbacksTable } from "./LoggingCallbacksTable";
import type { AlertingObject } from "./types";

const baseVars = {
  SLACK_WEBHOOK_URL: null,
  LANGFUSE_PUBLIC_KEY: null,
  LANGFUSE_SECRET_KEY: null,
  LANGFUSE_HOST: null,
  OPENMETER_API_KEY: null,
};

describe("LoggingCallbacksTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    renderWithProviders(<LoggingCallbacksTable callbacks={[]} availableCallbacks={{}} />);
    expect(screen.getByText("Active Logging Callbacks")).toBeInTheDocument();
  });

  it("should show the empty state when there are no callbacks", () => {
    renderWithProviders(<LoggingCallbacksTable callbacks={[]} availableCallbacks={{}} />);
    expect(screen.getByText("No callbacks configured")).toBeInTheDocument();
    expect(screen.getByText("Add your first callback to start logging data to external services.")).toBeInTheDocument();
  });

  it("should show skeleton rows while loading", () => {
    renderWithProviders(<LoggingCallbacksTable callbacks={[]} availableCallbacks={{}} isLoading />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("No callbacks configured")).not.toBeInTheDocument();
  });

  it('should map "otel" to "OpenTelemetry" on the table', () => {
    renderWithProviders(
      <LoggingCallbacksTable
        callbacks={[{ name: "otel", variables: baseVars }]}
        availableCallbacks={{
          otel: {
            litellm_callback_name: "otel",
            litellm_callback_params: [],
            ui_callback_name: "OpenTelemetry",
          },
        }}
      />,
    );
    expect(screen.getByText("OpenTelemetry")).toBeInTheDocument();
  });

  it("should fallback to original callback name when not in availableCallbacks", () => {
    renderWithProviders(
      <LoggingCallbacksTable
        callbacks={[{ name: "custom_callback_x", variables: baseVars }]}
        availableCallbacks={{}}
      />,
    );
    expect(screen.getByText("custom_callback_x")).toBeInTheDocument();
  });

  it("should call onAdd when the Add Callback button is clicked", async () => {
    const user = userEvent.setup();
    const onAdd = vi.fn();
    renderWithProviders(<LoggingCallbacksTable callbacks={[]} availableCallbacks={{}} onAdd={onAdd} />);
    await user.click(screen.getByRole("button", { name: /add callback/i }));
    expect(onAdd).toHaveBeenCalled();
  });

  it("should test, edit, and delete a callback through the actions menu", async () => {
    const user = userEvent.setup();
    const onTest = vi.fn();
    const onEdit = vi.fn();
    const onDelete = vi.fn();
    const callback = { name: "langfuse", type: "success" as const, variables: baseVars };
    renderWithProviders(
      <LoggingCallbacksTable
        callbacks={[callback]}
        availableCallbacks={{}}
        onTest={onTest}
        onEdit={onEdit}
        onDelete={onDelete}
      />,
    );

    await user.click(screen.getByTestId("callback-actions-langfuse-success"));
    await user.click(await screen.findByTestId("callback-action-test"));
    expect(onTest).toHaveBeenCalledWith(callback);

    await user.click(screen.getByTestId("callback-actions-langfuse-success"));
    await user.click(await screen.findByTestId("callback-action-edit"));
    expect(onEdit).toHaveBeenCalledWith(callback);

    await user.click(screen.getByTestId("callback-actions-langfuse-success"));
    await user.click(await screen.findByTestId("callback-action-delete"));
    expect(onDelete).toHaveBeenCalledWith(callback);
  });

  it("shows a read-only label instead of the actions menu for runtime-only callback rows", () => {
    renderWithProviders(
      <LoggingCallbacksTable
        callbacks={[
          { name: "langfuse", type: "success" as const, variables: baseVars },
          { name: "datadog", type: "success" as const, variables: baseVars, read_only: true },
        ]}
        availableCallbacks={{}}
        onTest={vi.fn()}
        onEdit={vi.fn()}
        onDelete={vi.fn()}
      />,
    );
    expect(screen.getByTestId("callback-actions-langfuse-success")).toBeInTheDocument();
    expect(screen.queryByTestId("callback-actions-datadog-success")).not.toBeInTheDocument();
    expect(screen.getAllByText("Read only")).toHaveLength(1);
  });

  // Regression: `/get_callbacks` returns the same `name` twice when a
  // callback is registered for both success and failure (e.g. `generic_api`
  // → POST to spend-log on both 200 and 4xx/5xx). The UI used to ignore
  // the `type` field and render every row as "Success", masking the
  // failure registration. Reading `record.type` fixes the badge AND
  // composing the row id with type avoids React's duplicate-key warning.
  it("renders distinct Success and Failure badges for same-name dual registration", () => {
    renderWithProviders(
      <LoggingCallbacksTable
        callbacks={[
          { name: "generic_api", type: "success", variables: baseVars },
          { name: "generic_api", type: "failure", variables: baseVars },
        ]}
        availableCallbacks={{
          generic_api: {
            litellm_callback_name: "generic_api",
            litellm_callback_params: [],
            ui_callback_name: "Custom Callback API",
          },
        }}
      />,
    );
    expect(screen.getAllByText("Custom Callback API")).toHaveLength(2);
    expect(screen.getByText("Success")).toBeInTheDocument();
    expect(screen.getByText("Failure")).toBeInTheDocument();
  });

  describe("URL state", () => {
    const availableCallbacks = {
      zeta_raw: { litellm_callback_name: "zeta_raw", litellm_callback_params: [], ui_callback_name: "Alpha Logger" },
      alpha_raw: { litellm_callback_name: "alpha_raw", litellm_callback_params: [], ui_callback_name: "Zulu Logger" },
    };
    const mixedCallbacks: AlertingObject[] = [
      { name: "alpha_raw", type: "success", variables: baseVars },
      { name: "zeta_raw", type: "failure", variables: baseVars },
      { name: "middle_callback", type: "success_and_failure", variables: baseVars },
    ];
    const manyCallbacks: AlertingObject[] = Array.from({ length: 30 }, (_, index) => ({
      name: `callback_${String(index + 1).padStart(2, "0")}`,
      type: "success",
      variables: baseVars,
    }));

    const rowIds = () =>
      screen
        .getAllByRole("row")
        .map((row) => row.getAttribute("data-row-id"))
        .filter((id): id is string => id !== null);

    const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
      onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

    it("sorts by the shown callback name by default", () => {
      renderWithProviders(<LoggingCallbacksTable callbacks={mixedCallbacks} availableCallbacks={availableCallbacks} />);
      expect(rowIds()).toEqual(["zeta_raw-failure", "middle_callback-success_and_failure", "alpha_raw-success"]);
    });

    it("sorts by the column and direction in the URL", () => {
      renderWithProviders(
        <LoggingCallbacksTable callbacks={mixedCallbacks} availableCallbacks={availableCallbacks} />,
        {
          searchParams: "?callbacks_sort_by=mode&callbacks_sort_order=desc",
        },
      );
      expect(rowIds()).toEqual(["middle_callback-success_and_failure", "alpha_raw-success", "zeta_raw-failure"]);
    });

    it("writes header sort clicks to the prefixed sort keys", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(
        <LoggingCallbacksTable callbacks={mixedCallbacks} availableCallbacks={availableCallbacks} />,
        {
          onUrlUpdate,
        },
      );

      await user.click(screen.getByTestId("sort-header-mode"));
      expect(lastUrl(onUrlUpdate)?.get("callbacks_sort_by")).toBe("mode");
      expect(lastUrl(onUrlUpdate)?.has("sort_by")).toBe(false);
      expect(rowIds()).toEqual(["zeta_raw-failure", "alpha_raw-success", "middle_callback-success_and_failure"]);
    });

    it("filters by the search in the URL across name, shown name and mode", () => {
      const { unmount } = renderWithProviders(
        <LoggingCallbacksTable callbacks={mixedCallbacks} availableCallbacks={availableCallbacks} />,
        { searchParams: "?callbacks_search=zulu" },
      );
      expect(screen.getByTestId("datatable-search")).toHaveValue("zulu");
      expect(rowIds()).toEqual(["alpha_raw-success"]);
      unmount();

      renderWithProviders(
        <LoggingCallbacksTable callbacks={mixedCallbacks} availableCallbacks={availableCallbacks} />,
        {
          searchParams: "?callbacks_search=FAILURE",
        },
      );
      expect(rowIds()).toEqual(["zeta_raw-failure", "middle_callback-success_and_failure"]);
    });

    it("shows a no-matches state instead of the setup prompt when the search finds nothing", () => {
      renderWithProviders(
        <LoggingCallbacksTable callbacks={mixedCallbacks} availableCallbacks={availableCallbacks} />,
        {
          searchParams: "?callbacks_search=nothing-here",
        },
      );
      expect(screen.getByText("No matching callbacks")).toBeInTheDocument();
      expect(screen.queryByText("No callbacks configured")).not.toBeInTheDocument();
    });

    it("writes the search to callbacks_search and returns to the first page", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<LoggingCallbacksTable callbacks={manyCallbacks} availableCallbacks={{}} />, {
        searchParams: "?callbacks_page=2",
        onUrlUpdate,
      });
      expect(rowIds()[0]).toBe("callback_26-success");

      fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "callback_0" } });

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.get("callbacks_search")).toBe("callback_0"));
      expect(lastUrl(onUrlUpdate)?.has("callbacks_page")).toBe(false);
      expect(rowIds()).toHaveLength(9);
      expect(rowIds()[0]).toBe("callback_01-success");
    });

    it("writes page changes to callbacks_page", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<LoggingCallbacksTable callbacks={manyCallbacks} availableCallbacks={{}} />, {
        onUrlUpdate,
      });
      expect(rowIds()).toHaveLength(25);

      await user.click(screen.getByTestId("pagination-next"));

      expect(lastUrl(onUrlUpdate)?.get("callbacks_page")).toBe("2");
      expect(lastUrl(onUrlUpdate)?.has("page")).toBe(false);
      expect(rowIds()).toEqual([
        "callback_26-success",
        "callback_27-success",
        "callback_28-success",
        "callback_29-success",
        "callback_30-success",
      ]);
    });
  });
});
