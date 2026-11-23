import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { LoggingCallbacksTable } from "./LoggingCallbacksTable";

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

  it("should render the callbacks table", () => {
    render(<LoggingCallbacksTable callbacks={[]} availableCallbacks={{}} />);
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("should show the empty state when there are no callbacks", () => {
    render(<LoggingCallbacksTable callbacks={[]} availableCallbacks={{}} />);
    expect(screen.getByText("No callbacks configured")).toBeInTheDocument();
    expect(screen.getByText("Add your first callback to start logging data to external services.")).toBeInTheDocument();
  });

  it("should show skeleton rows while loading", () => {
    render(<LoggingCallbacksTable callbacks={[]} availableCallbacks={{}} isLoading />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("No callbacks configured")).not.toBeInTheDocument();
  });

  it('should map "otel" to "OpenTelemetry" on the table', () => {
    render(
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
    render(
      <LoggingCallbacksTable
        callbacks={[{ name: "custom_callback_x", variables: baseVars }]}
        availableCallbacks={{}}
      />,
    );
    expect(screen.getByText("custom_callback_x")).toBeInTheDocument();
  });

  it("should test, edit, and delete a callback through the actions menu", async () => {
    const user = userEvent.setup();
    const onTest = vi.fn();
    const onEdit = vi.fn();
    const onDelete = vi.fn();
    const callback = { name: "langfuse", type: "success" as const, variables: baseVars };
    render(
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
    render(
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
    expect(screen.getAllByText("DB")).toHaveLength(1);
  });

  it("shows which rows config.yaml sets and blocks deleting them there", async () => {
    const user = userEvent.setup();
    const onTest = vi.fn();
    const onDelete = vi.fn();
    const fromConfig = { name: "langfuse", type: "success" as const, variables: baseVars, source: "config" as const };
    const fromDatabase = { name: "langfuse", type: "failure" as const, variables: baseVars };
    render(
      <LoggingCallbacksTable
        callbacks={[fromConfig, fromDatabase]}
        availableCallbacks={{}}
        onTest={onTest}
        onEdit={vi.fn()}
        onDelete={onDelete}
      />,
    );

    const sourceColumn = screen.getAllByRole("columnheader").findIndex((header) => header.textContent === "Source");
    const sourceCell = (mode: RegExp) =>
      within(screen.getByRole("row", { name: mode })).getAllByRole("cell")[sourceColumn];
    expect(sourceColumn).toBeGreaterThan(-1);
    expect(sourceCell(/Success/)).toHaveTextContent(/^Config$/);
    expect(sourceCell(/Failure/)).toHaveTextContent(/^DB$/);

    await user.click(screen.getByTestId("callback-actions-langfuse-success"));
    expect(await screen.findByTestId("callback-config-owned-hint")).toBeInTheDocument();
    const remove = screen.getByTestId("callback-action-delete");
    expect(remove).toHaveAttribute("data-disabled");
    await user.click(remove);
    expect(onDelete).not.toHaveBeenCalled();
    await user.click(screen.getByTestId("callback-action-test"));
    expect(onTest).toHaveBeenCalledWith(fromConfig);

    await user.click(screen.getByTestId("callback-actions-langfuse-failure"));
    expect(screen.queryByTestId("callback-config-owned-hint")).not.toBeInTheDocument();
    await user.click(await screen.findByTestId("callback-action-delete"));
    expect(onDelete).toHaveBeenCalledWith(fromDatabase);
  });

  // Regression: `/get_callbacks` returns the same `name` twice when a
  // callback is registered for both success and failure (e.g. `generic_api`
  // → POST to spend-log on both 200 and 4xx/5xx). The UI used to ignore
  // the `type` field and render every row as "Success", masking the
  // failure registration. Reading `record.type` fixes the badge AND
  // composing the row id with type avoids React's duplicate-key warning.
  it("renders distinct Success and Failure badges for same-name dual registration", () => {
    render(
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
});
