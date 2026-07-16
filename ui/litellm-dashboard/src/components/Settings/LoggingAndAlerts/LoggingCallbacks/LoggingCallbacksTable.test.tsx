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

  it("should render", () => {
    render(<LoggingCallbacksTable callbacks={[]} availableCallbacks={{}} />);
    expect(screen.getByText("Active Logging Callbacks")).toBeInTheDocument();
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

  it("should call onAdd when the Add Callback button is clicked", async () => {
    const user = userEvent.setup();
    const onAdd = vi.fn();
    render(<LoggingCallbacksTable callbacks={[]} availableCallbacks={{}} onAdd={onAdd} />);
    await user.click(screen.getByRole("button", { name: /add callback/i }));
    expect(onAdd).toHaveBeenCalled();
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

  it("renders a global destination's scope and manual assignment mode", () => {
    render(
      <LoggingCallbacksTable
        callbacks={[
          {
            name: "langfuse-eu",
            variables: baseVars,
            credentialName: "langfuse-eu",
            access: { global: true },
            resolvedScope: { global: true, teams: [], orgs: [] },
          },
        ]}
        availableCallbacks={{}}
      />,
    );
    expect(screen.getByText("Global access")).toBeInTheDocument();
    expect(screen.getByText("Manual assignment")).toBeInTheDocument();
    expect(screen.queryByText("Success")).not.toBeInTheDocument();
  });

  it("renders a scoped destination's resolved teams and orgs", () => {
    render(
      <LoggingCallbacksTable
        callbacks={[
          {
            name: "arize-eu",
            variables: baseVars,
            credentialName: "arize-eu",
            access: { teams: ["t1", "t2"], orgs: ["o1"] },
            resolvedScope: { global: false, teams: ["t1", "t2"], orgs: ["o1"] },
          },
        ]}
        availableCallbacks={{}}
      />,
    );
    expect(screen.getByText("team: t1")).toBeInTheDocument();
    expect(screen.getByText("team: t2")).toBeInTheDocument();
    expect(screen.getByText("org: o1")).toBeInTheDocument();
  });

  it("renders auto-enable mode for a destination", () => {
    render(
      <LoggingCallbacksTable
        callbacks={[
          {
            name: "otel-auto",
            variables: baseVars,
            credentialName: "otel-auto",
            access: { teams: ["t1"] },
            autoEnable: true,
            resolvedScope: { global: false, teams: ["t1"], orgs: [] },
          },
        ]}
        availableCallbacks={{}}
      />,
    );
    expect(screen.getByText("Auto-enabled")).toBeInTheDocument();
  });

  it("a destination row edits access and deletes without exposing callback actions", async () => {
    const user = userEvent.setup();
    const onEditAccess = vi.fn();
    const onDelete = vi.fn();
    const onTest = vi.fn();
    const callback = {
      name: "dest",
      variables: baseVars,
      credentialName: "dest",
      access: { global: true },
      resolvedScope: { global: true, teams: [], orgs: [] },
    };
    render(
      <LoggingCallbacksTable
        callbacks={[callback]}
        availableCallbacks={{}}
        onEditAccess={onEditAccess}
        onDelete={onDelete}
        onTest={onTest}
      />,
    );

    await user.click(screen.getByTestId("callback-actions-dest-success"));
    expect(screen.queryByTestId("callback-action-test")).not.toBeInTheDocument();
    expect(screen.queryByTestId("callback-action-edit")).not.toBeInTheDocument();
    await user.click(await screen.findByTestId("destination-action-edit-access"));
    expect(onEditAccess).toHaveBeenCalledWith(callback);

    await user.click(screen.getByTestId("callback-actions-dest-success"));
    await user.click(await screen.findByTestId("destination-action-delete"));
    expect(onDelete).toHaveBeenCalledWith(callback);
    expect(onTest).not.toHaveBeenCalled();
  });

  it("a config callback row renders an empty scope", () => {
    render(
      <LoggingCallbacksTable
        callbacks={[{ name: "datadog", type: "success", variables: baseVars }]}
        availableCallbacks={{}}
      />,
    );
    const row = screen.getByText("datadog").closest("tr");
    expect(row).not.toBeNull();
    expect(within(row as HTMLElement).getByText("—")).toBeInTheDocument();
  });
});
