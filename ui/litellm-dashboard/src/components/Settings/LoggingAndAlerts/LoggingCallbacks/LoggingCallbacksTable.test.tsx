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

  it("renders a global destination's scope", () => {
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

  it("gives a destination and a config callback of the same name distinct row ids", () => {
    const errors: string[] = [];
    const spy = vi.spyOn(console, "error").mockImplementation((...args: unknown[]) => {
      errors.push(args.map(String).join(" "));
    });

    render(
      <LoggingCallbacksTable
        callbacks={[
          { name: "arize", type: "success", variables: baseVars },
          {
            name: "arize",
            variables: baseVars,
            credentialName: "arize",
            access: { global: true },
            resolvedScope: { global: true, teams: [], orgs: [] },
          },
        ]}
        availableCallbacks={{}}
      />,
    );

    expect(errors.filter((e) => /same key/i.test(e))).toHaveLength(0);
    spy.mockRestore();
  });
});

describe("read-only admin actions", () => {
  // Regression: readOnly dropped the whole actions column, so an Admin Viewer lost Test
  // on pre-existing callback rows even though that role is backend-authorized for
  // /health/services. Only the mutating actions belong behind readOnly.
  const row = { name: "langfuse", variables: { ...baseVars }, type: "success" as const };

  it("keeps Test and hides the mutating actions for a read-only admin", async () => {
    const user = userEvent.setup();
    render(<LoggingCallbacksTable callbacks={[row]} readOnly />);

    await user.click(screen.getByTestId("callback-actions-langfuse-success"));

    expect(await screen.findByTestId("callback-action-test")).toBeInTheDocument();
    expect(screen.queryByTestId("callback-action-edit")).toBeNull();
    expect(screen.queryByTestId("callback-action-delete")).toBeNull();
  });

  it("keeps every action for a full admin", async () => {
    const user = userEvent.setup();
    render(<LoggingCallbacksTable callbacks={[row]} />);

    await user.click(screen.getByTestId("callback-actions-langfuse-success"));

    expect(await screen.findByTestId("callback-action-test")).toBeInTheDocument();
    expect(screen.getByTestId("callback-action-edit")).toBeInTheDocument();
    expect(screen.getByTestId("callback-action-delete")).toBeInTheDocument();
  });
});

describe("destination rows must not overstate what a destination does", () => {
  const destination = (over: Record<string, unknown> = {}) => ({
    name: "d1",
    variables: baseVars,
    credentialName: "d1",
    destinationLabel: "Generic OTLP Collector",
    resolvedScope: { global: true, teams: [], orgs: [] },
    ...over,
  });

  it("shows Not active, never a scope badge, when the backend cannot build the destination", () => {
    // Regression: the cell read credential_info.access alone, so a destination the
    // resolver excludes (no backend name, or values its adapter rejects) still rendered
    // "Global access" and read as live.
    render(
      <LoggingCallbacksTable
        callbacks={[destination({ resolvesToDestination: false }) as never]}
        availableCallbacks={{}}
      />,
    );
    expect(screen.getByText("Not active")).toBeInTheDocument();
    expect(screen.queryByText("Global access")).not.toBeInTheDocument();
  });

  it("still shows the scope badge when the destination does build", () => {
    render(
      <LoggingCallbacksTable
        callbacks={[destination({ resolvesToDestination: true }) as never]}
        availableCallbacks={{}}
      />,
    );
    expect(screen.getByText("Global access")).toBeInTheDocument();
    expect(screen.queryByText("Not active")).not.toBeInTheDocument();
  });

  it("keeps the admin's own name for a destination named after a config callback", () => {
    // Regression: the name column applied the callback registry's display label, so a
    // destination the admin called "datadog" rendered as "Datadog", indistinguishable
    // from the real Datadog callback row.
    render(
      <LoggingCallbacksTable
        callbacks={[destination({ name: "datadog", credentialName: "datadog" }) as never]}
        availableCallbacks={{
          datadog: { litellm_callback_name: "datadog", litellm_callback_params: [], ui_callback_name: "Datadog" },
        }}
      />,
    );
    expect(screen.getByText("datadog")).toBeInTheDocument();
    expect(screen.queryByText("Datadog")).not.toBeInTheDocument();
  });

  it("renders no actions trigger for a read-only admin, rather than one that opens empty", () => {
    // Regression: readOnly suppressed both Edit scope and Delete, and destinations never
    // get Test, so the trigger opened a menu with zero items and looked broken.
    render(<LoggingCallbacksTable callbacks={[destination() as never]} availableCallbacks={{}} readOnly />);
    expect(screen.queryByTestId("callback-actions-d1-success")).not.toBeInTheDocument();
  });

  it("still renders the actions trigger for a config callback a read-only admin can Test", async () => {
    render(
      <LoggingCallbacksTable
        callbacks={[{ name: "datadog", type: "success_and_failure", variables: baseVars }]}
        availableCallbacks={{}}
        readOnly
      />,
    );
    const trigger = screen.getByTestId("callback-actions-datadog-success_and_failure");
    await userEvent.click(trigger);
    expect(await screen.findByTestId("callback-action-test")).toBeInTheDocument();
  });
});
